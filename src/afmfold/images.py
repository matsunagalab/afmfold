# This implementation was adapted and modified based on:
# Y. Matsunaga, S. Fuchigami, T. Ogane, and S. Takada,
# "End-to-end differentiable blind tip reconstruction for noisy atomic force microscopy images,"
# Sci. Rep. 13, 129 (2023). https://doi.org/10.1038/s41598-022-27057-2
# Original implementation available at:
# https://github.com/matsunagalab/ColabBTR/blob/main/colabbtr/morphology.py

import os
import glob
import math
import numpy as np
import torch
from skimage.exposure import match_histograms
from tqdm import tqdm
import torch.nn.functional as F
from skimage.exposure import match_histograms

def sample_uniform_so3(n, device='cpu'):
    # Step 1: Sample unit quaternions (uniform on S^3)
    u1 = torch.rand(n, 1, device=device)
    u2 = torch.rand(n, 1, device=device)
    u3 = torch.rand(n, 1, device=device)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    q = torch.cat([q1, q2, q3, q4], dim=1)  # shape: (n, 4)

    # Step 2: Convert quaternion to rotation matrix
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2 * (q2**2 + q3**2),   2 * (q1 * q2 - q0 * q3),   2 * (q1 * q3 + q0 * q2),
        2 * (q1 * q2 + q0 * q3),   1 - 2 * (q1**2 + q3**2),   2 * (q2 * q3 - q0 * q1),
        2 * (q1 * q3 - q0 * q2),   2 * (q2 * q3 + q0 * q1),   1 - 2 * (q1**2 + q2**2)
    ], dim=-1).reshape(n, 3, 3)

    return R

def apply_rotations(xyz, rot_matrices):
    """
    xyz          : (X, N, 3) tensor
    rot_matrices : (Y, 3, 3) tensor (Y rotation matrices)

    Returns:
        rotated : (X, Y, N, 3) tensor
    """
    # (X, N, 3) → (X, 1, N, 3)
    xyz = xyz.unsqueeze(1)  # Make it broadcastable along rotation axis Y
    # (Y, 3, 3) → (1, Y, 3, 3)
    rot_matrices = rot_matrices.unsqueeze(0)  # Broadcast over samples X

    # (X, Y, N, 3) = (X, 1, N, 3) @ (1, Y, 3, 3).transpose(-1, -2) → (X, Y, N, 3)
    rotated = torch.matmul(xyz, rot_matrices.transpose(-1, -2))

    return rotated

def generate_landscape(coords, xedges, yedges):
    """
    Divide coordinates into a grid defined by xedges[b], yedges[b],
    and extract the maximum z within each grid cell while assigning the center coordinates.

    Args:
        coords (torch.Tensor): [..., B, N, 3] tensor.
        xedges (torch.Tensor): [B, W+1] tensor.
        yedges (torch.Tensor): [B, H+1] tensor.

    Returns:
        landscape (torch.Tensor): [..., B, H, W, 3] tensor.
    """
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)
    if isinstance(xedges, np.ndarray):
        xedges = torch.from_numpy(xedges)
    if isinstance(yedges, np.ndarray):
        yedges = torch.from_numpy(yedges)
    if xedges.ndim == 1:
        xedges = torch.tile(xedges[None,:], (len(coords), 1))
    if yedges.ndim == 1:
        yedges = torch.tile(yedges[None,:], (len(coords), 1))
    
    # Expand input shape
    *outer_shape, B, N, _ = coords.shape  # outer_shape ≔ additional batch dimensions
    outer_prod = int(torch.tensor(outer_shape).prod()) if outer_shape else 1
    
    W = xedges.shape[-1] - 1
    H = yedges.shape[-1] - 1
    device = coords.device

    # Reshape to [T, N, 3] (T = outer_prod * B)
    coords = coords.reshape(outer_prod * B, N, 3)  # [T, N, 3]

    # Broadcast xedges, yedges to T rows
    xedges = xedges.unsqueeze(0).expand(outer_prod, -1, -1).reshape(outer_prod * B, W + 1)  # [T, W+1]
    yedges = yedges.unsqueeze(0).expand(outer_prod, -1, -1).reshape(outer_prod * B, H + 1)  # [T, H+1]

    # Grid center coordinates
    x_centers = (xedges[:, :-1] + xedges[:, 1:]) / 2  # [T, W]
    y_centers = (yedges[:, :-1] + yedges[:, 1:]) / 2  # [T, H]

    # Assign x, y to grid cells
    x, y, z = coords.unbind(-1)  # each [T, N]

    x_bin = (x.unsqueeze(-1) >= xedges[:, :-1].unsqueeze(1)) & (x.unsqueeze(-1) <  xedges[:, 1:].unsqueeze(1))  # [T, N, W]
    y_bin = (y.unsqueeze(-1) >= yedges[:, :-1].unsqueeze(1)) & (y.unsqueeze(-1) <  yedges[:, 1:].unsqueeze(1))  # [T, N, H]

    valid = x_bin.any(-1) & y_bin.any(-1)  # [T, N]

    x_idx = x_bin.float().argmax(-1)  # [T, N]
    y_idx = y_bin.float().argmax(-1)  # [T, N]
    grid_idx = y_idx * W + x_idx  # [T, N]

    flat_idx = torch.arange(outer_prod * B, device=device).unsqueeze(1) * (H * W) + grid_idx                       # [T, N]
    flat_idx = flat_idx[valid].reshape(-1)  # Extract only valid ones

    z_flat = z[valid].reshape(-1)  # [N_valid]

    # Compute max z in each grid cell
    max_z = torch.full((outer_prod * B * H * W,), 0.0, device=device, dtype=z_flat.dtype)
    max_z = max_z.scatter_reduce(0, flat_idx, z_flat, reduce='amax', include_self=True)
    landscape = max_z.view(outer_prod, B, H, W)  # [T, H, W]
    return landscape

def fixed_padding(inputs, kernel_size, dilation=1, padding_type="idilation"):
    if padding_type == "idilation":
        pad_total = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
    elif padding_type == "ierosion":
        pad_total = dilation * (kernel_size - 1)
    else:
        raise ValueError("padding_type must be either 'idilation' or 'erosion'")
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    # F.pad uses (left, right, top, bottom) order for 2-D
    return F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end), mode='constant', value=0)

def ierosion(surface, tip):
    """
    Compute the erosion of batched images by a single tip (structuring element).
    
    Args:
        surface: Tensor of shape (B, H, W)  -- a batch of grayscale images
        tip: Tensor of shape (K, K)         -- the structuring element (kernel)
    
    Returns:
        Tensor of shape (B, H, W) -- eroded result for each image in the batch
    """
    B, H, W = surface.shape
    kernel_size, _ = tip.shape

    x = surface.unsqueeze(1)  # (B, 1, H, W)
    x = fixed_padding(x, kernel_size, dilation=1, padding_type="ierosion")  # (B, 1, H+pad, W+pad)
    x = F.unfold(x, kernel_size=kernel_size, dilation=1, padding=0, stride=1)  # (B, K*K, H*W)

    x = x.unsqueeze(1)  # (B, 1, K*K, H*W)
    
    # Prepare kernel
    weight = tip.view(1, 1, -1, 1)  # (1, 1, K*K, 1)
    
    x = weight - x  # (B, 1, K*K, H*W)
    x, _ = torch.max(x, dim=2, keepdim=False)  # (B, 1, H*W)
    x = -1 * x
    x = x.view(B, 1, H, W)  # (B, 1, H, W)
    
    return x.squeeze(1)  # (B, H, W)

def idilation(image, tip):
    """
    Morphological dilation on a batch of height-maps.

    Args
    ----
    image : (B, H, W) tensor – batch of input surfaces
    tip   : (K, K) tensor – structuring element (probe tip)

    Returns
    -------
    (B, H, W) tensor – dilated surfaces
    """
    if image.dim() != 3:
        raise ValueError("image must be (B, H, W)")
    B, H, W = image.shape
    K = tip.shape[0]          # assume square tip

    # Convert to (B, 1, H, W) and add channel=1
    x = image.unsqueeze(1)

    # Padding for valid receptive field
    x = fixed_padding(x, K, dilation=1, padding_type="idilation")

    # Use unfold to extract KxK patches around each pixel center
    # Returns (B, 1*K*K, H*W)
    x = F.unfold(x, kernel_size=K, dilation=1, padding=0, stride=1)

    # Expand weights (tip) and broadcast-add
    # weight: (1, 1, K*K, 1)
    weight = tip.view(1, 1, -1, 1)
    # x: (B, 1, K*K, H*W)
    x = x.unsqueeze(1) + weight

    # Take max along K*K dimension → (B, 1, H*W)
    x, _ = x.max(dim=2)

    # Reshape to original spatial dimensions and drop channel
    x = x.view(B, H, W)
    return x

def generate_tip_shape(radius, angle, device="cpu"):
    """
    Generate a shape with a hemisphere + outer taper (slope).

    Args:
        radius (float): radius of hemisphere
        angle (float): slope angle (degrees) outside the radius

    Returns:
        torch.Tensor: height tensor of shape (dim, dim)
    """
    angle_rad = angle / 180 * math.pi
    dim = int(math.ceil(radius) * 3)
    center = (dim - 1) / 2  # Ensure pixel center is aligned with grid center
    tangent = math.tan(0.5 * math.pi - angle_rad)

    # Grid coordinates (dim, dim)
    y = torch.arange(dim).float()
    x = torch.arange(dim).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dx = xx - center
    dy = yy - center
    r = torch.sqrt(dx**2 + dy**2).to(device)  # Distance from each cell center to the center point

    # Empty height tensor
    height = torch.zeros_like(r).to(device)

    # Inside hemisphere: height = sqrt(radius^2 - r^2)
    mask_inside = r <= radius
    height[mask_inside] = torch.sqrt(radius**2 - r[mask_inside]**2)

    # Outside hemisphere: height = - (r - radius) * tan(angle)
    mask_outside = r > radius
    deltas = r[mask_outside] - radius
    height[mask_outside] = -deltas * tangent

    # Normalize so that min = 0
    height -= height.min()

    return height.to(device)

def add_noise(real_images, pseudo_images, r=1.0, noise_threshold=0.0, seed=None):
    # Convert to numpy
    inputs_is_tensor = False
    if isinstance(real_images, torch.Tensor) and isinstance(pseudo_images, torch.Tensor):
        device = real_images.device
        real_images = real_images.detach().cpu().numpy().copy()
        pseudo_images = pseudo_images.detach().cpu().numpy().copy()
        inputs_is_tensor = True
    assert isinstance(real_images, np.ndarray) and isinstance(pseudo_images, np.ndarray), "Inputs must be numpy arrays or tensors."
    
    rng = np.random.default_rng(seed)
    
    # Normalize means
    real_images = real_images.copy()
    pseudo_images = pseudo_images.copy()
    
    real_images -= np.median(real_images)
    pseudo_images -= np.median(pseudo_images, axis=(1,2))[:,None,None]
    
    # Ignore noisy parts in pseudo AFM
    pseudo_images[pseudo_images < noise_threshold] = 0.0
    
    # Create reference histogram (> +σ) (same process)
    neg = real_images[real_images < 0]
    sigma = np.sqrt((neg**2).mean())
    ref = real_images[real_images > (r * sigma)]
    
    # Histogram matching (excluding substrate=0, apply in batch)
    mask = pseudo_images > 0
    matched = pseudo_images.copy()
    matched[mask] = match_histograms(pseudo_images[mask], ref, channel_axis=None)
    matched[~mask] = 0.0
    
    # Noise generation
    #    ─ line offset + per-pixel Gaussian noise
    N, H, W = matched.shape
    
    # Broadcast row offsets to each frame
    random_noise  = rng.normal(0.0, sigma, size=(N, H, W))
    
    # Combine
    matched_noise = matched + random_noise
    
    if inputs_is_tensor:
        matched_noise = torch.from_numpy(matched_noise).to(device)
        matched = torch.from_numpy(matched).to(device)
        ref = torch.from_numpy(ref).to(device)
        
    return (ref, matched), matched_noise

def generate_images(
    traj, resolution_nm, width, height, epochs, dataset_size, 
    distance=None, batch_size=1, min_z=0.0, noise_nm=0.0,
    max_tip_radius=1.0, min_tip_radius=3.0, max_tip_angle=10.0, min_tip_angle=30.0,
    ref_images=None, is_tqdm=True, match_histgram=False, save_dir=None, device="cuda",
    ):
    device = torch.device(device)
    
    # Load trajectory
    xyz = torch.from_numpy(traj.xyz).to(device)
    xedges = resolution_nm * torch.arange(-int(width/2) - 1, width - int(width/2), device=device)
    yedges = resolution_nm * torch.arange(-int(height/2) - 1, height - int(height/2), device=device)
    
    # Load labels
    if distance is None:
        ref_labels = None
    else:
        ref_labels = torch.from_numpy(distance).to(device)
    
    # Convert reference images to torch
    if ref_images is not None:
        if isinstance(ref_images, np.ndarray):
            ref_images = torch.from_numpy(ref_images).to(device)
        elif isinstance(ref_images, torch.Tensor):
            ref_images = ref_images.to(device)
        else:
            raise ValueError("ref_images must be either numpy array or torch tensor.")
    
    # Generate images
    output_image_list = []
    output_label_list = []
    for epoch in tqdm(range(epochs), disable=is_tqdm):
        batch_image_list = []
        batch_label_list = []
        for step in tqdm(range(math.ceil(dataset_size / (len(xyz) * batch_size))), disable=not is_tqdm, desc=f"Epoch {epoch+1}/{epochs}"):
            # Apply rotations
            rots = sample_uniform_so3(batch_size, device=device)
            rotated = apply_rotations(xyz, rots)  # [Nframe, Nrot, N, 3]
            
            # Rotate labels accordingly
            rotated = rotated.reshape((-1, *rotated.shape[-2:]))  # [Nframe * Nrot, N, 3]
            if ref_labels is not None:
                labels = torch.tile(ref_labels[:,None,:], (1,batch_size,1))  # [Nframe, Nrot, Ndistance]
                labels = labels.reshape((-1, labels.shape[-1]))  # [Nframe * Nrot, Ndistance]
            
            # Align by center of mass, then translate
            rotated_com = torch.mean(rotated, dim=-2, keepdim=True)
            translated = rotated - rotated_com

            # Adjust z coordinates
            z_unit = torch.tensor([[[0.0, 0.0, 1.0]]]).to(device)
            min_coord, _ = torch.min(translated, dim=-2, keepdim=True)
            translated = translated + (- min_coord + min_z) * z_unit
            
            # Create pseudo-AFM image
            pure_image = generate_landscape(translated, xedges, yedges)
            pure_image = pure_image.reshape((-1, height, width))

            # Generate pseudo tip
            tip_radius = torch.rand((1,)).to(device) * (max_tip_radius - min_tip_radius) + min_tip_radius
            tip_angle = torch.rand((1,)).to(device) * (max_tip_angle - min_tip_angle) + min_tip_angle
            tip = generate_tip_shape(radius=tip_radius, angle=tip_angle, device=device)
            del tip_radius, tip_angle
        
            # Dilate pseudo-AFM image
            afm_image = idilation(pure_image, tip)

            # Add noise
            if noise_nm > 0.0:
                noise = np.random.normal(0.0, noise_nm, afm_image.shape)
                noise = torch.from_numpy(noise).to(afm_image.dtype).to(device)
                afm_image += noise
                
            # Match image histogram to reference images
            if match_histgram and ref_images is not None:
                _, afm_image = add_noise(ref_images, afm_image, r=1.0, noise_threshold=0.0, seed=step)
            
            # Convert images/labels to numpy and store
            batch_image_list.append(afm_image.detach().cpu().numpy())
            if ref_labels is not None:
                batch_label_list.append(labels.detach().cpu().numpy())
        
        # Concatenate generated images and labels
        batch_images = np.concatenate(batch_image_list, axis=0)
        batch_images = batch_images[0:dataset_size]  # Truncate to dataset_size
        
        if len(batch_label_list) > 0:
            assert len(batch_label_list) == len(batch_image_list), f"Mismatch in number of batches between images and labels: {len(batch_label_list)} != {len(batch_image_list)}."
            batch_labels = np.concatenate(batch_label_list, axis=0)
            batch_labels = batch_labels[0:dataset_size]  # Truncate to dataset_size
        
        if save_dir is not None:
            # Determine filenames for saving
            max_index = max([int(p.split("image_")[-1].split(".npy")[0]) for p in glob.glob(os.path.join(save_dir, "image_*.npy"))], default=-1)
            save_image_path = os.path.join(save_dir, f"image_{max_index + 1}.npy")
            save_label_path = os.path.join(save_dir, f"label_{max_index + 1}.npy")
        
            np.save(save_image_path, batch_images)
            if len(batch_label_list) > 0:
                np.save(save_label_path, batch_labels)
        
        output_image_list.append(batch_images)
        if len(batch_label_list) > 0:
            output_label_list.append(batch_labels)
    
    output_images = np.concatenate(output_image_list, axis=0)
    if len(output_label_list) > 0:
        assert len(output_label_list) == len(output_image_list), f"Mismatch in number of batches between images and labels: {len(output_label_list)} != {len(output_image_list)}."
        output_labels = np.concatenate(output_label_list, axis=0)
    else:
        output_labels = None
    
    return output_images, output_labels
