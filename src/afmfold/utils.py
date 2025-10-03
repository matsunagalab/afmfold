import os
import numpy as np
import logging
import json
import warnings
import mdtraj as md
from contextlib import contextmanager, redirect_stdout, redirect_stderr

from afmfold.domain import compute_domain_distance

def move_all_tensors_in_device(*tensors, device="cpu"):
    output = []
    for t in tensors:
        output.append(t.to(device))
    return output

@contextmanager
def suppress_output(suppress=True):
    """
    Context manager to mute stdout, stderr, logging, warnings,
    and even low-level C extension outputs.
    """
    if not suppress:
        # Do nothing, just yield
        yield
        return
    
    # ---------- Low-level: save original fds ----------
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    devnull_fd      = os.open(os.devnull, os.O_RDWR)

    # Redirect 1,2 -> /dev/null
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    # ---------- High-level: logging & warnings ----------
    prev_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)         # Disable all loggers
    warnings_filters_copy = warnings.filters[:]

    try:
        with open(os.devnull, 'w') as dn, \
             redirect_stdout(dn), \
             redirect_stderr(dn), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        # ---------- Restore fds ----------
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        for fd in (saved_stdout_fd, saved_stderr_fd, devnull_fd):
            os.close(fd)

        # ---------- Restore logging & warnings ----------
        logging.disable(prev_logging_disable)
        warnings.filters = warnings_filters_copy

def compute_rmsd_single_frame(t1, t2, atom_indices=None):
    """
    Args:
        t1, t2 : md.Trajectory. Two trajectory objects, each must have exactly one frame.
        atom_indices : list of int, optional. Indices of atoms to use for alignment and RMSD calculation. If None, use all atoms.

    Returns:
        rmsd_val: float. RMSD value (nm)
    """
    # 1. Check number of frames
    if t1.n_frames != 1 or t2.n_frames != 1:
        raise ValueError("Both trajectories must have exactly one frame.")

    # 2. Set atom_indices (use all atoms if None)
    if atom_indices is None:
        atom_indices = np.arange(t1.n_atoms)

    # 3. Perform alignment (if necessary)
    t2_aligned = t2.superpose(t1, frame=0, atom_indices=atom_indices)
    
    # 4. Compute RMSD
    rmsd_val = md.rmsd(t2_aligned, t1, atom_indices=atom_indices)[0]

    return rmsd_val

def add_arg_to_json(json_path, new_dict, out_path=None):
    # Load JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Add new arguments to the dictionary
    for key, value in new_dict.items():
        data[key] = value

    # Determine output path
    if out_path is None:
        out_path = json_path

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data

def k_nearest_weighted_average(
    key_arr: np.ndarray,      # (N, D)
    value_arr: np.ndarray,    # (N, D)
    target_arr: np.ndarray,   # (M, D)
    k: int,
    *,
    weight_mode: str = "inverse", # "boltzmann" | "softmax" | "inverse"
    temperature: float = 1.0,        # for softmax
    power: float = 1.0,              # for inverse-distance: 1/r^power
    eps: float = 1e-12,
    d_threshold = 3.0,
):
    """
    Returns:
        closest_value:(M, D)  Weighted average of value
    """
    if target_arr.ndim == 1:
        target_arr = target_arr[None,:]
        
    # 1) Distance matrix (Euclidean)
    #  Use ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b for stability and speed
    a2 = np.sum(target_arr**2, axis=1, keepdims=True)       # (M,1)
    b2 = np.sum(key_arr**2, axis=1, keepdims=True).T        # (1,N)
    cross = target_arr @ key_arr.T                           # (M,N)
    dist2 = np.maximum(a2 + b2 - 2.0 * cross, 0.0)
    distance_mat = np.sqrt(dist2, dtype=dist2.dtype)         # (M,N)
    
    M, N = distance_mat.shape
    if not (1 <= k <= N):
        raise ValueError(f"k must be in [1, N={N}]")

    # 2) Get indices of top-k smallest distances in each row
    #    Use np.argpartition to extract in O(N), then sort the k results
    part = np.argpartition(distance_mat, kth=k-1, axis=1)[:, :k]  # (M,k)
    part_dist = np.take_along_axis(distance_mat, part, axis=1)    # (M,k)
    order_in_part = np.argsort(part_dist, axis=1)                 # (M,k)
    neighbor_ind = np.take_along_axis(part, order_in_part, axis=1)  # (M,k)
    neighbor_dist = np.take_along_axis(distance_mat, neighbor_ind, axis=1)  # (M,k)
    
    # 3) Compute weights (row-wise normalization so that sum=1)
    if weight_mode == "boltzmann":
        if temperature <= 0:
            raise ValueError("temperature (tau) must be > 0 for boltzmann weighting.")

        # E = D^2 / (2 sigma^2)
        E = (neighbor_dist ** 2) / 2.0
        logits = -E / temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)  # Stabilize
        w = np.exp(logits)
        weight = w / (np.sum(w, axis=1, keepdims=True) + eps)
            
    elif weight_mode == "softmax":
        # Closer distances get higher weights → softmax(-distance / temperature)
        if temperature <= 0:
            raise ValueError("temperature must be > 0 for softmax weighting.")
        logits = -neighbor_dist**2 / temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)  # Stabilize
        w = np.exp(logits)
        weight = w / (np.sum(w, axis=1, keepdims=True) + eps)
        
    elif weight_mode == "inverse":
        # w_i ∝ 1 / (d_i^power + eps). If zero distance exists, assign one-hot
        zero_mask = neighbor_dist <= eps
        any_zero = np.any(zero_mask, axis=1, keepdims=True)  # (M,1)
        inv = 1.0 / (np.power(neighbor_dist, power) + eps)
        weight = inv / (np.sum(inv, axis=1, keepdims=True) + eps)
        # Replace with one-hot if zero distance exists (first occurrence set to 1)
        if np.any(any_zero):
            first_zero_pos = np.argmax(zero_mask, axis=1)                 # (M,)
            rows = np.arange(M)
            weight[any_zero.ravel()] = 0.0
            weight[rows[any_zero.ravel()], first_zero_pos[any_zero.ravel()]] = 1.0
    else:
        raise ValueError("weight_mode must be 'softmax' or 'inverse'.")
    
    # 4) Weighted average of corresponding values
    #    value_arr[neighbor_ind] -> (M, k, D)
    gathered_vals = value_arr[neighbor_ind]                # (M,k,D)
    closest_value = np.sum(gathered_vals * weight[..., None], axis=1)  # (M,D)

    # 2.1) If all neighbors are too far
    outlier_mask = np.all(neighbor_dist > d_threshold, axis=1)
    closest_value[outlier_mask] = target_arr[outlier_mask]
    
    return closest_value
