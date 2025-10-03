import os
import time
from datetime import datetime
import math
import numpy as np
import torch
import torch.nn.functional as F
import json
import glob
import mdtraj as md
from skimage.exposure import match_histograms
from scipy.ndimage import center_of_mass
from pprint import pprint
from tqdm import tqdm

from afmfold.images import generate_landscape, sample_uniform_so3, apply_rotations, generate_tip_shape, idilation, add_noise
from afmfold.utils import compute_rmsd_single_frame

def hat(v):
    """Skew map (R^3 -> so(3)).
    v: (..., 3)
    returns: (..., 3, 3)
    """
    x, y, z = v.unbind(dim=-1)
    O = torch.zeros_like(x)
    return torch.stack([
        torch.stack([ O, -z,  y], dim=-1),
        torch.stack([ z,  O, -x], dim=-1),
        torch.stack([-y,  x,  O], dim=-1),
    ], dim=-2)

def _taylor_sinc(x):
    # sinc(x) = sin(x)/x, stable around 0
    x2 = x * x
    return 1 - x2/6 + x2*x2/120 - x2*x2*x2/5040

def _taylor_omc_over_x2(x):
    # (1 - cos x) / x^2, stable around 0
    x2 = x * x
    return 0.5 - x2/24 + x2*x2/720 - x2*x2*x2/40320

def exp_so3(omega):
    """Exponential map from so(3) (axis-angle vector) to SO(3).
    omega: (..., 3) axis-angle vector
    returns: (..., 3, 3) rotation matrix
    """
    # Angle
    theta = torch.linalg.norm(omega, dim=-1)
    theta_expand2 = theta.unsqueeze(-1).unsqueeze(-1)
    K = hat(omega)  # (..., 3, 3)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(K.shape)

    # Safe coefficients
    eps = torch.finfo(omega.dtype).eps if omega.dtype.is_floating_point else 1e-8
    small = theta < 1e-4

    # Compute sinc and (1-cos)/theta^2 with stable series near 0
    sinc = torch.empty_like(theta)
    omc_over_t2 = torch.empty_like(theta)

    # Series for small angles
    sinc[small] = _taylor_sinc(theta[small])
    omc_over_t2[small] = _taylor_omc_over_x2(theta[small])

    # Direct for others
    th_safe = theta[~small] + eps
    sinc[~small] = torch.sin(th_safe) / th_safe
    omc_over_t2[~small] = (1 - torch.cos(th_safe)) / (th_safe * th_safe)

    sinc = sinc.unsqueeze(-1).unsqueeze(-1)
    omc_over_t2 = omc_over_t2.unsqueeze(-1).unsqueeze(-1)

    R = I + sinc * K + omc_over_t2 * (K @ K)
    return R

def sample_so3_gaussian(
    center,
    n,
    sigma=0.2,
    Sigma=None,
    seed=None,
):
    """Sample rotations on SO(3) around `center` via tangent-space Gaussian + exp.
    
    Args:
        center: (3,3) rotation matrix (torch.Tensor)
        n: number of samples
        sigma: isotropic std (radians) if Sigma is None
        Sigma: (3,3) SPD covariance in the tangent space (overrides sigma)
        seed: optional RNG seed for reproducibility
    Returns:
        rots: (n,3,3) sampled rotation matrices
    """
    assert center.shape == (3,3), "center must be (3,3)"
    device = center.device
    dtype = center.dtype if center.dtype.is_floating_point else torch.float64

    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    else:
        gen = None

    if Sigma is None:
        # isotropic covariance
        L = torch.eye(3, dtype=dtype, device=device) * sigma
    else:
        # Cholesky factor for general SPD covariance
        Sigma = Sigma.to(device=device, dtype=dtype)
        L = torch.linalg.cholesky(Sigma)

    eps = torch.randn((n, 3), dtype=dtype, device=device, generator=gen) @ L.T  # (n,3)
    R_delta = exp_so3(eps)                                                      # (n,3,3)
    center_expanded = center.to(dtype=dtype).unsqueeze(0).expand(n, -1, -1)
    rots = center_expanded @ R_delta
    return rots

def sample_so3_gaussian_batched(
    center: torch.Tensor,          # (B,3,3)
    n: int,
    sigma: float = 0.2,
    Sigma: torch.Tensor | None = None,  # (3,3) or (B,3,3)
    seed: int | None = None,
):
    """
    Sample rotations on SO(3) around each center[b] via tangent-space Gaussian + exp.

    Args:
        center: (B,3,3) rotation matrices (torch.Tensor)
        n: number of samples per batch item
        sigma: isotropic std (radians) when Sigma is None
        Sigma: (3,3) shared SPD covariance or (B,3,3) per-batch SPD covariance
        seed: RNG seed (optional)
    Returns:
        rots: (B,n,3,3) sampled rotation matrices
    """
    assert center.ndim == 3 and center.shape[-2:] == (3,3), "center must be (B,3,3)"
    B = center.shape[0]
    device = center.device
    dtype = center.dtype if center.dtype.is_floating_point else torch.float64

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    # tangent noise eps ~ N(0, Sigma) in R^3
    eps = torch.randn((B, n, 3), dtype=dtype, device=device, generator=gen)  # (B,n,3)

    if Sigma is None:
        eps = eps * float(sigma)
    else:
        Sigma = Sigma.to(device=device, dtype=dtype)
        if Sigma.ndim == 2:
            # shared covariance
            L = torch.linalg.cholesky(Sigma)                    # (3,3)
            eps = eps @ L.transpose(-1, -2)                     # (B,n,3)
        elif Sigma.ndim == 3:
            # per-batch covariance
            assert Sigma.shape == (B,3,3), "Sigma must be (3,3) or (B,3,3)"
            L = torch.linalg.cholesky(Sigma)                    # (B,3,3)
            eps = torch.einsum('bnc,bcf->bnf', eps, L.transpose(1,2))  # (B,n,3)
        else:
            raise ValueError("Sigma must be None, (3,3), or (B,3,3)")

    # Map to SO(3): exp_so3 expects (...,3) -> (...,3,3)
    R_delta = exp_so3(eps.reshape(B*n, 3)).reshape(B, n, 3, 3)  # (B,n,3,3)

    # Left-multiply around each center
    C = center.to(dtype=dtype).unsqueeze(1).expand(B, n, 3, 3)  # (B,n,3,3)
    rots = C @ R_delta                                          # (B,n,3,3)
    return rots

def compute_correlation_coeﬃcient(image1, image2):
    """
    Args:
        image1: (F, B1, H, W) or (B1, H, W). 後者の場合、F = 1として扱う.
        image2: (F, B2, H, W) or (B2, H, W). 後者の場合、F = 1として扱う.
        
    Returns:
        cc: (F, B1, B2).
    """
    # 形状の確認
    if isinstance(image1, np.ndarray) and isinstance(image2, np.ndarray):
        image1 = torch.from_numpy(image1)
        image2 = torch.from_numpy(image2)
        is_numpy = True
    else:
        is_numpy = False
        
    image1 = image1.reshape((-1, *image1.shape[-3:]))
    image2 = image2.reshape((-1, *image2.shape[-3:]))
    F1, B1, H1, W1 = image1.shape
    F2, B2, H2, W2 = image2.shape
    assert F1 == F2
    assert H1 == H2
    assert W1 == W2
    F, H, W = F1, H1, W1
    image1 = image1.reshape((F, B1, 1, H, W))
    image2 = image2.reshape((F, 1, B2, H, W))
    
    # cc の計算
    cc = torch.sum(image1 * image2, dim=(-2,-1)) / torch.sqrt(torch.sum(image1**2, dim=(-2,-1))) / torch.sqrt(torch.sum(image2**2, dim=(-2,-1)))  # [F, B1, B2]
    
    if is_numpy:
        cc = cc.detach().cpu().numpy()
    return cc

class RigidBodyFitting:
    def __init__(
        self, target_image, traj, steps, 
        resolution_nm=0.98, min_z=None, translation_range=(-3, 3), 
        translation_batch=10, rot_batch=32, prove_radius=4.2, prove_angle=20,
        ref_pdb=None, log_interval=1, dry_run=False, match_histgram=False, device="cpu",
        ):
        # 入力をインスタンス化
        if isinstance(target_image, np.ndarray):
            target_image = target_image.reshape((-1, *target_image.shape[-2:]))
            self.target_image = torch.from_numpy(target_image).to(device)
        else:
            target_image = target_image.reshape((-1, *target_image.shape[-2:]))
            self.target_image = target_image.to(device)
        self.traj = traj
        self.steps = steps
        self.resolution_nm = resolution_nm
        self.min_z = min_z
        self.translation_range = translation_range
        self.translation_batch = translation_batch
        self.rot_batch = rot_batch
        self.prove_radius = prove_radius
        self.prove_angle = prove_angle
        self.ref_pdb = ref_pdb
        self.log_interval = log_interval
        self.dry_run = dry_run
        self.match_histgram = match_histgram
        self.device = device
        assert len(traj) == len(self.target_image)
        
        # ref_pdb に合わせてアライン
        if ref_pdb is not None:
            ref_traj = md.load(ref_pdb)
            ref_traj = ref_traj.atom_slice(ref_traj.topology.select("element != H"))
            traj.superpose(ref_traj, frame=0)
            del ref_traj
            
        # 初期設定
        self.num_frames, self.H, self.W = target_image.shape
        assert self.num_frames == len(traj)
        self.xyz = torch.from_numpy(traj.xyz).to(device)
        self.center = resolution_nm * torch.cat([torch.tensor(list(center_of_mass(self.target_image[i].detach().cpu().numpy())) + [0.0,]).reshape(-1, 3) for i in range(len(self.target_image))], dim=0).to(device)  # [F, 3]
        self.xedges = - 0.5 * resolution_nm + resolution_nm * torch.arange(self.W + 1, device=device)
        self.xgrid = resolution_nm * torch.arange(self.W, device=device)
        self.yedges = - 0.5 * resolution_nm + resolution_nm * torch.arange(self.H + 1, device=device)
        self.ygrid = resolution_nm * torch.arange(self.H, device=device)
        self.target_minvalue = torch.min(self.target_image)
        self.target_maxvalue = torch.max(self.target_image)
        self.target_medianvalue = torch.median(self.target_image)
        self.log = {}
        
        # 並進グリッドの作成
        delta = resolution_nm * torch.linspace(translation_range[0], translation_range[1], translation_batch)
        X, Y = torch.meshgrid(delta, delta, indexing='ij')  # shape: (5, 3) each
        grid = torch.stack((X, Y), dim=-1)
        self.trans = torch.cat([grid.reshape(-1,2), torch.zeros((translation_batch**2, 1))], axis=1).to(device)  # [Btrans, 3]
        
    def fit_translation(self, xyz):
        com = torch.mean(xyz, dim=-2, keepdim=True)  # [..., N, 3] => [..., 1, 3]
        translated = xyz - com + self.center.reshape([-1,] + [1 for _ in range(xyz.ndim - 2)] + [3,])  # [..., N, 3]
        return translated
    
    def sample(self, is_tqdm=True, desc=None):
        initial_time = time.time()
        for step in tqdm(range(math.ceil(self.steps / self.rot_batch)), disable=not is_tqdm, desc=desc):
            # 回転させる
            rots = sample_uniform_so3(self.rot_batch, device=self.device)  # [Brot, 3, 3]
            rotated = apply_rotations(self.xyz, rots)  # [F, Brot, N, 3]
            
            # 画像と重心を合わせてから並進させる
            centered = self.fit_translation(rotated).unsqueeze(1)  # [F, 1, Brot, N, 3]
            delta = self.trans.unsqueeze(1).unsqueeze(1).unsqueeze(0)  # [Btrans, 3] => [1, Btrans, 1, 1, 3]
            translated = (centered + delta).reshape((-1, *self.xyz.shape[-2:]))  # [F*Btrans*Brot, N, 3]

            # z座標を調整
            if self.min_z is not None:
                # z座標を調節する
                z_unit = torch.tensor([[[0.0, 0.0, 1.0]]]).to(self.device)
                min_coord, _ = torch.min(translated, dim=-2, keepdim=True)
                translated = translated + (- min_coord + self.min_z) * z_unit
            
            # 擬似AFM画像を作成
            pseudo_image = self.pseudo_afm(translated)  # [F*Btrans*Brot, H, W]

            # 画像をマッチさせる
            if self.match_histgram:
                (_, matched_image_np), _ = add_noise(self.target_image.unsqueeze(0).detach().cpu().numpy(), pseudo_image.detach().cpu().numpy())
                matched_image = torch.from_numpy(matched_image_np).to(self.device)  # [F*Btrans*Brot, H, W]
            else:
                min_values, _ = torch.min(pseudo_image.reshape(len(pseudo_image), -1), dim=1, keepdim=True)
                max_values, _ = torch.max(pseudo_image.reshape(len(pseudo_image), -1), dim=1, keepdim=True)
                max_values, min_values = max_values.unsqueeze(-1), min_values.unsqueeze(-1)
                matched_image = (pseudo_image - min_values) / (max_values - min_values + 1e-6) * (self.target_maxvalue - self.target_medianvalue) + self.target_medianvalue  # [F*Btrans*Brot, H, W]
                
            # 相関を計算
            matched_image = matched_image.reshape((self.num_frames, -1, *matched_image.shape[-2:]))  # [F, Btrans*Brot, H, W]
            cc = compute_correlation_coefficient(self.target_image.unsqueeze(1), matched_image)  # [F, 1, H, W] * [F, Btrans*Brot, H, W] -> [F, 1, Btrans*Brot]
            
            # 相関が最も高い並進インデックスを決定
            reshaped_cc = cc.reshape((self.num_frames, self.translation_batch**2, self.rot_batch))  # [F, Btrans, Brot]
            best_ccs, best_trans_index = torch.max(reshaped_cc, dim=-2)  # [F, Brot]
            
            # 相関が最も高い並進を抜き出す
            best_trans_index_4image = best_trans_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.H, self.W).unsqueeze(1)  # [F, 1, Brot, H, W]
            matched_image = matched_image.reshape((self.num_frames, self.translation_batch**2, self.rot_batch, *matched_image.shape[-2:]))  # [F, Btrans, Brot, H, W]
            best_images = torch.gather(matched_image, dim=1, index=best_trans_index_4image)  # [F, 1, Brot, H, W]
            best_images = best_images.squeeze(1)  # [F, Brot, H, W]
            
            translated = translated.reshape((self.num_frames, self.translation_batch**2, self.rot_batch, *self.xyz.shape[-2:]))# [F, Btrans, Brot, N, 3]
            best_trans_index_4crd = best_trans_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.xyz.shape[-2:]).unsqueeze(1)  # [F, 1, Brot, N, 3]
            best_translated = torch.gather(translated, dim=1, index=best_trans_index_4crd)  # [F, 1, Brot, N, 3]
            best_translated = best_translated.squeeze(1)  # [F, Brot, N, 3]
            assert best_ccs.shape == (self.num_frames, self.rot_batch), best_ccs.shape
            assert best_images.shape == (self.num_frames, self.rot_batch, self.H, self.W), best_images.shape
            assert best_translated.shape == (self.num_frames, self.rot_batch, *self.xyz.shape[-2:]), best_translated.shape
            
            # 結果を保存
            if step % self.log_interval == 0:
                self.log[step] = {"cc": best_ccs.detach().cpu().numpy(), "image": best_images.detach().cpu().numpy(), "R": rots.detach().cpu().numpy(), "coords": best_translated.detach().cpu().numpy()}
            
            if self.dry_run and time.time() - initial_time > 10:
                break
        
        summary = self.summarize_results()
        return summary
    
    def pseudo_afm(self, xyz):
        pure_image = generate_landscape(xyz, self.xedges, self.yedges)
        tip = generate_tip_shape(self.prove_radius, self.prove_angle, device=self.device)
        
        pure_image = pure_image.reshape((-1, self.H, self.W))
        afm_image = idilation(pure_image, tip)
        return afm_image
    
    def return_best(self):
        if len(self.log) == 0:
            return
        else:
            ccs = np.array([v["cc"] for v in self.log.values()])
            best_index = np.argmax(ccs)
            return self.log[best_index.item()]
        
    def summarize_results(self):
        # 保存対象がない場合
        if len(self.log) == 0:
            return {}
        
        ccs = np.concatenate([v["cc"] for v in self.log.values()], axis=1)  # [F, S]
        images = np.concatenate([v["image"] for v in self.log.values()], axis=1)  # [F, S, H, W]
        rots = np.concatenate([v["R"] for v in self.log.values()], axis=0)  # [S, 3, 3]
        coords = np.concatenate([v["coords"] for v in self.log.values()], axis=1)  # [F, S, N, 3]
        
        # 保存
        summary = {
            "cc": ccs,
            "images": images,
            "rots": rots,
            "coords": coords,
        }
        return summary
        
def save_args_to_file(args, json_path, **kwargs):
    # Namespace → dict に変換
    args_dict = vars(args)
    args_dict = {**args_dict, **kwargs}

    # JSON ファイルに保存（インデント付き）
    with open(json_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(f"Arguments saved to {json_path}")

def seconds_since(timestr):
    time_format = "%Y%m%d_%H%M%S"
    try:
        past_time = datetime.strptime(timestr, time_format)
        now = datetime.now()
        delta = now - past_time
        return int(delta.total_seconds())
    except ValueError as e:
        print("時刻の形式が正しくありません:", e)
        return None

def load_results(output_dirs, stop_at=None):
    # 結果の収集
    image_list = []
    true_xyz_list = []
    pred_xyz_list = []
    truth_list = []
    target_list = []
    restraint_list = []
    prediction_list = []
    for i in tqdm(range(len(output_dirs))):
        output_dir = output_dirs[i]
        cifs = glob.glob(os.path.join(output_dir, "predictions", "*.cif"))
        jsons = glob.glob(os.path.join(output_dir, "predictions", "*.json"))
        npzs = glob.glob(os.path.join(output_dir, "predictions", "*.npz"))
        assert all(len(flist) == 1 for flist in [cifs, jsons, npzs]), output_dir
        
        inputs = np.load(npzs[0])
        inputs = dict(inputs)
        
        image_list.append(inputs["image"][None,:,:])
        if "coords" in inputs:
            true_xyz_list.append(inputs["coords"][None,:,:])
        
        traj = md.load(cifs[0])
        pred_xyz_list.append(traj.xyz)
        
        with open(jsons[0], mode="r") as f:
            settings = json.load(f)
        
        if "truth" in settings:
            truth = np.array(settings["truth"])
            truth_list.append(truth[None,:])
        
        target = np.array(settings["target"])
        target_list.append(target[None,:])
        
        restraint = np.array(settings["restraint"])
        restraint_list.append(restraint[None,:])

        prediction = np.array(settings["prediction"])
        prediction_list.append(prediction[None,:])
        
        if stop_at is not None and (i+1) >= stop_at:
            break
        
    images = np.concatenate(image_list, axis=0)
    pred_xyzs = np.concatenate(pred_xyz_list, axis=0)
    if len(true_xyz_list) > 0:
        true_xyzs = np.concatenate(true_xyz_list, axis=0)
    else:
        true_xyzs = None
    if len(truth_list) > 0:
        truths = np.concatenate(truth_list, axis=0)
    else:
        truths = None
    targets = np.concatenate(target_list, axis=0)
    restraints = np.concatenate(restraint_list, axis=0)
    predictions = np.concatenate(prediction_list, axis=0)
    
    pred_traj = md.Trajectory(pred_xyzs, topology=traj.topology)
    if true_xyzs is not None:
        true_traj = md.Trajectory(true_xyzs, topology=traj.topology)
    else:
        true_traj = None
        
    return images, pred_traj, true_traj, truths, targets, restraints, predictions

def cat_data(data_dicts):
    if not data_dicts:
        return {}

    largest = sorted(data_dicts, key=lambda d: len(d))[-1]
    keys = list(largest.keys())
    out = {}

    for k in keys:
        values = [d[k] for d in data_dicts if k in d]
        out[k] = np.concatenate(values, axis=0)
    return out

def run_rigid_body_fitting(
    output_dirs, 
    ref_pdb, 
    steps=50000, 
    stop_at=None, 
    batchsize=20,
    resolution_nm=0.3, 
    prove_radius=2.0, 
    min_z=0.0,
    rot_batch=1,
    translation_range=(-5.0, 5.0),
    use_ref_structure=False,
    device="cuda",
    ):
    total_summary = {}
    images, pred_traj, true_traj, truths, targets, restraints, predictions = load_results(output_dirs, stop_at=stop_at)
    for i in range(math.ceil(len(images)/batchsize)):
        _ref_images = images[i*batchsize:(i+1)*batchsize]
        _pred_traj = pred_traj[i*batchsize:(i+1)*batchsize]
        if true_traj is not None:
            _true_traj = true_traj[i*batchsize:(i+1)*batchsize]
        if truths is not None:
            _truths = 0.1 * truths[i*batchsize:(i+1)*batchsize]
        _targets = 0.1 * targets[i*batchsize:(i+1)*batchsize]
        #_restraints = 0.1 * restraints[i*batchsize:(i+1)*batchsize]
        _predictions = 0.1 * predictions[i*batchsize:(i+1)*batchsize]
        
        if use_ref_structure:
            # それぞれの画像について、ref_trajを使った剛体フィッティングも実行する
            ref_traj = md.join([md.load(ref_pdb) for _ in range(len(_ref_images))])
            _ref_images = np.concatenate([_ref_images, _ref_images], axis=0)
            _target_traj = md.Trajectory(np.concatenate([_pred_traj.xyz, ref_traj.xyz], axis=0), topology=_pred_traj.topology)
        else:
            _target_traj = _pred_traj
            
        fitting = RigidBodyFitting(
            _ref_images, _target_traj, steps, 
            resolution_nm=resolution_nm, prove_radius=prove_radius, min_z=min_z, 
            ref_pdb=ref_pdb, rot_batch=rot_batch, translation_range=translation_range, device=device,
            )
        summary = fitting.sample()
        
        # Extract the best CC values in all rotations
        _ccs = summary["cc"]
        _best_rot_indices = np.argmax(_ccs, axis=1)
        _best_ccs = _ccs[np.arange(len(_ccs)), _best_rot_indices]
        _best_images = summary["images"][np.arange(len(_ccs)), _best_rot_indices]
        _best_rots = summary["rots"][_best_rot_indices]
        _best_coords = summary["coords"][np.arange(len(_ccs)), _best_rot_indices]
        
        if truths is not None:
            _sqerrors = np.sum((_truths - _targets)**2, axis=1)
        
        if true_traj is not None:
            _rmsds = np.zeros((len(_pred_traj),))
            for j in range(len(_pred_traj)):
                _rmsd = compute_rmsd_single_frame(_pred_traj[j], _true_traj[j])
                _rmsds[j] = _rmsd
        
        summary = {
            "all_cc": summary["cc"],
            "all_rots": summary["rots"],
            "cc": _best_ccs,
            "rots": _best_rots,
            "ref_images": _ref_images,
            "pred_images": _best_images,
            "pred_coords": _best_coords,
            "ref_domain_distance": _targets,
            "pred_domain_distance": _predictions,
        }
        if truths is not None:
            summary["squared_error"] = _sqerrors
        if true_traj is not None:
            summary["rmsd"] = _rmsds
            summary["ref_coords"] = _true_traj.xyz,
            
        total_summary = cat_data([total_summary, summary])
    
    return total_summary
