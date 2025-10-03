from pathlib import Path
import numpy as np

# Refer to storage/domain
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOMAIN_DIR = BASE_DIR / "storage" / "domain"

def get_domain_pairs(protein_name="4ake"):
    if protein_name.lower() in ["3a5i", "flhac"]:
        dom1 = DOMAIN_DIR / "flhac/acd1.npy"
        dom2 = DOMAIN_DIR / "flhac/acd2.npy"
        dom3 = DOMAIN_DIR / "flhac/acd3.npy"
        dom4 = DOMAIN_DIR / "flhac/acd4.npy"

        domain_pairs = [
            (np.load(dom4), np.load(dom1)),
            (np.load(dom2), np.load(dom3)),
        ]
    
    elif protein_name.lower() in ["1ake", "4ake", "ak"]:
        dom1 = DOMAIN_DIR / "ak/atpbd.npy"
        dom2 = DOMAIN_DIR / "ak/core.npy"
        dom3 = DOMAIN_DIR / "ak/ampbd.npy"

        domain_pairs = [
            (np.load(dom1), np.load(dom2)),
            (np.load(dom2), np.load(dom3)),
            (np.load(dom3), np.load(dom1)),
        ]
    
    else:
        raise NotImplementedError(f"Invalid Protein Name: {protein_name}")
    
    return domain_pairs

def compute_domain_distance(traj, domain1, domain2):
    """
    トラジェクトリ[B, N, 3]をドメイン間距離[B, 1]に射影する関数

    Args:
        traj (mdtraj.Trajectory): トラジェクトリ
        domain1 (list of int): ドメイン1を構成する残基インデックス
        domain2 (list of int): ドメイン2を構成する残基インデックス

    Returns:
        np.ndarray: ドメイン間距離 [B, 1]
    """
    # traj から Cα 原子だけを抽出
    ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == 'CA']
    traj = traj.atom_slice(ca_indices)
    
    # 残基数を取得
    num_atoms = traj.xyz.shape[1]  # traj.xyz.shape = [B, N, 3]

    # 入力検証：インデックスが範囲内にあることを確認
    if max(domain1) >= num_atoms or max(domain2) >= num_atoms:
        raise ValueError("domain1またはdomain2のインデックスがtraj内の原子数を超えています。")

    # 各ドメインの座標を抽出
    coords1 = traj.xyz[:, domain1, :]  # shape: [B, len(domain1), 3]
    coords2 = traj.xyz[:, domain2, :]  # shape: [B, len(domain2), 3]

    # 各ドメインの平均座標（重心）を計算
    centroid1 = np.mean(coords1, axis=1)  # shape: [B, 3]
    centroid2 = np.mean(coords2, axis=1)  # shape: [B, 3]

    # 各フレームにおける重心間のユークリッド距離を計算
    distances = np.linalg.norm(centroid1 - centroid2, axis=1, keepdims=True)  # shape: [B, 1]

    return distances
