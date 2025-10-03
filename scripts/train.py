import os
import sys
import argparse
import logging
import os
import numpy as np
import torch
import mdtraj as md
import pickle
from tqdm import tqdm
import glob
import json
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from argparse import Namespace
from afmfold.cnn import CNSteerableCNN
from afmfold.train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="AFM-CNN Training Script")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data-dir", type=str, required=True, help="Path to directory which contains image_*.npy and label_*.npy")
    parser.add_argument("--ckpt-dir", type=str, default="", help="Model path")
    parser.add_argument("--output-root", type=str, required=True, help="Directory to save training output")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs (default: 500)")
    return parser.parse_args()

def get_logger(name):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        # ログレベル設定（環境変数 LOG_LEVEL を優先、なければ INFO）
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))

        # ハンドラー作成
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logger.level)

        # フォーマット設定
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        # ハンドラー追加
        logger.addHandler(handler)
        logger.propagate = False  # ルートロガーへの伝播を防ぐ

    return logger

def pairwise_distance(x1, x2):
    """
    形状 (B, D) のテンソル x1, x2 に対して、形状 (B,) の距離テンソルを計算し、その平均を返す。
    """
    return torch.mean(torch.norm(x1 - x2, dim=1))

def json_to_namespace(
    path,
    *,
    expanduser=True,
    expandvars=True,
    sanitize_keys=True,
):
    """
    Load a JSON file and return an argparse.Namespace.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    def _expand(v):
        if isinstance(v, str):
            s = v
            if expanduser:
                s = os.path.expanduser(s)
            if expandvars:
                s = os.path.expandvars(s)
            return s
        if isinstance(v, list):
            return [_expand(x) for x in v]
        if isinstance(v, dict):
            return {k: _expand(x) for k, x in v.items()}
        return v

    def _sanitize_key(k):
        return k.replace("-", "_") if sanitize_keys else k

    expanded = _expand(data)
    sanitized = {_sanitize_key(k): v for k, v in expanded.items()}
    
    return Namespace(**sanitized)

def search_best_model(ckpt_path, k=1, max_epoch=None, model_type=None):
    assert model_type is None or model_type in ["train", "val"]
    assert max_epoch is None or isinstance(max_epoch, int)
    # モデル情報をまとめる
    path_loss_dict = {
        p: {
            "loss": float(os.path.basename(p).split("loss")[-1].split("_")[0]),
            "epoch": int(os.path.basename(p).split("epoch")[-1].split("_")[0]),
            "model_type": os.path.basename(p).split("_")[-1].split(".pt")[0],
            "train_loss_path": os.path.join(ckpt_path, "loss", f"train_losses_epoch{os.path.basename(p).split('epoch')[-1].split('_')[0]}.npy"),
            "val_loss_path": os.path.join(ckpt_path, "loss", f"val_losses_epoch{os.path.basename(p).split('epoch')[-1].split('_')[0]}.npy"),
            }
        for p in glob.glob(os.path.join(ckpt_path, "model", "model_epoch*.pt"))
        }
    
    # 絞り込み
    if max_epoch is not None:
        path_loss_dict = {
            path: model_info for path, model_info in path_loss_dict.items() if model_info["epoch"] <= max_epoch
        }
        
    if model_type is not None:
        path_loss_dict = {
            path: model_info for path, model_info in path_loss_dict.items() if model_info["model_type"] == model_type
        }
    
    path_loss_dict = {
        path: model_info for path, model_info in path_loss_dict.items() if os.path.exists(model_info["train_loss_path"]) and os.path.exists(model_info["val_loss_path"])
    }
    
    # ロスの小さい順番に並べ替え
    sorted_paths = sorted([(p, path_loss_dict[p]["train_loss_path"], path_loss_dict[p]["val_loss_path"]) for p in path_loss_dict.keys()], key=lambda x: path_loss_dict[x[0]]["loss"])
    
    if len(sorted_paths) > k:
        top_k_model_paths = [p[0] for p in sorted_paths[:k]]
        top_k_train_losses_paths = [p[1] for p in sorted_paths[:k]]
        top_k_val_losses_paths = [p[2] for p in sorted_paths[:k]]
    else:
        top_k_model_paths = [p[0] for p in sorted_paths]
        top_k_train_losses_paths = [p[1] for p in sorted_paths]
        top_k_val_losses_paths = [p[2] for p in sorted_paths]
    return top_k_model_paths, top_k_train_losses_paths, top_k_val_losses_paths

def save_args_to_file(args, filename):
    """
    argsの内容を辞書に変換し、指定されたファイルにJSON形式で保存する。

    Parameters:
        args (Namespace or dict): argparse.Namespace または dict 型の設定オブジェクト
        filename (str): 保存するファイル名（例："config.json"）
    """
    if isinstance(args, argparse.Namespace):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        raise TypeError("args は Namespace または dict である必要があります。")

    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
if __name__ == "__main__":
    args = parse_args()
    logger = get_logger(__name__)

    # もし ckpt が指定されている場合、それを採用
    if os.path.exists(args.ckpt_dir):
        eff_args = json_to_namespace(os.path.join(args.ckpt_dir, "args.json"))
    else:
        eff_args = args
    
    # データの読み込み
    image_files = glob.glob(os.path.join(eff_args.data_dir, "image_*.npy"))
    label_files = glob.glob(os.path.join(eff_args.data_dir, "label_*.npy"))
    
    image_indices = sorted([int(os.path.basename(file).split("_")[-1].split(".npy")[0]) for file in image_files])
    label_indices = sorted([int(os.path.basename(file).split("_")[-1].split(".npy")[0]) for file in label_files])
    assert image_indices == label_indices
    
    image_list = []
    label_list = []

    for idx in image_indices:
        image_path = os.path.join(eff_args.data_dir, f"image_{idx}.npy")
        label_path = os.path.join(eff_args.data_dir, f"label_{idx}.npy")
        
        assert os.path.exists(image_path) and os.path.exists(label_path)
        image = torch.from_numpy(np.load(image_path))
        label = torch.from_numpy(np.load(label_path))
        image_list.append(image)
        label_list.append(label)
    
    all_image = torch.cat(image_list, dim=0).to(torch.float)
    all_label = torch.cat(label_list, dim=0).to(torch.float)
    assert len(all_image) == len(all_label)
    dataset_size = len(all_image)
    
    train_batchsize = int(0.8 * dataset_size)
    val_batchsize = dataset_size - train_batchsize
    logger.info(f"Train dataset size: {train_batchsize}, validation dataset size: {val_batchsize}")

    train_dataset = TensorDataset(all_image[:train_batchsize], all_label[:train_batchsize])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(all_image[train_batchsize:], all_label[train_batchsize:])
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # モデルの読み込み
    if os.path.exists(args.ckpt_dir):
        model_paths, train_losses_paths, val_losses_paths = search_best_model(args.ckpt_dir, k=1)
        model, _, _ = CNSteerableCNN.load_from_checkpoint(model_paths[0])
        prev_train_losses = np.load(train_losses_paths[0])
        prev_val_losses = np.load(val_losses_paths[0])
        logger.info(f"Loaded CNN from {model_paths[0]}.")
        
        output_subdir = os.path.basename(args.ckpt_dir)
    else:
        model = CNSteerableCNN(
            N=8, 
            image_shape=all_image.shape[-2:], 
            output_dim=all_label.shape[-1],
            )
        prev_train_losses = None
        prev_val_losses = None
        output_subdir = None
    
    os.makedirs(eff_args.output_root, exist_ok=True)
    
    # トレイナーの設定
    trainer = Trainer(
        model=model,
        criterion=pairwise_distance,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        output_root=eff_args.output_root,
        output_subdir=output_subdir,
        prev_train_losses=prev_train_losses,
        prev_val_losses=prev_val_losses,
        device=args.device,
        )
    
    os.makedirs(trainer.output_dir, exist_ok=True)
    
    # 設定の保存
    if not os.path.exists(os.path.join(trainer.output_dir, "args.json")):
        save_args_to_file(args, os.path.join(trainer.output_dir, "args.json"))
    
    # 訓練
    trainer.train(epochs=args.epochs)
    