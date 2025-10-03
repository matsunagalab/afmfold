import numpy as np
import torch
import os
import time
import glob
from tqdm import tqdm
import shutil
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.optim import Adam
from scipy.spatial.transform import Rotation as R
import mdtraj as md
from tqdm import tqdm
from afmfold.utils import move_all_tensors_in_device

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 output_root: str,
                 output_subdir: str = None, 
                 optimizer: torch.optim.Optimizer = None,
                 prev_train_losses: np.ndarray = None,
                 prev_val_losses: np.ndarray = None,
                 run_name: str = "",
                 top_k: int = 3,
                 save_interval_sec: int = 3600,
                 device: str = "cuda"):
        """
        初期化：全構成要素と保存先などを受け取る。
        """
        device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_root = output_root
        self.output_subdir = output_subdir
        if optimizer is None:
            self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer
        self.prev_train_losses = prev_train_losses
        self.prev_val_losses = prev_val_losses
        self.run_name = run_name
        self.top_k = top_k
        self.save_interval_sec = save_interval_sec
        self.device = device
        
        # ログ保存ディレクトリ構造の作成
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        if output_subdir is None:
            self.output_dir = os.path.join(output_root, run_name + self.timestamp)
        else:
            self.output_dir = os.path.join(output_root, output_subdir)
        self.model_dir = os.path.join(self.output_dir, "model")
        self.loss_dir = os.path.join(self.output_dir, "loss")
        self.last_save_time = time.time()
        
        # 途中から実行する場合、過去の結果を読み取る
        if os.path.exists(self.model_dir) and len(os.listdir(self.model_dir)) > 0:
            prev_model_paths = glob.glob(os.path.join(self.model_dir, "model_epoch*.pt"))
            prev_model_epochs = [int(os.path.basename(p).split("epoch")[-1].split("_")[0]) for p in prev_model_paths]
            self.initial_epoch = max(prev_model_epochs) + 1
        else:
            self.initial_epoch = 0
            
        # 保存管理
        self.top_k_losses = []  # List of (loss, path)

        # 損失履歴
        if prev_train_losses is None:
            self.train_losses = []
        else:
            self.train_losses = prev_train_losses.tolist()
        if prev_val_losses is None:
            self.val_losses = []
        else:
            self.val_losses = prev_val_losses.tolist()

    def train(self, epochs: int):
        """
        学習ループ全体の開始
        """
        # 学習ループ
        for epoch in range(self.initial_epoch, self.initial_epoch + epochs):
            self.model.train()
            self.train_one_epoch(epoch)  # 1エポック分の訓練
            
            self.model.eval()
            self.validate_one_epoch(epoch)  # 1エポック分の検証
        
        # 最終結果を保存
        self.save_model(epoch, self.val_losses[-1])
        self.save_loss_log(epoch)

    def train_one_epoch(self, epoch):
        """
        1エポック分の訓練
        """
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} - Training"):  # tqdm表示
            loss = self.compute_loss(batch)  # 訓練
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            # 一定時間ごとにモデルと損失を保存
            if time.time() - self.last_save_time >= self.save_interval_sec:
                self.save_model(epoch, loss.detach().cpu().item(), save_topk=False, suffix="_train")
                self.save_loss_log(epoch)
            
            # 損失を記録
            self.train_losses.append(loss.detach().cpu().item())
        
        # 最後にモデルと損失を保存
        self.save_model(epoch, self.train_losses[-1], save_topk=False, suffix="_train")
        self.save_loss_log(epoch)
    
    def validate_one_epoch(self, epoch):
        with torch.no_grad():
            for batch in self.val_loader:  # 検証ステップでは表示は不要
                loss = self.compute_loss(batch)  # 検証
                
                # 一定時間ごとにモデルと損失を保存
                if time.time() - self.last_save_time >= self.save_interval_sec:
                    self.save_model(epoch, loss.detach().cpu().item(), save_topk=False, suffix="_val")
                    self.save_loss_log(epoch)

                # エポックごとの検証損失を計算
                self.val_losses.append(loss.detach().cpu().item())
        
        # 一定時間ごとにモデルと損失を保存
        self.save_model(epoch, self.val_losses[-1], save_topk=True, suffix="_val")
        self.save_loss_log(epoch)
            
    def compute_loss(self, batch):
        """
        バッチから損失を計算する。
        """
        data, label = batch
        data = data.unsqueeze(1)
        
        # デバイスに移動
        data, label = move_all_tensors_in_device(data, label, device=self.device)
        
        # フォワードパス
        output = self.model(data)

        # 検証損失を計算
        loss = self.criterion(output, label)

        return loss
        
    def save_model(self, epoch, loss, save_topk=False, suffix=""):
        """
        モデルを保存し、Top-Kのうち最も良いものを保持。
        通常ファイルとtop-kファイルの両方を保存する。
        """
        # 保存先ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.last_save_time = time.time()
        
        # 通常の保存ファイル名
        filename = f"model_epoch{epoch}_loss{loss:.4f}{suffix}.pt"
        filepath = os.path.join(self.model_dir, filename)

        # モデルと状態を保存
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': {
                'N': self.model.N,
                'image_shape': self.model.image_shape,
                'channels': self.model.channels,
                'kernel_size': self.model.kernel_size,
                'num_hidden_layers': self.model.num_hidden_layers,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim,
            },
        }, filepath)

        if save_topk:
            # Top-Kに登録
            self.top_k_losses.append((loss, filepath, epoch))  # epochも記録
            self.top_k_losses.sort(key=lambda x: x[0])  # 損失でソート

            # Top-Kモデルとして保存
            for k, (top_loss, top_path, top_epoch) in enumerate(self.top_k_losses[:self.top_k], 1):
                top_filename = f"model_top{k}.pt"
                top_filepath = os.path.join(self.model_dir, top_filename)
                if not os.path.exists(top_filepath):
                    shutil.copyfile(top_path, top_filepath)

            # Top-Kを超えた古いモデルを削除（オリジナルのみ削除、top_kファイルは残す）
            while len(self.top_k_losses) > self.top_k:
                _, path_to_remove, _ = self.top_k_losses.pop(-1)
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)

    def save_loss_log(self, epoch, suffix=""):
        """
        損失履歴をnumpy形式でエポック付きファイル名で保存
        """
        # 保存先ディレクトリの作成
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        self.last_save_time = time.time()
        
        train_path = os.path.join(self.loss_dir, f"train_losses_epoch{epoch}.npy")
        val_path = os.path.join(self.loss_dir, f"val_losses_epoch{epoch}.npy")

        np.save(train_path, np.array(self.train_losses))
        np.save(val_path, np.array(self.val_losses))
        