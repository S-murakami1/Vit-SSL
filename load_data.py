import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

TARGET_SIZE = 1024  # DINOv2に入力する画像の辺の長さ


def preprocess(x: np.ndarray) -> torch.Tensor:
    """NumPy配列をDINOv2用のテンソルに前処理する"""
    # NumPy配列をPyTorchテンソルに変換
    x = torch.from_numpy(x).float()

    # バッチとチャンネル次元を追加 (B,C,H,W形式にする)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)

    # 1024x1024にリサイズ
    x = F.interpolate(x, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    # 3チャンネルに変換
    x = x.repeat(1, 3, 1, 1)  # バッチ次元を保持したまま3チャンネルに複製

    # 0-255正規化
    min_val = x.min()
    max_val = x.max()
    if max_val > min_val:
        x = (x - min_val) / (max_val - min_val) * 255
    
    # DataLoaderがバッチ次元を追加するため、余分なバッチ次元を削除
    x = x.squeeze(0)  # (1,3,H,W) -> (3,H,W)
    return x


class MriDataset(Dataset):
    """
    前処理を行い、データセットを作成するクラス
    """

    def __init__(self, nii_path):
        self.nii_path = nii_path
        
        # MRIデータを読み込む
        self.mri_img = nib.load(nii_path)
        self.mri_data = self.mri_img.get_fdata()

    def __len__(self) -> int:
        # 有効なスライスの数を返す
        return self.mri_data.shape[2]

    def __getitem__(self, idx):
        """
        DINOv2用のデータ取得
        
        Returns:
            (PIL.Image, target)のタプル
        """
        mri_slice = self.mri_data[:, :, idx]  # H, W (2D slice)
        
        # 0-1正規化
        slice_min, slice_max = mri_slice.min(), mri_slice.max()
        if slice_max > slice_min:
            mri_slice = (mri_slice - slice_min) / (slice_max - slice_min)
        
        mri_slice = preprocess(mri_slice)

        
        # ターゲット（自己教師あり学習では空のタプル）
        target = ()
        
        return mri_slice, target


if __name__ == "__main__":
    # 以下動作確認用コード-------------------------------------------------------------
    nii_dir = r"/mnt/c/brain-segmentation/data/372/153035372/brats2025-gli-pre-challenge-trainingdata/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000"
    for file in os.listdir(nii_dir):
        if file.endswith(".nii.gz") and "seg" not in file:
            nii_path = os.path.join(nii_dir, file)
            break
    
    # DINOv2用の動作確認
    dataset = MriDataset(nii_path)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in loader:
        image, target = batch
        print(f"Image type: {type(image)}")
        print(f"Target: {target}")
        if hasattr(image, 'shape'):
            print(f"Image shape: {image.shape}")
        break
