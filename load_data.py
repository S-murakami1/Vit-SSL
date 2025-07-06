import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from PIL import Image

TARGET_SIZE = 1024  # SAMに入力する画像の辺の長さ


def preprocess(x: np.ndarray) -> torch.Tensor:
    # NumPy配列をPyTorchテンソルに変換
    x = torch.from_numpy(x).float()

    # バッチとチャンネル次元を追加 (B,C,H,W形式にする)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)

    x = apply_image(x)

    min_val = x.min()
    max_val = x.max()

    # 正規化を実行
    x = (x - min_val) / (max_val - min_val) * 255
    x = F.interpolate(x, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    
    # DataLoaderがバッチ次元を追加するため、余分なバッチ次元を削除
    x = x.squeeze(0)  # (1,1,H,W) -> (1,H,W)
    return x


def apply_image(image: torch.Tensor) -> torch.Tensor:
    target_size = get_preprocess_shape(image.shape[2], image.shape[3], TARGET_SIZE)
    return F.interpolate(
        image, target_size, mode="bilinear", align_corners=False, antialias=True
    )


def get_preprocess_shape(
    oldh: int, oldw: int, long_side_length: int
) -> tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)



class MriDataset(Dataset):
    """
    前処理を行い、データセットを作成するクラス
    """

    def __init__(self, nii_path, transform=None, target_transform=None):
        self.nii_path = nii_path
        self.transform = transform
        self.target_transform = target_transform
        
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
        
        # 0-255の範囲に変換
        mri_slice = (mri_slice * 255).astype(np.uint8)
        
        # PIL.Imageに変換（グレースケールのまま）
        image = Image.fromarray(mri_slice, mode='L')
        
        # 1024x1024にリサイズ
        image = image.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR)
        
        # Transformを適用
        if self.transform:
            image = self.transform(image)
        
        # ターゲット（自己教師あり学習では空のタプル）
        target = ()
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target


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
