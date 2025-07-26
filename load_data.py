import os

import nibabel as nib
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from PIL import Image

TARGET_SIZE = 896  # DINOv2に入力する画像の辺の長さ


def numpy_to_pil(x: np.ndarray) -> Image.Image:
    """NumPy配列を直接PIL画像に変換する"""
    # 0-1正規化
    min_val = x.min()
    max_val = x.max()
    if max_val > min_val:
        x = (x - min_val) / (max_val - min_val)
    
    # 0-255の範囲に変換してuint8に
    x = (x * 255).astype(np.uint8)
    
    # PIL画像を作成（グレースケール）
    pil_image = Image.fromarray(x, mode='L')
    
    # TARGET_SIZEにリサイズ
    pil_image = pil_image.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
    
    # RGBに変換（3チャンネル化）
    pil_image = pil_image.convert('RGB')
    
    return pil_image


class MriDataset(Dataset):
    """
    MRIデータをPIL画像として提供するデータセット

    Args:
        nii_path: NIfTIファイルのパス 
        transform: PIL画像に適用する変換（DataAugmentationDINO等）
    Returns:
        PIL画像または変換後の辞書, target（空のタプル）
    """

    def __init__(self, nii_path, transform=None):
        self.nii_path = nii_path
        self.transform = transform
        
        # MRIデータを読み込む
        self.mri_img = nib.load(nii_path)
        self.mri_data = self.mri_img.get_fdata()

    def __len__(self) -> int:
        return self.mri_data.shape[2]

    def __getitem__(self, idx):
        """
        MRIスライスをPIL画像として取得
        
        Returns:
            (PIL画像 or 変換後の辞書, target)のタプル
        """
        mri_slice = self.mri_data[:, :, idx]  # H, W (2D slice)
        
        # numpy配列を直接PIL画像に変換
        pil_image = numpy_to_pil(mri_slice)
        
        # データ拡張適用
        if self.transform:
            pil_image = self.transform(pil_image)
        
        # ターゲット（自己教師あり学習では空のタプル）
        target = ()
        
        return pil_image, target


if __name__ == "__main__":
    # 動作確認用コード
    nii_dir = r"/mnt/c/brain-segmentation/data/372/153035372/brats2025-gli-pre-challenge-trainingdata/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000"
    for file in os.listdir(nii_dir):
        if file.endswith(".nii.gz") and "seg" not in file:
            nii_path = os.path.join(nii_dir, file)
            break
    
    # PIL画像での動作確認
    print("=== PIL形式での動作確認 ===")
    dataset = MriDataset(nii_path)
    
    # 1つのサンプルを確認
    sample_image, sample_target = dataset[0]
    print(f"Image type: {type(sample_image)}")
    print(f"Image size: {sample_image.size}")
    print(f"Image mode: {sample_image.mode}")
    print(f"Target: {sample_target}")
    
    print(f"Dataset length: {len(dataset)}")
