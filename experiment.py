import os
import matplotlib
matplotlib.use('Agg')  # Add this line before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("dinov2")
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
from loguru import logger
from sklearn.decomposition import PCA
from load_data import MriDataset
from torch.utils.data import DataLoader
from dinov2.models.vision_transformer import DinoVisionTransformer

nii_dir = r"/mnt/c/brain-segmentation/data/372/153035372/brats2025-gli-pre-challenge-trainingdata/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000"
if __name__ == "__main__":
    for file in os.listdir(nii_dir):
        if file.endswith(".nii.gz") and "seg" not in file:
            nii_path = os.path.join(nii_dir, file)
            break
    dataset = MriDataset(nii_path)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in loader:
        logger.info(batch.shape)
        break

    # --- 1. モデルの準備 ---
    model = DinoVisionTransformer(
        img_size=1024,
        patch_size=16,
        in_chans=1,
        embed_dim=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4,
    )

    x = batch  # [B, 1, 1024, 1024]

    # --- 3. 中間特徴取得（パッチ埋め込み＋ブロック出力） ---
    with torch.no_grad():
        feats = model.forward_features(x)

    # 辞書から必要なテンソルを取得
    patch_tokens = feats['x_norm_patchtokens']  # パッチトークンを取得
    cls_token = feats['x_norm_clstoken']  # CLSトークンを取得

    logger.info(f"patch_tokens: {patch_tokens.shape}")
    logger.info(f"cls_token: {cls_token.shape}")

    # パッチ数を計算（1024x1024の画像、パッチサイズ16なら64x64=4096パッチ）
    num_patches = int(patch_tokens.shape[1] ** 0.5)  # 4096の平方根 = 64
    logger.info(f"num_patches per side: {num_patches}")

    # パッチトークンを2Dに変形 [1, 4096, 1024] -> [1, 64, 64, 256] -> [1, 256, 64, 64]
    feat_map = patch_tokens[0].reshape(1, num_patches, num_patches, -1).permute(0, 3, 1, 2)
    logger.info(f"feat_map: {feat_map.shape}")

    # 最初の16チャネルを4x4グリッドで表示
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(4):
        for j in range(4):
            channel_idx = i * 4 + j
            feature_map = feat_map.detach().squeeze(0)[channel_idx].cpu().numpy()
            axes[i, j].imshow(feature_map)
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Feature {channel_idx}')

    plt.tight_layout()
    plt.savefig('feature_maps.png')
    plt.close()

    logger.info(f"feat_map: {feat_map.shape}")
