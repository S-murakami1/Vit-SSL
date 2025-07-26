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
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from sklearn.decomposition import PCA
from load_data import MriDataset
from torch.utils.data import DataLoader
from dinov2.models.vision_transformer import DinoVisionTransformer
import timm
from tqdm import tqdm

nii_dir = r"/mnt/c/brain-segmentation/data/372/153035372/brats2025-gli-pre-challenge-trainingdata/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000"
def dino(images: torch.Tensor):
    # --- 1. モデルの準備 ---
    model = DinoVisionTransformer(
        img_size=1024,
        patch_size=16,
        in_chans=3,  # RGB画像なので3チャンネル
        embed_dim=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4,
    )

    # GPU使用設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    # モデルをGPUに移動
    model = model.to(device)
    # データもGPUに移動
    x = images.to(device)
    


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

def use_timm(images: torch.Tensor, name: str, i: int, model):
    output_dir = "./pictures"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = images.to(device)
    logger.info(f"{x.shape}")
    
    # オリジナル画像のサイズを取得
    original_size = x.shape[-1]  # 1024
    
    with torch.no_grad():
        feats = model.forward_features(x)
        logger.info(f"feats: {feats.shape}")
        out_channel = feats.shape[2]
        # 最初のサンプルのパッチトークン（CLSトークンを除く）を取得
        mri_feature_map = feats[0, 1:, :].reshape(64, 64, out_channel).detach().cpu().numpy()  # [4096, 1024] -> [64, 64, 1024]
        logger.info(f"mri_feature_map: {mri_feature_map.shape}")
    
    # PCA処理
    pca = PCA(n_components=3)
    mri_pca = pca.fit_transform(mri_feature_map.reshape(-1, out_channel))
    mri_pca = mri_pca.reshape(64, 64, 3)
    
    # PCA結果を0-255の範囲に正規化
    mri_pca_norm = (mri_pca - mri_pca.min()) / (mri_pca.max() - mri_pca.min()) * 255
    mri_pca_norm = mri_pca_norm.astype(np.uint8)
    
    # オリジナル画像を取得（最初のサンプル）
    original_img = x[0].detach().cpu().numpy().transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    print("max:", original_img.max())
    print("min:", original_img.min())
    
    # オリジナル画像を0-255の範囲に正規化
    original_img_norm = (original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255
    original_img_norm = original_img_norm.astype(np.uint8)
    
    # PCA画像をオリジナルサイズにリサイズ
    # (H, W, C) -> (C, H, W) -> (1, C, H, W) でPyTorchのinterpolateに渡す
    mri_pca_tensor = torch.from_numpy(mri_pca_norm).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, 64, 64)
    mri_pca_resized_tensor = F.interpolate(mri_pca_tensor, size=(original_size, original_size), mode='bilinear', align_corners=False)
    mri_pca_resized = mri_pca_resized_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # (H, W, C)
    
    # オリジナル画像とPCA画像を並べて表示
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # オリジナル画像を表示
    axes[0].imshow(original_img_norm)
    axes[0].set_title(f'Original MRI Image ({original_size}x{original_size})', fontsize=14)
    axes[0].axis('off')
    
    # PCA画像を表示
    axes[1].imshow(mri_pca_resized)
    axes[1].set_title(f'PCA Features ({name}) - Resized to {original_size}x{original_size}', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{i}_mri_comparison_{name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    
    # 最初の16チャネルを4x4グリッドで表示
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i_fig in range(4):
        for j in range(4):
            channel_idx = i_fig * 4 + j
            # mri_feature_mapの特定チャネルを取得
            feature_map = mri_feature_map[:, :, channel_idx]
            axes[i_fig, j].imshow(feature_map, cmap='viridis')
            axes[i_fig, j].axis('off')
            axes[i_fig, j].set_title(f'Feature {channel_idx}')
    
    plt.tight_layout()
    #plt.savefig(os.path.join(output_dir,f'{i}_timm_feature_maps_{name}.png'))
    plt.close()
    
    return mri_feature_map

if __name__ == "__main__":
    for file in os.listdir(nii_dir):
        if file.endswith(".nii.gz") and "seg" not in file:
            nii_path = os.path.join(nii_dir, file)
            break
    dataset = MriDataset(nii_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # GPU設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # モデルを一度だけ作成
    model_large = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_size=True)
    model_large.eval()
    model_large.to(device)
    
    model_base = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, dynamic_img_size=True)
    model_base.eval()
    model_base.to(device)
    
    logger.info("Models loaded successfully!")
    
    i = 0
    for batch in tqdm(loader):
        if i < 71:
            i = i+1
            continue
        logger.info(f"{i}")
        images, targets = batch
        #dino(images)
        
        # large model
        use_timm(images, "large", i, model_large)
        
        # base model
        #use_timm(images, "base", i, model_base)
        
        i = i+1
    