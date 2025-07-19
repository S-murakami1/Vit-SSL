import os
import sys

import numpy as np
import openslide
from PIL import Image
from loguru import logger
from matplotlib import pyplot as plt

sys.path.append("dinov2")  # dinov2のパスを追加
from dinov2.data.augmentations import DataAugmentationDINO

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)
# Matplotlibのバックエンドを'Agg'に設定してGUI不要にする
plt.switch_backend('Agg')

data_transform = DataAugmentationDINO(
        global_crops_scale=[0.32, 1.0],
        local_crops_scale=[0.05, 0.32],
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96,
    )

wsi_path = "/mnt/c/DLBCL-Patho2/20/20-0494_1_1.ndpi"
slide = openslide.OpenSlide(wsi_path)
height, width = slide.dimensions
middle_coodinate = (width // 2, height // 2)
coordinate = (30000, 30000)
logger.info(f"Middle coordinate: {middle_coodinate}")
img = slide.read_region(coordinate, 1, (256, 256)).convert("RGB")
img_array = np.array(img)

# 原寸大で保存
Image.fromarray(img_array).save(f"{output_dir}/original_img.png")

output = data_transform(img)
global_crops = output["global_crops"]
for i, crop in enumerate(global_crops):
    # Tensorを(H,W,C)形式に変換し、0-1→0-255にスケール
    crop_np = crop.permute(1,2,0).numpy()
    crop_np = np.clip(crop_np, 0, 1)
    crop_uint8 = (crop_np * 255).astype(np.uint8)
    # PIL Imageに変換して原寸大で保存
    Image.fromarray(crop_uint8).save(f"{output_dir}/global_crop_{i}.png")

local_crops = output["local_crops"]
for i, crop in enumerate(local_crops):
    # Tensorを(H,W,C)形式に変換し、0-1→0-255にスケール
    crop_np = crop.permute(1,2,0).numpy()
    crop_np = np.clip(crop_np, 0, 1)
    crop_uint8 = (crop_np * 255).astype(np.uint8)
    # PIL Imageに変換して原寸大で保存
    Image.fromarray(crop_uint8).save(f"{output_dir}/local_crop_{i}.png")
