#!/usr/bin/env python
"""
Minimal debug script that focuses only on the DINOv2 forward pass (equivalent to Step-8).
It loads a single batch from the MRI dataset, moves everything to the same GPU device,
and runs the forward / backward routine once under autocast(fp16).

Usage:
    python test_step8.py --nii-path /path/to/file.nii.gz [--config path/to/yaml] [--batch-size 2]
"""
import argparse
import os
import torch
from functools import partial

# Disable xformers to avoid nested-tensor constraints
os.environ.setdefault("XFORMERS_DISABLED", "1")

from dinov2.utils.config import setup
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.data import DataAugmentationDINO, MaskingGenerator, collate_data_and_cast
from dinov2.data.loaders_mri import make_dataset, make_data_loader, SamplerType


def build_single_batch_loader(cfg, nii_path, batch_size):
    # ----- transforms & mask generator -----
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=torch.half,  # to be promoted by autocast
    )

    dataset_str = f"MriDataset:path={nii_path}"
    dataset = make_dataset(dataset_str=dataset_str, transform=data_transform)
    loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        sampler_type=SamplerType.EPOCH,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nii-path", required=True, help="Path to MRI .nii.gz file")
    parser.add_argument("--config", default="dinov2/dinov2/configs/ssl_mri_vits_config.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    # minimal cfg
    class _Args:
        config_file = args.config
        output_dir = "./tmp_out"
        opts = [
            f"train.dataset_path=MriDataset:path={args.nii_path}",
            "student.block_chunks=1",
            "teacher.block_chunks=1",
        ]

    cfg = setup(_Args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = SSLMetaArch(cfg).to(device)
    model.train()

    # Data
    loader = build_single_batch_loader(cfg, args.nii_path, args.batch_size)
    batch = next(iter(loader))
    gpu_batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Forward pass with autocast
    with torch.no_grad():
        try:
            with torch.cuda.amp.autocast():
                loss_dict = model.forward_backward(gpu_batch, teacher_temp=0.04)
            print("Forward pass successful. Loss keys:", list(loss_dict.keys()))
        except Exception as e:
            print("Forward pass failed:", e)
            raise


if __name__ == "__main__":
    main() 