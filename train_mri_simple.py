#!/usr/bin/env python3
"""
DINOv2の学習でMriDatasetを使用するスクリプト
"""

import os
import sys
import argparse
from functools import partial

# プロジェクトのパスを追加
sys.path.append("dinov2")

import torch
from dinov2.train.train import (
    build_optimizer, build_schedulers, apply_optim_scheduler, do_test
)
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.data.augmentations import DataAugmentationDINO, DataAugmentationDINOMRI
from dinov2.data.collate import collate_data_and_cast
from dinov2.data.masking import MaskingGenerator
from dinov2.data.loaders import make_data_loader, SamplerType
from dinov2.utils.config import setup
from dinov2.logging import MetricLogger
from dinov2.fsdp import FSDPCheckpointer
from fvcore.common.checkpoint import PeriodicCheckpointer
import dinov2.distributed as distributed

# カスタムデータセットをインポート
from load_data import MriDataset

import logging
import math

logger = logging.getLogger("dinov2")


def do_train_with_mri(cfg, model, nii_path, resume=False):
    """
    MriDatasetを使用した学習関数
    """
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler

    # optimizer setup
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # data preprocessing setup
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    # MRI用のデータ拡張を使用
    data_transform = DataAugmentationDINOMRI(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # === ここでMriDatasetを使用 ===
    print(f"MRIデータセットを作成中: {nii_path}")
    dataset = MriDataset(
        nii_path=nii_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    print(f"データセットサイズ: {len(dataset)}")

    # data loader setup (シングルGPU用)
    sampler_type = SamplerType.INFINITE  # 非分散サンプラーを使用
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,
        sampler_type=sampler_type,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update
        model.update_teacher(mom)

        # logging (シングルGPU用に調整)
        world_size = distributed.get_global_size()
        if world_size > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / world_size for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():
    parser = argparse.ArgumentParser(description="MRI画像でDINOv2を学習")
    parser.add_argument("--nii-path", required=True, help="MRI NIIファイルのパス")
    parser.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--config-file", default="dinov2/dinov2/configs/ssl_default_config.yaml", help="設定ファイル")
    parser.add_argument("--resume", action="store_true", help="学習を再開")
    
    args = parser.parse_args()
    
    # 分散学習環境の設定（シングルGPU用）
    # 既存の分散学習環境変数をクリア
    distributed_env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    for env_var in distributed_env_vars:
        if env_var in os.environ:
            del os.environ[env_var]
    
    # シングルGPU用の環境変数を明示的に設定
    # これによりローカルモードが確実に使用される
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    
    print("分散学習環境をシングルGPUモードに設定しました")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDAデバイス数: {torch.cuda.device_count()}")
        print(f"現在のCUDAデバイス: {torch.cuda.current_device()}")
    
    # 設定ファイルの読み込み
    sys.argv = [
        "train_mri_simple.py",
        "--config-file", args.config_file,
        "--output-dir", args.output_dir,
    ]
    
    # DINOv2の引数パーサーを使用
    from dinov2.train.train import get_args_parser
    dinov2_args = get_args_parser().parse_args()
    
    # 設定を読み込み
    cfg = setup(dinov2_args)
    
    # 出力ディレクトリを設定
    cfg.train.output_dir = args.output_dir
    
    # MRI用の設定調整
    cfg.crops.global_crops_size = 1024
    cfg.crops.local_crops_size = 256
    cfg.train.batch_size_per_gpu = 1  # シングルGPU + 大画像のため最小に
    cfg.train.num_workers = 2  # シングルGPUでは少なめに
    cfg.student.in_chans = 1  # 1チャネル（グレースケール）に設定
    
    print("=== MRI DINOv2学習開始 ===")
    print(f"MRIファイル: {args.nii_path}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"バッチサイズ: {cfg.train.batch_size_per_gpu}")
    print(f"ワーカー数: {cfg.train.num_workers}")
    print(f"画像サイズ: {cfg.crops.global_crops_size}")
    print(f"入力チャネル数: {cfg.student.in_chans} (グレースケール)")
    print(f"データ拡張: MRI専用")
    print(f"実行モード: シングルGPU")
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデル作成
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    
    # 分散学習の準備（シングルGPUでもエラーを回避）
    try:
        model.prepare_for_distributed_training()
    except Exception as e:
        print(f"分散学習の準備をスキップ: {e}")
        print("シングルGPUモードで継続します")
    
    # 学習実行
    do_train_with_mri(cfg, model, args.nii_path, resume=args.resume)
    
    print("=== 学習完了 ===")


if __name__ == "__main__":
    main() 