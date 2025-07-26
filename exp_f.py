#!/usr/bin/env python3
"""
load_data.pyの動作をステップバイステップで検証するスクリプト
各段階で問題が発生した場合に原因を特定できるように設計
"""

import os
import sys
import traceback
from typing import Any, Tuple

# プロジェクトのパスを追加
sys.path.append("dinov2")

import torch
import numpy as np
from PIL import Image
from functools import partial

# DINOv2関連のインポート
from dinov2.data.augmentations import DataAugmentationDINO, DataAugmentationDINOMRI
from dinov2.data.collate import collate_data_and_cast
from dinov2.data.masking import MaskingGenerator
from dinov2.data.loaders import make_data_loader, SamplerType
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.utils.config import setup
import dinov2.distributed as distributed

# カスタムデータセットをインポート
from load_data import MriDataset

def step1_basic_dataset_test(nii_path: str) -> bool:
    """
    ステップ1: MriDatasetの基本動作確認
    """
    print("\n" + "="*60)
    print("ステップ1: MriDatasetの基本動作確認")
    print("="*60)
    
    try:
        # データセット作成
        print(f"データセット作成中: {nii_path}")
        dataset = MriDataset(nii_path)
        
        # 基本情報確認
        print(f"✓ データセット作成成功")
        print(f"  データセット長: {len(dataset)}")
        print(f"  MRIデータ形状: {dataset.mri_data.shape}")
        
        # 最初のサンプル取得
        print(f"最初のサンプル取得中...")
        sample_image, sample_target = dataset[0]
        
        # 結果確認
        print(f"✓ サンプル取得成功")
        print(f"  画像型: {type(sample_image)}")
        print(f"  画像サイズ: {sample_image.size}")
        print(f"  画像モード: {sample_image.mode}")
        print(f"  ターゲット: {sample_target}")
        print(f"  ターゲット型: {type(sample_target)}")
        
        # 複数のサンプル確認
        print(f"複数サンプルの確認...")
        for i in [0, len(dataset)//2, len(dataset)-1]:
            try:
                img, tgt = dataset[i]
                print(f"  インデックス {i}: 画像サイズ={img.size}, ターゲット={tgt}")
            except Exception as e:
                print(f"  ✗ インデックス {i} でエラー: {e}")
                return False
        
        print(f"✓ ステップ1完了: 基本動作確認成功")
        return True
        
    except Exception as e:
        print(f"✗ ステップ1失敗: {e}")
        traceback.print_exc()
        return False


def step2_data_augmentation_test(nii_path: str) -> bool:
    """
    ステップ2: データ拡張の動作確認
    """
    print("\n" + "="*60)
    print("ステップ2: データ拡張の動作確認")
    print("="*60)
    
    try:
        # データセット作成（データ拡張なし）
        dataset = MriDataset(nii_path)
        
        # MRI用データ拡張の作成
        print("MRI用データ拡張を作成中...")
        data_transform = DataAugmentationDINOMRI(
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8,
            global_crops_size=896,
            local_crops_size=256,
        )
        print(f"✓ データ拡張作成成功")
        
        # 拡張前の画像
        original_image, _ = dataset[0]
        print(f"拡張前画像: サイズ={original_image.size}, モード={original_image.mode}")
        
        # データ拡張適用
        print("データ拡張適用中...")
        augmented_data = data_transform(original_image)
        
        print(f"✓ データ拡張適用成功")
        print(f"  拡張後のデータ型: {type(augmented_data)}")
        
        if isinstance(augmented_data, dict):
            for key, value in augmented_data.items():
                if isinstance(value, list):
                    print(f"  {key}: リスト (長さ={len(value)})")
                    for i, item in enumerate(value):
                        if hasattr(item, 'shape'):
                            print(f"    [{i}]: shape={item.shape}, dtype={item.dtype}")
                        else:
                            print(f"    [{i}]: {type(item)}")
                elif hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
        
        print(f"✓ ステップ2完了: データ拡張確認成功")
        return True
        
    except Exception as e:
        print(f"✗ ステップ2失敗: {e}")
        traceback.print_exc()
        return False


def step3_dataloader_test(nii_path: str) -> bool:
    """
    ステップ3: DataLoaderの動作確認
    """
    print("\n" + "="*60)
    print("ステップ3: DataLoaderの動作確認")
    print("="*60)
    
    try:
        # データセット作成
        dataset = MriDataset(nii_path)
        
        # データ拡張設定
        data_transform = DataAugmentationDINOMRI(
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8,
            global_crops_size=896,
            local_crops_size=256,
        )
        
        # マスキング設定
        img_size = 896
        patch_size = 16
        n_tokens = (img_size // patch_size) ** 2
        
        mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
        
        # コレート関数設定
        collate_fn = partial(
            collate_data_and_cast,
            mask_ratio_tuple=(0.1, 0.5),
            mask_probability=1.0,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            dtype=torch.float32,
        )
        
        # 修正されたMriDatasetクラス（データ拡張対応）
        class MriDatasetWithTransform(MriDataset):
            def __init__(self, nii_path, transform=None):
                super().__init__(nii_path)
                self.transform = transform
            
            def __getitem__(self, idx):
                mri_slice = self.mri_data[:, :, idx]
                from load_data import numpy_to_pil
                pil_image = numpy_to_pil(mri_slice)
                
                # データ拡張適用
                if self.transform:
                    pil_image = self.transform(pil_image)
                
                target = ()
                return pil_image, target
        
        # データセット再作成（変換付き）
        dataset_with_transform = MriDatasetWithTransform(nii_path, transform=data_transform)
        
        print(f"DataLoader作成中...")
        data_loader = make_data_loader(
            dataset=dataset_with_transform,
            batch_size=1,
            num_workers=0,  # シングルプロセスでテスト
            shuffle=False,
            seed=0,
            sampler_type=SamplerType.EPOCH,
            sampler_size=5,  # 少数のサンプルでテスト
            drop_last=False,
            collate_fn=collate_fn,
        )
        
        print(f"✓ DataLoader作成成功")
        print(f"  DataLoaderの長さ: {len(data_loader)}")
        
        # 最初のバッチ取得
        print(f"最初のバッチ取得中...")
        data_iter = iter(data_loader)
        batch = next(data_iter)
        
        print(f"✓ バッチ取得成功")
        print(f"  バッチのキー: {list(batch.keys())}")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
        
        print(f"✓ ステップ3完了: DataLoader確認成功")
        return True
        
    except Exception as e:
        print(f"✗ ステップ3失敗: {e}")
        traceback.print_exc()
        return False


def step4_model_input_test(nii_path: str) -> bool:
    """
    ステップ4: モデルへの入力確認
    """
    print("\n" + "="*60)
    print("ステップ4: モデルへの入力確認")
    print("="*60)
    
    try:
        # 簡単な設定作成
        print("設定作成中...")
        
        # 設定ファイルの読み込み（簡易版）
        class SimpleConfig:
            def __init__(self):
                self.crops = type('obj', (object,), {
                    'global_crops_size': 896,
                    'local_crops_size': 256,
                    'global_crops_scale': (0.32, 1.0),
                    'local_crops_scale': (0.05, 0.32),
                    'local_crops_number': 8
                })
                self.student = type('obj', (object,), {
                    'patch_size': 16,
                    'in_chans': 3,  # RGB入力
                    'arch': 'vit_small',
                    'embed_dim': 384,
                    'depth': 12,
                    'num_heads': 6,
                    'mlp_ratio': 4.0,
                    'drop_path_rate': 0.1,
                    'drop_path_uniform': True,
                    'layerscale': 1.0e-4,
                    'block_chunks': 0
                })
                self.teacher = self.student
                self.dino = type('obj', (object,), {
                    'head_n_prototypes': 65536,
                    'head_bottleneck_dim': 256,
                    'head_nlayers': 3,
                    'head_hidden_dim': 2048,
                    'koleo_loss_weight': 0.1
                })
                self.ibot = type('obj', (object,), {
                    'mask_ratio_min_max': (0.1, 0.5),
                    'mask_sample_probability': 1.0,
                    'head_n_prototypes': 65536,
                    'head_bottleneck_dim': 256,
                    'head_nlayers': 3,
                    'head_hidden_dim': 2048
                })
                self.optim = type('obj', (object,), {
                    'epochs': 100,
                    'batch_size_per_gpu': 1,
                    'lr': 1e-4,
                    'clip_grad': 3.0
                })
                self.train = type('obj', (object,), {
                    'batch_size_per_gpu': 1,
                    'output_dir': './output',
                    'OFFICIAL_EPOCH_LENGTH': 1250
                })
        
        cfg = SimpleConfig()
        
        # デバイス設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用デバイス: {device}")
        
        # モデル作成
        print("モデル作成中...")
        model = SSLMetaArch(cfg).to(device)
        print(f"✓ モデル作成成功")
        
        # データ準備（ステップ3と同様）
        
        data_transform = DataAugmentationDINOMRI(
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8,
            global_crops_size=896,
            local_crops_size=256,
        )
        
        img_size = 896
        patch_size = 16
        n_tokens = (img_size // patch_size) ** 2
        
        mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
        
        collate_fn = partial(
            collate_data_and_cast,
            mask_ratio_tuple=(0.1, 0.5),
            mask_probability=1.0,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            dtype=torch.float32,
        )
        
        # 修正されたMriDatasetクラス（再定義）
        class MriDatasetWithTransform(MriDataset):
            def __init__(self, nii_path, transform=None):
                super().__init__(nii_path)
                self.transform = transform
            
            def __getitem__(self, idx):
                mri_slice = self.mri_data[:, :, idx]
                from load_data import numpy_to_pil
                pil_image = numpy_to_pil(mri_slice)
                
                if self.transform:
                    pil_image = self.transform(pil_image)
                
                return pil_image, ()
        
        dataset = MriDatasetWithTransform(nii_path, transform=data_transform)
        
        data_loader = make_data_loader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            seed=0,
            sampler_type=SamplerType.EPOCH,
            sampler_size=2,  # 2サンプルのみテスト
            drop_last=False,
            collate_fn=collate_fn,
        )
        
        # バッチ取得
        print("バッチ取得中...")
        data_iter = iter(data_loader)
        batch = next(data_iter)
        
        # モデルに入力
        print("モデルに入力中...")
        model.eval()
        with torch.no_grad():
            try:
                # フォワードパス実行
                outputs = model(batch)
                print(f"✓ モデル入力成功")
                print(f"  出力の型: {type(outputs)}")
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if hasattr(value, 'shape'):
                            print(f"  {key}: shape={value.shape}")
                        else:
                            print(f"  {key}: {type(value)}")
                
            except Exception as e:
                print(f"モデルフォワードパスでエラー: {e}")
                print("入力データの詳細:")
                for key, value in batch.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        if torch.is_tensor(value):
                            print(f"    min={value.min()}, max={value.max()}")
                raise
        
        print(f"✓ ステップ4完了: モデル入力確認成功")
        return True
        
    except Exception as e:
        print(f"✗ ステップ4失敗: {e}")
        traceback.print_exc()
        return False


def step5_training_steps_test(nii_path: str) -> bool:
    """
    ステップ5: 学習の最初の数ステップ確認
    """
    print("\n" + "="*60)
    print("ステップ5: 学習の最初の数ステップ確認")
    print("="*60)
    
    try:
        print("実際の学習設定を読み込み中...")
        
        # 実際のtrain_mri_simple.pyの設定を使用
        import argparse
        from dinov2.train.train import get_args_parser
        
        # 設定読み込み
        sys.argv = [
            "test_script",
            "--config-file", "dinov2/dinov2/configs/ssl_default_config.yaml",
            "--output-dir", "./test_output",
        ]
        
        dinov2_args = get_args_parser().parse_args()
        cfg = setup(dinov2_args)
        
        # MRI用の設定調整
        cfg.crops.global_crops_size = 896
        cfg.crops.local_crops_size = 256
        cfg.train.batch_size_per_gpu = 1
        cfg.train.num_workers = 0
        cfg.optim.epochs = 1  # 1エポックのみ
        
        print(f"✓ 設定読み込み成功")
        
        # 簡単な学習ステップテスト
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("モデル作成中...")
        model = SSLMetaArch(cfg).to(device)
        model.train()
        
        # オプティマイザー作成
        from dinov2.train.train import build_optimizer
        optimizer = build_optimizer(cfg, model.get_params_groups())
        
        print(f"✓ 学習準備完了")
        print(f"  デバイス: {device}")
        print(f"  バッチサイズ: {cfg.train.batch_size_per_gpu}")
        
        # データ準備（簡略版）
        print("データローダー作成中...")
        
        # 前のステップと同じデータ準備
        data_transform = DataAugmentationDINOMRI(
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8,
            global_crops_size=896,
            local_crops_size=256,
        )
        
        class MriDatasetWithTransform(MriDataset):
            def __init__(self, nii_path, transform=None):
                super().__init__(nii_path)
                self.transform = transform
            
            def __getitem__(self, idx):
                mri_slice = self.mri_data[:, :, idx]
                from load_data import numpy_to_pil
                pil_image = numpy_to_pil(mri_slice)
                
                if self.transform:
                    pil_image = self.transform(pil_image)
                
                return pil_image, ()
        
        dataset = MriDatasetWithTransform(nii_path, transform=data_transform)
        
        img_size = cfg.crops.global_crops_size
        patch_size = cfg.student.patch_size
        n_tokens = (img_size // patch_size) ** 2
        
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
            dtype=torch.half if torch.cuda.is_available() else torch.float32,
        )
        
        data_loader = make_data_loader(
            dataset=dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=0,
            shuffle=True,
            seed=0,
            sampler_type=SamplerType.EPOCH,
            sampler_size=3,  # 3ステップのみテスト
            drop_last=True,
            collate_fn=collate_fn,
        )
        
        print(f"✓ データローダー作成成功")
        
        # 学習ステップ実行
        print("学習ステップ実行中...")
        
        step_count = 0
        max_steps = 3
        
        for batch in data_loader:
            if step_count >= max_steps:
                break
                
            print(f"  ステップ {step_count + 1}/{max_steps}")
            
            try:
                # フォワード + バックワード
                optimizer.zero_grad()
                
                loss_dict = model.forward_backward(batch, teacher_temp=0.04)
                
                # 勾配計算確認
                has_grads = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        has_grads = True
                        break
                
                print(f"    ✓ フォワード/バックワード成功")
                print(f"    ✓ 勾配計算: {'成功' if has_grads else '失敗'}")
                
                # ロス値確認
                loss_values = {k: v.item() for k, v in loss_dict.items()}
                total_loss = sum(loss_values.values())
                print(f"    総ロス: {total_loss:.6f}")
                for k, v in loss_values.items():
                    print(f"      {k}: {v:.6f}")
                
                # オプティマイザーステップ
                optimizer.step()
                print(f"    ✓ オプティマイザーステップ完了")
                
                step_count += 1
                
            except Exception as e:
                print(f"    ✗ ステップ {step_count + 1} でエラー: {e}")
                raise
        
        print(f"✓ ステップ5完了: {step_count}ステップの学習成功")
        return True
        
    except Exception as e:
        print(f"✗ ステップ5失敗: {e}")
        traceback.print_exc()
        return False


def main():
    """
    メイン実行関数
    """
    print("="*70)
    print("load_data.py ステップバイステップ検証スクリプト")
    print("="*70)
    
    # MRIファイルのパス（デフォルト）
    default_nii_path = "/mnt/c/brain-segmentation/data/372/153035372/brats2025-gli-pre-challenge-trainingdata/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000"
    
    # 実際のファイルを探す
    nii_path = None
    if os.path.exists(default_nii_path):
        for file in os.listdir(default_nii_path):
            if file.endswith(".nii.gz") and "seg" not in file:
                nii_path = os.path.join(default_nii_path, file)
                break
    
    if not nii_path or not os.path.exists(nii_path):
        print(f"✗ MRIファイルが見つかりません: {default_nii_path}")
        print("使用方法: python test_load_data_step_by_step.py [nii_file_path]")
        if len(sys.argv) > 1:
            nii_path = sys.argv[1]
            if not os.path.exists(nii_path):
                print(f"✗ 指定されたファイルが存在しません: {nii_path}")
                return
        else:
            return
    
    print(f"使用するMRIファイル: {nii_path}")
    
    # 各ステップを実行
    steps = [
        ("基本データセット動作", step1_basic_dataset_test),
        ("データ拡張", step2_data_augmentation_test),
        ("DataLoader", step3_dataloader_test),
        ("モデル入力", step4_model_input_test),
        ("学習ステップ", step5_training_steps_test),
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        
        try:
            success = step_func(nii_path)
            results[step_name] = success
            
            if success:
                print(f"✓ {step_name}: 成功")
            else:
                print(f"✗ {step_name}: 失敗")
                print("以降のステップをスキップします")
                break
                
        except KeyboardInterrupt:
            print(f"\n中断されました")
            break
        except Exception as e:
            print(f"✗ {step_name}: 予期しないエラー: {e}")
            results[step_name] = False
            break
    
    # 結果サマリー
    print("\n" + "="*70)
    print("検証結果サマリー")
    print("="*70)
    
    for step_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失敗"
        print(f"{step_name:20}: {status}")
    
    # 全体の成功率
    success_count = sum(results.values())
    total_count = len(results)
    success_rate = success_count / total_count if total_count > 0 else 0
    
    print(f"\n成功率: {success_count}/{total_count} ({success_rate:.1%})")
    
    if success_rate == 1.0:
        print("\n🎉 すべてのステップが成功しました！train_mri_simple.pyを実行できます。")
    else:
        print(f"\n⚠️  一部のステップが失敗しました。上記のエラーを確認してください。")


if __name__ == "__main__":
    main() 