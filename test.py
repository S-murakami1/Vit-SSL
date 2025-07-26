#!/usr/bin/env python
"""
MRI DINOv2学習用デバッグテストスクリプト

ステップバイステップで各コンポーネントをテストし、問題を特定します。

使用方法:
python test.py --nii-path /path/to/your/mri.nii.gz [--auto] [--step N]

オプション:
--auto: 自動実行（各ステップで停止しない）
--step N: 指定したステップから開始
--batch-size: テスト用バッチサイズ（デフォルト: 2）
"""

import argparse
import sys
import os
import traceback
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MRIDebugTester:
    def __init__(self, nii_path, batch_size=2, auto_mode=False):
        self.nii_path = nii_path
        self.batch_size = batch_size
        self.auto_mode = auto_mode
        self.current_step = 0
        self.test_results = {}
        
    def log_step(self, step_num, description):
        """ステップの開始をログに記録"""
        print(f"\n{'='*60}")
        print(f"ステップ {step_num}: {description}")
        print(f"{'='*60}")
        self.current_step = step_num
        
    def wait_for_user(self, message="次のステップに進みますか？ [y/n/q]: "):
        """ユーザーの入力を待つ（自動モードでない場合）"""
        if self.auto_mode:
            return True
            
        while True:
            response = input(message).lower().strip()
            if response in ['y', 'yes', '']:
                return True
            elif response in ['n', 'no']:
                return False
            elif response in ['q', 'quit', 'exit']:
                print("テストを終了します。")
                sys.exit(0)
            else:
                print("y/n/q のいずれかを入力してください。")
    
    def test_step_1_basic_imports(self):
        """ステップ1: 基本的なインポートテスト"""
        self.log_step(1, "基本的なインポートテスト")
        
        try:
            # 基本ライブラリ
            import torch
            import numpy as np
            from PIL import Image
            import nibabel as nib
            
            print(f"✓ PyTorch version: {torch.__version__}")
            print(f"✓ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
                print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            print(f"✓ NumPy version: {np.__version__}")
            print(f"✓ PIL version: {Image.__version__}")
            print(f"✓ nibabel version: {nib.__version__}")
            
            self.test_results[1] = "SUCCESS"
            return True
            
        except Exception as e:
            print(f"✗ インポートエラー: {e}")
            traceback.print_exc()
            self.test_results[1] = f"FAILED: {e}"
            return False
    
    def test_step_2_load_data(self):
        """ステップ2: load_data.pyのテスト"""
        self.log_step(2, "load_data.pyのMriDatasetテスト")
        
        try:
            # load_data.pyをインポート
            from load_data import MriDataset, numpy_to_pil
            
            print(f"✓ load_data.pyのインポート成功")
            
            # MRIファイルの存在確認
            if not os.path.exists(self.nii_path):
                raise FileNotFoundError(f"MRIファイルが見つかりません: {self.nii_path}")
            
            print(f"✓ MRIファイル確認: {self.nii_path}")
            
            # MriDatasetの作成
            dataset = MriDataset(self.nii_path)
            print(f"✓ MriDataset作成成功")
            print(f"✓ データセットサイズ: {len(dataset)} スライス")
            
            # サンプルデータの取得
            sample_image, sample_target = dataset[0]
            print(f"✓ サンプルデータ取得成功")
            print(f"  - Image type: {type(sample_image)}")
            print(f"  - Image size: {sample_image.size}")
            print(f"  - Image mode: {sample_image.mode}")
            print(f"  - Target: {sample_target}")
            
            self.dataset = dataset  # 後のステップで使用
            self.test_results[2] = "SUCCESS"
            return True
            
        except Exception as e:
            print(f"✗ load_data.pyテストエラー: {e}")
            traceback.print_exc()
            self.test_results[2] = f"FAILED: {e}"
            return False
    
    def test_step_3_dinov2_imports(self):
        """ステップ3: DINOv2関連のインポートテスト"""
        self.log_step(3, "DINOv2関連インポートテスト")
        
        try:
            # DINOv2の基本インポート
            from dinov2.data import DataAugmentationDINO, MaskingGenerator, collate_data_and_cast
            print("✓ dinov2.data基本インポート成功")
            
            # MRI用ローダーのインポート
            from dinov2.data.loaders_mri import make_data_loader, make_dataset, SamplerType
            print("✓ dinov2.data.loaders_mriインポート成功")
            
            # 学習関連インポート
            from dinov2.train.ssl_meta_arch import SSLMetaArch
            print("✓ dinov2.train.ssl_meta_archインポート成功")
            
            # 設定関連インポート
            from dinov2.utils.config import setup
            print("✓ dinov2.utils.configインポート成功")
            
            self.test_results[3] = "SUCCESS"
            return True
            
        except Exception as e:
            print(f"✗ DINOv2インポートエラー: {e}")
            traceback.print_exc()
            self.test_results[3] = f"FAILED: {e}"
            return False
    
    def test_step_4_config_loading(self):
        """ステップ4: 設定ファイルの読み込みテスト"""
        self.log_step(4, "設定ファイル読み込みテスト")
        
        try:
            import argparse
            from dinov2.utils.config import setup
            
            # 設定ファイルのパス確認
            config_path = "dinov2/dinov2/configs/ssl_mri_vits_config.yaml"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
            
            print(f"✓ 設定ファイル確認: {config_path}")
            
            # 引数の準備
            class MockArgs:
                def __init__(self):
                    self.config_file = config_path
                    self.opts = [
                        f"train.dataset_path=MriDataset:path={self.nii_path}",
                        "train.output_dir=./test_output",
                        f"train.batch_size_per_gpu={self.batch_size}"
                    ]
            
            args = MockArgs()
            
            # 設定の読み込み
            cfg = setup(args)
            print("✓ 設定ファイル読み込み成功")
            
            # 重要な設定項目の確認
            print(f"  - Student arch: {cfg.student.arch}")
            print(f"  - Patch size: {cfg.student.patch_size}")
            print(f"  - Global crop size: {cfg.crops.global_crops_size}")
            print(f"  - Batch size: {cfg.train.batch_size_per_gpu}")
            print(f"  - Dataset path: {cfg.train.dataset_path}")
            
            self.cfg = cfg  # 後のステップで使用
            self.test_results[4] = "SUCCESS"
            return True
            
        except Exception as e:
            print(f"✗ 設定ファイル読み込みエラー: {e}")
            traceback.print_exc()
            self.test_results[4] = f"FAILED: {e}"
            return False
    
    def test_step_5_data_loader(self):
        """ステップ5: データローダーの作成テスト"""
        self.log_step(5, "データローダー作成テスト")
        
        try:
            from dinov2.data import DataAugmentationDINO, MaskingGenerator
            from dinov2.data.loaders_mri import make_data_loader, make_dataset, SamplerType
            from functools import partial
            from dinov2.data import collate_data_and_cast
            
            # データ変換の準備
            data_transform = DataAugmentationDINO(
                self.cfg.crops.global_crops_scale,
                self.cfg.crops.local_crops_scale,
                self.cfg.crops.local_crops_number,
                global_crops_size=self.cfg.crops.global_crops_size,
                local_crops_size=self.cfg.crops.local_crops_size,
            )
            print("✓ データ変換設定成功")
            
            # マスク生成器の準備
            img_size = self.cfg.crops.global_crops_size
            patch_size = self.cfg.student.patch_size
            n_tokens = (img_size // patch_size) ** 2
            
            mask_generator = MaskingGenerator(
                input_size=(img_size // patch_size, img_size // patch_size),
                max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
            )
            print("✓ マスク生成器設定成功")
            
            # コレート関数の準備
            collate_fn = partial(
                collate_data_and_cast,
                mask_ratio_tuple=self.cfg.ibot.mask_ratio_min_max,
                mask_probability=self.cfg.ibot.mask_sample_probability,
                n_tokens=n_tokens,
                mask_generator=mask_generator,
                dtype=torch.half,
            )
            print("✓ コレート関数設定成功")
            
            # データセットの作成
            dataset = make_dataset(
                dataset_str=self.cfg.train.dataset_path,
                transform=data_transform,
                target_transform=lambda _: (),
            )
            print(f"✓ データセット作成成功: {len(dataset)} サンプル")
            
            # データローダーの作成
            data_loader = make_data_loader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=2,  # テスト用に削減
                shuffle=True,
                seed=42,
                sampler_type=SamplerType.INFINITE,
                sampler_advance=0,
                drop_last=True,
                collate_fn=collate_fn,
            )
            print("✓ データローダー作成成功")
            
            self.data_loader = data_loader  # 後のステップで使用
            self.test_results[5] = "SUCCESS"
            return True
            
        except Exception as e:
            print(f"✗ データローダー作成エラー: {e}")
            traceback.print_exc()
            self.test_results[5] = f"FAILED: {e}"
            return False
    
    def test_step_6_model_creation(self):
        """ステップ6: モデル作成テスト"""
        self.log_step(6, "モデル作成テスト")
        
        try:
            import torch
            from dinov2.train.ssl_meta_arch import SSLMetaArch
            
            # GPU確認
            if not torch.cuda.is_available():
                print("⚠ CUDA不使用でテストを続行します")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda")
                print(f"✓ CUDA使用: {torch.cuda.get_device_name()}")
            
            # モデル作成
            model = SSLMetaArch(self.cfg).to(device)
            print("✓ SSLMetaArchモデル作成成功")
            
            # モデル情報の表示
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - Trainable parameters: {trainable_params:,}")
            print(f"  - Model device: {next(model.parameters()).device}")
            
            self.model = model  # 後のステップで使用
            self.device = device
            self.test_results[6] = "SUCCESS"
            return True
            
        except Exception as e:
            print(f"✗ モデル作成エラー: {e}")
            traceback.print_exc()
            self.test_results[6] = f"FAILED: {e}"
            return False
    
    def test_step_7_data_batch(self):
        """ステップ7: データバッチ取得テスト"""
        self.log_step(7, "データバッチ取得テスト")
        
        try:
            # データローダーから1バッチ取得
            data_iter = iter(self.data_loader)
            batch = next(data_iter)
            
            print("✓ データバッチ取得成功")
            print(f"  - Batch keys: {list(batch.keys())}")
            
            if "collated_global_crops" in batch:
                global_crops = batch["collated_global_crops"]
                print(f"  - Global crops shape: {global_crops.shape}")
                print(f"  - Global crops dtype: {global_crops.dtype}")
                print(f"  - Global crops device: {global_crops.device}")
            
            if "collated_local_crops" in batch:
                local_crops = batch["collated_local_crops"]
                print(f"  - Local crops shape: {local_crops.shape}")
                print(f"  - Local crops dtype: {local_crops.dtype}")
            
            self.sample_batch = batch  # 後のステップで使用
            self.test_results[7] = "SUCCESS"
            return True
            
        except Exception as e:
            print(f"✗ データバッチ取得エラー: {e}")
            traceback.print_exc()
            self.test_results[7] = f"FAILED: {e}"
            return False
    
    def test_step_8_forward_pass(self):
        """ステップ8: フォワードパステスト"""
        self.log_step(8, "モデルフォワードパステスト")
        
        try:
            # モデルを学習モードに設定
            self.model.train()
            
            # フォワードパス実行
            with torch.cuda.amp.autocast(enabled=True):
                loss_dict = self.model.forward_backward(self.sample_batch, teacher_temp=0.04)
            
            print("✓ フォワードパス成功")
            print(f"  - Loss keys: {list(loss_dict.keys())}")
            
            for key, value in loss_dict.items():
                print(f"  - {key}: {value.item():.6f}")
            
            # 総損失計算
            total_loss = sum(loss_dict.values())
            print(f"  - Total loss: {total_loss.item():.6f}")
            
            # NaN確認
            if any(torch.isnan(v) for v in loss_dict.values()):
                print("⚠ NaN detected in losses!")
                return False
            
            self.test_results[8] = "SUCCESS"
            return True
            
        except Exception as e:
            print(f"✗ フォワードパスエラー: {e}")
            traceback.print_exc()
            self.test_results[8] = f"FAILED: {e}"
            return False
    
    def test_step_9_optimizer_step(self):
        """ステップ9: オプティマイザーステップテスト"""
        self.log_step(9, "オプティマイザーステップテスト")
        
        try:
            import torch.optim as optim
            
            # オプティマイザー作成
            params_groups = self.model.get_params_groups()
            optimizer = optim.AdamW(
                params_groups, 
                betas=(self.cfg.optim.adamw_beta1, self.cfg.optim.adamw_beta2)
            )
            
            print("✓ オプティマイザー作成成功")
            
            # グラディエント確認
            total_grad_norm = 0
            param_count = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    param_count += 1
            
            total_grad_norm = total_grad_norm ** (1. / 2)
            print(f"  - Gradient norm: {total_grad_norm:.6f}")
            print(f"  - Parameters with gradients: {param_count}")
            
            # オプティマイザーステップ
            optimizer.step()
            optimizer.zero_grad()
            
            print("✓ オプティマイザーステップ成功")
            
            self.test_results[9] = "SUCCESS"
            return True
            
        except Exception as e:
            print(f"✗ オプティマイザーステップエラー: {e}")
            traceback.print_exc()
            self.test_results[9] = f"FAILED: {e}"
            return False
    
    def print_summary(self):
        """テスト結果のサマリー表示"""
        print(f"\n{'='*60}")
        print("テスト結果サマリー")
        print(f"{'='*60}")
        
        for step, result in self.test_results.items():
            status = "✓" if result == "SUCCESS" else "✗"
            print(f"ステップ {step}: {status} {result}")
        
        successful_steps = sum(1 for result in self.test_results.values() if result == "SUCCESS")
        total_steps = len(self.test_results)
        
        print(f"\n成功: {successful_steps}/{total_steps} ステップ")
        
        if successful_steps == total_steps:
            print("\n🎉 すべてのテストが成功しました！学習を開始できます。")
        else:
            print(f"\n⚠ {total_steps - successful_steps} 個のステップで問題が発生しました。")
    
    def run_tests(self, start_step=1):
        """テストの実行"""
        test_methods = [
            self.test_step_1_basic_imports,
            self.test_step_2_load_data,
            self.test_step_3_dinov2_imports,
            self.test_step_4_config_loading,
            self.test_step_5_data_loader,
            self.test_step_6_model_creation,
            self.test_step_7_data_batch,
            self.test_step_8_forward_pass,
            self.test_step_9_optimizer_step,
        ]
        
        for i, test_method in enumerate(test_methods[start_step-1:], start=start_step):
            success = test_method()
            
            if not success:
                print(f"\nステップ {i} で問題が発生しました。")
                if not self.wait_for_user("エラーを無視して続行しますか？ [y/n/q]: "):
                    break
            elif i < len(test_methods):
                if not self.wait_for_user():
                    break
        
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description='MRI DINOv2 Debug Tester')
    parser.add_argument('--nii-path', required=True, type=str,
                       help='Path to MRI .nii.gz file for testing')
    parser.add_argument('--auto', action='store_true',
                       help='Run all tests automatically without user prompts')
    parser.add_argument('--step', type=int, default=1, choices=range(1, 10),
                       help='Start from specific step (1-9)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for testing (default: 2)')
    
    args = parser.parse_args()
    
    print("MRI DINOv2 デバッグテストスクリプト")
    print("=" * 40)
    print(f"MRIファイル: {args.nii_path}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"自動実行: {'Yes' if args.auto else 'No'}")
    print(f"開始ステップ: {args.step}")
    
    if not args.auto:
        print("\n各ステップで実行を一時停止します。")
        print("y: 次へ, n: スキップ, q: 終了")
    
    tester = MRIDebugTester(
        nii_path=args.nii_path,
        batch_size=args.batch_size,
        auto_mode=args.auto
    )
    
    tester.run_tests(start_step=args.step)


if __name__ == "__main__":
    main() 