#!/usr/bin/env python
"""
MRI DINOv2学習実行スクリプト

使用方法:
python train_mri_simple.py --nii-path /path/to/your/mri.nii.gz --output-dir ./output_mri_vits

制約条件:
- 学習データはload_data.pyを経由して使用
- 分散学習は行わない  
- 使用モデルはVit-S
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='MRI DINOv2 Training Script')
    parser.add_argument('--nii-path', required=True, type=str, 
                       help='Path to MRI .nii.gz file for training')
    parser.add_argument('--output-dir', default='./output_mri_vits', type=str,
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--epochs', default=200, type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', default=8, type=int,
                       help='Batch size per GPU')
    
    args = parser.parse_args()
    
    # MRIファイルの存在確認
    if not os.path.exists(args.nii_path):
        print(f"エラー: MRIファイルが見つかりません: {args.nii_path}")
        sys.exit(1)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 設定ファイルのパス
    config_path = "dinov2/dinov2/configs/ssl_mri_vits_config.yaml"
    if not os.path.exists(config_path):
        print(f"エラー: 設定ファイルが見つかりません: {config_path}")
        sys.exit(1)
    
    # 学習実行コマンドの構築
    train_script = "dinov2/dinov2/train/train_mri.py"
    
    cmd = [
        sys.executable, train_script,
        "--config-file", config_path,
        "--output-dir", args.output_dir,
        f"train.dataset_path=MriDataset:path={args.nii_path}",
        f"train.output_dir={args.output_dir}",
        f"optim.epochs={args.epochs}",
        f"train.batch_size_per_gpu={args.batch_size}"
    ]
    
    print("=== MRI DINOv2 学習開始 ===")
    print(f"MRIファイル: {args.nii_path}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"エポック数: {args.epochs}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"実行コマンド: {' '.join(cmd)}")
    print("=" * 50)
    
    # 学習実行
    os.execvp(sys.executable, cmd)

if __name__ == "__main__":
    main() 