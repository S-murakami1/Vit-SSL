#!/usr/bin/env python3
"""
MRIデータでDINOv2データパイプラインの基本動作確認
FSDPやcomplex設定を避けた簡単なテスト
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# プロジェクトのパスを追加
sys.path.append("dinov2")

# 基本的なインポート
from load_data import MriDataset, numpy_to_pil

def test_basic_dataset():
    """基本的なMriDatasetのテスト"""
    print("="*60)
    print("1. 基本的なMriDatasetのテスト")
    print("="*60)
    
    # MRIファイルのパスを探す
    nii_dir = r"/mnt/c/brain-segmentation/data/372/153035372/brats2025-gli-pre-challenge-trainingdata/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000"
    nii_path = None
    
    if os.path.exists(nii_dir):
        for file in os.listdir(nii_dir):
            if file.endswith(".nii.gz") and "seg" not in file:
                nii_path = os.path.join(nii_dir, file)
                break
    
    if not nii_path:
        print("❌ MRIファイルが見つかりません")
        return False
    
    print(f"使用ファイル: {nii_path}")
    
    try:
        # データセット作成
        dataset = MriDataset(nii_path)
        print(f"✅ データセット作成成功")
        print(f"   データセット長: {len(dataset)}")
        
        # サンプル取得
        sample_image, sample_target = dataset[0]
        print(f"✅ サンプル取得成功")
        print(f"   画像タイプ: {type(sample_image)}")
        print(f"   画像サイズ: {sample_image.size}")
        print(f"   画像モード: {sample_image.mode}")
        print(f"   ターゲット: {sample_target}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_data_augmentation():
    """データ拡張のテスト"""
    print("\n" + "="*60)
    print("2. データ拡張のテスト")
    print("="*60)
    
    try:
        # DINOv2のデータ拡張をインポート
        from dinov2.data.augmentations import DataAugmentationDINO
        
        # データ拡張設定
        data_transform = DataAugmentationDINO(
            global_crops_scale=(0.4, 1.0),
            local_crops_scale=(0.05, 0.4),
            local_crops_number=8,
            global_crops_size=224,  # より軽量なサイズ
            local_crops_size=96,
        )
        print(f"✅ データ拡張設定成功")
        
        # テスト用PIL画像作成
        test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # データ拡張適用
        augmented_data = data_transform(test_image)
        print(f"✅ データ拡張適用成功")
        print(f"   データタイプ: {type(augmented_data)}")
        print(f"   キー: {list(augmented_data.keys())}")
        
        if 'global_crops' in augmented_data:
            print(f"   Global crops: {len(augmented_data['global_crops'])} 個")
            print(f"   Global crop shape: {augmented_data['global_crops'][0].shape}")
        
        if 'local_crops' in augmented_data:
            print(f"   Local crops: {len(augmented_data['local_crops'])} 個")
            print(f"   Local crop shape: {augmented_data['local_crops'][0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mri_with_augmentation():
    """MRIデータとデータ拡張の組み合わせテスト"""
    print("\n" + "="*60)
    print("3. MRIデータ + データ拡張テスト")
    print("="*60)
    
    try:
        from dinov2.data.augmentations import DataAugmentationDINO
        
        # MRIファイルパス
        nii_dir = r"/mnt/c/brain-segmentation/data/372/153035372/brats2025-gli-pre-challenge-trainingdata/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000"
        nii_path = None
        
        if os.path.exists(nii_dir):
            for file in os.listdir(nii_dir):
                if file.endswith(".nii.gz") and "seg" not in file:
                    nii_path = os.path.join(nii_dir, file)
                    break
        
        if not nii_path:
            print("❌ MRIファイルが見つかりません")
            return False
        
        # データセット作成
        dataset = MriDataset(nii_path)
        
        # データ拡張設定
        data_transform = DataAugmentationDINO(
            global_crops_scale=(0.4, 1.0),
            local_crops_scale=(0.05, 0.4),
            local_crops_number=8,
            global_crops_size=224,
            local_crops_size=96,
        )
        
        # PIL画像取得
        pil_image, target = dataset[0]
        print(f"✅ MRI PIL画像取得: {pil_image.size}")
        
        # データ拡張適用
        augmented_data = data_transform(pil_image)
        print(f"✅ MRIにデータ拡張適用成功")
        
        # 結果確認
        print(f"   Global crops: {len(augmented_data['global_crops'])} 個")
        print(f"   Local crops: {len(augmented_data['local_crops'])} 個")
        
        for i, global_crop in enumerate(augmented_data['global_crops']):
            print(f"   Global crop {i}: {global_crop.shape}, dtype={global_crop.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_dataloader():
    """シンプルなDataLoaderテスト（collate関数なし）"""
    print("\n" + "="*60)
    print("4. シンプルなDataLoaderテスト")
    print("="*60)
    
    try:
        from torch.utils.data import DataLoader
        from dinov2.data.augmentations import DataAugmentationDINO
        
        # MRIファイルパス
        nii_dir = r"/mnt/c/brain-segmentation/data/372/153035372/brats2025-gli-pre-challenge-trainingdata/BraTS2025-GLI-PRE-Challenge-TrainingData/BraTS-GLI-00000-000"
        nii_path = None
        
        if os.path.exists(nii_dir):
            for file in os.listdir(nii_dir):
                if file.endswith(".nii.gz") and "seg" not in file:
                    nii_path = os.path.join(nii_dir, file)
                    break
        
        if not nii_path:
            print("❌ MRIファイルが見つかりません")
            return False
        
        # データ拡張付きデータセット
        class MriDatasetWithTransform(MriDataset):
            def __init__(self, nii_path, transform=None):
                super().__init__(nii_path)
                self.transform = transform
            
            def __getitem__(self, idx):
                mri_slice = self.mri_data[:, :, idx]
                pil_image = numpy_to_pil(mri_slice)
                
                if self.transform:
                    result = self.transform(pil_image)
                    return result, ()
                
                return pil_image, ()
        
        # データ拡張設定
        data_transform = DataAugmentationDINO(
            global_crops_scale=(0.4, 1.0),
            local_crops_scale=(0.05, 0.4),
            local_crops_number=8,
            global_crops_size=224,
            local_crops_size=96,
        )
        
        # データセット作成
        dataset = MriDatasetWithTransform(nii_path, transform=data_transform)
        print(f"✅ 拡張付きデータセット作成成功: {len(dataset)} サンプル")
        
        # カスタムcollate関数（簡単版）
        def simple_collate_fn(batch):
            """簡単なcollate関数"""
            samples, targets = zip(*batch)
            
            # 各サンプルは辞書形式
            global_crops = []
            local_crops = []
            
            for sample in samples:
                global_crops.extend(sample['global_crops'])
                local_crops.extend(sample['local_crops'])
            
            return {
                'global_crops': torch.stack(global_crops),
                'local_crops': torch.stack(local_crops),
                'targets': list(targets)
            }
        
        # DataLoader作成
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=simple_collate_fn
        )
        print(f"✅ DataLoader作成成功")
        
        # バッチ取得
        data_iter = iter(data_loader)
        batch = next(data_iter)
        
        print(f"✅ バッチ取得成功")
        print(f"   Global crops shape: {batch['global_crops'].shape}")
        print(f"   Local crops shape: {batch['local_crops'].shape}")
        print(f"   Targets: {len(batch['targets'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_model():
    """基本的なVision Transformerモデルテスト"""
    print("\n" + "="*60)
    print("5. 基本的なモデルテスト")
    print("="*60)
    
    try:
        # 簡単なVision Transformerモデルを作成
        from dinov2.models.vision_transformer import vit_small
        
        # モデル作成
        model = vit_small(
            patch_size=16,
            img_size=224,
        )
        print(f"✅ ViT-Smallモデル作成成功")
        
        # テスト用入力
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 224, 224)
        print(f"✅ テスト入力作成: {test_input.shape}")
        
        # フォワードパス
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ フォワードパス成功")
        print(f"   出力shape: {output['x_norm_clstoken'].shape}")
        print(f"   出力keys: {list(output.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("MRIデータでDINOv2パイプライン - 基本動作確認")
    print("="*70)
    
    tests = [
        ("基本データセット", test_basic_dataset),
        ("データ拡張", test_data_augmentation),
        ("MRI + データ拡張", test_mri_with_augmentation),
        ("シンプルDataLoader", test_simple_dataloader),
        ("基本モデル", test_basic_model),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
            
            if not success:
                print(f"\n⚠️ {test_name}で問題が発生しました。続行しますか？")
                response = input("続行する場合はEnterを押してください（q で終了）: ")
                if response.lower() == 'q':
                    break
        except Exception as e:
            print(f"❌ {test_name}で予期しないエラー: {e}")
            results[test_name] = False
            break
    
    # 結果サマリー
    print("\n" + "="*70)
    print("テスト結果サマリー")
    print("="*70)
    
    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"{test_name:20}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n成功率: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 すべてのテストが成功しました！")
        print("次のステップ: より複雑な学習設定を試すことができます")
    else:
        print(f"\n⚠️ 一部のテストが失敗しました。上記のエラーを確認してください。")

if __name__ == "__main__":
    main() 