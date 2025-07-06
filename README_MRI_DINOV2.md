# MRI画像でDINOv2を学習する方法

このガイドでは、MRI画像データを使用してDINOv2の自己教師あり学習を実行する方法を説明します。

## 前提条件

- Python 3.8以上
- PyTorch 1.12以上
- CUDA対応GPU
- 必要なライブラリ: `nibabel`, `PIL`, `torchvision`, `numpy`

## ファイル構成

```
.
├── load_data.py                      # MRIデータセットクラス（修正済み）
├── dinov2/dinov2/data/loaders.py     # データローダー統合（修正済み）
├── dinov2/dinov2/data/augmentations.py # データ拡張（MRI用追加）
├── dinov2/dinov2/configs/ssl_mri_config.yaml # MRI用設定ファイル
├── train_mri_dinov2.py               # 学習実行スクリプト
└── README_MRI_DINOV2.md              # このファイル
```

## 主な修正点

### 1. `load_data.py`の修正
- **DINOv2対応**: `(image, target)`タプルの返却
- **PIL.Image形式**: RGB変換でDINOv2のデータ拡張に対応
- **複数ファイル対応**: `MriDatasetForDINOv2`クラス追加

### 2. データローダー統合
- **カスタムデータセット登録**: `MriDataset`を`make_dataset`関数に統合
- **パス指定**: `"MriDataset:path=/path/to/file.nii.gz"`形式での指定

### 3. MRI用データ拡張
- **医用画像特化**: `DataAugmentationDINOMRI`クラス追加
- **適切な拡張**: 色彩変換除去、回転・ブラー調整
- **構造保持**: 脳の重要な構造を保持するスケール設定

### 4. 設定ファイル
- **大画像対応**: 1024x1024サイズ、バッチサイズ調整
- **長期学習**: 200エポック、ウォームアップ20エポック
- **保守的設定**: マスク比率、学習率の調整

## 使用方法

### 1. 基本的な学習実行

```bash
python train_mri_dinov2.py \
    --nii-path "/path/to/your/mri_file.nii.gz" \
    --output-dir "./output_mri_training"
```

### 2. 複数GPU での学習

```bash
python train_mri_dinov2.py \
    --nii-path "/path/to/your/mri_file.nii.gz" \
    --output-dir "./output_mri_training" \
    --gpus 2
```

### 3. 学習の再開

```bash
python train_mri_dinov2.py \
    --nii-path "/path/to/your/mri_file.nii.gz" \
    --output-dir "./output_mri_training" \
    --resume
```

### 4. 手動設定での学習

```bash
python -m dinov2.run.train.train \
    --config-file dinov2/dinov2/configs/ssl_mri_config.yaml \
    --output-dir "./output_mri_training"
```

## 設定のカスタマイズ

### データセットパス
```yaml
train:
  dataset_path: "MriDataset:path=/path/to/your/data.nii.gz"
  # または複数ファイル
  # dataset_path: "MriDataset:paths=/path/1.nii.gz,/path/2.nii.gz"
```

### 画像サイズ
```yaml
crops:
  global_crops_size: 1024  # または 512
  local_crops_size: 256    # または 128
```

### バッチサイズ（GPU メモリに応じて調整）
```yaml
train:
  batch_size_per_gpu: 4  # または 2, 1
```

### モデルサイズ
```yaml
student:
  arch: vit_large  # または vit_base, vit_small
```

## 学習の監視

### ログの確認
```bash
# 学習進捗の確認
tail -f output_mri_training/log.txt

# 損失値の確認
grep "loss" output_mri_training/log.txt
```

### チェックポイント
- **保存場所**: `output_mri_training/checkpoint_*.pth`
- **保存頻度**: 10エポックごと（設定で変更可能）
- **教師モデル**: `output_mri_training/eval/*/teacher_checkpoint.pth`

## トラブルシューティング

### メモリ不足の場合
```yaml
train:
  batch_size_per_gpu: 1  # バッチサイズを削減
crops:
  global_crops_size: 512  # 画像サイズを削減
```

### 学習が不安定な場合
```yaml
optim:
  base_lr: 0.0001  # 学習率を下げる
  warmup_epochs: 30  # ウォームアップを延長
```

### データセット読み込みエラー
- NIIファイルのパスが正しいか確認
- `nibabel`ライブラリがインストールされているか確認
- ファイルの読み込み権限があるか確認

## 学習後の活用

### 学習済みモデルの読み込み
```python
import torch
from dinov2.models import build_model_from_cfg

# 教師モデルの読み込み
checkpoint = torch.load("output_mri_training/eval/100000/teacher_checkpoint.pth")
model = build_model_from_cfg(config)
model.load_state_dict(checkpoint["teacher"])
```

### 特徴抽出
```python
# 特徴マップの抽出
with torch.no_grad():
    features = model.forward_features(mri_batch)
    patch_tokens = features['x_norm_patchtokens']
    cls_token = features['x_norm_clstoken']
```

## 推奨設定

### 小規模データセット（< 10,000 スライス）
```yaml
optim:
  epochs: 100
  base_lr: 0.0001
train:
  batch_size_per_gpu: 2
```

### 中規模データセット（10,000-50,000 スライス）
```yaml
optim:
  epochs: 200
  base_lr: 0.0005
train:
  batch_size_per_gpu: 4
```

### 大規模データセット（> 50,000 スライス）
```yaml
optim:
  epochs: 300
  base_lr: 0.001
train:
  batch_size_per_gpu: 8
```

## 注意事項

1. **データのプライバシー**: MRI画像には患者情報が含まれる可能性があるため、適切な匿名化を実施してください。
2. **計算リソース**: 1024x1024の画像では大量のGPUメモリが必要です。
3. **学習時間**: 完全な学習には数日から数週間かかる場合があります。
4. **評価**: 下流タスク（セグメンテーション、分類など）で学習済みモデルの性能を評価してください。

## サポート

問題が発生した場合は、以下を確認してください：
- ログファイルのエラーメッセージ
- GPU メモリ使用量
- データセットの形式とサイズ
- 環境変数の設定 