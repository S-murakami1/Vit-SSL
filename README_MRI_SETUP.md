# MRI DINOv2 学習セットアップガイド

独自のMRIデータを用いてDINOv2の学習を行うためのセットアップガイドです。

## 制約条件
- 学習データは`load_data.py`を経由して使用
- 分散学習は行わない
- 使用モデルはVit-S

## 修正されたファイル一覧

### 1. 設定ファイル
- **`dinov2/dinov2/configs/ssl_mri_vits_config.yaml`** (新規作成)
  - VitS用のMRI学習設定
  - 分散学習無効化 (sharding_strategy: NO_SHARD)
  - MRI画像用パラメータ調整

### 2. データローダー
- **`dinov2/dinov2/data/loaders_mri.py`** (loaders.pyを修正)
  - `load_data.py`のMriDatasetクラスを使用可能にする
  - MriDataset対応のパーサー追加

### 3. 学習スクリプト
- **`dinov2/dinov2/train/train_mri.py`** (train.pyを修正)
  - 分散学習関連処理を無効化
  - MRI用データローダーを使用
  - 単一GPU用ログ処理

### 4. 実行スクリプト
- **`train_mri_simple.py`** (新規作成)
  - MRI学習を実行するためのシンプルなスクリプト
  - コマンドライン引数で設定可能

### 5. データモジュール
- **`dinov2/dinov2/data/__init___mri.py`** (__init__.pyを修正)
  - MRI用ローダーをインポートできるように修正

## 使用方法

### 1. 基本的な実行
```bash
python train_mri_simple.py --nii-path /path/to/your/mri.nii.gz --output-dir ./output_mri_vits
```

### 2. パラメータ調整
```bash
python train_mri_simple.py \
    --nii-path /path/to/your/mri.nii.gz \
    --output-dir ./output_mri_vits \
    --epochs 100 \
    --batch-size 4
```

### 3. 直接実行 (詳細設定)
```bash
python dinov2/dinov2/train/train_mri.py \
    --config-file dinov2/dinov2/configs/ssl_mri_vits_config.yaml \
    --output-dir ./output_mri_vits \
    train.dataset_path=MriDataset:path=/path/to/your/mri.nii.gz
```

## 主な変更点

### 分散学習の無効化
- `sharding_strategy: SHARD_GRAD_OP` → `NO_SHARD`
- `distributed.all_reduce()` 処理を削除
- `sampler_type: SHARDED_INFINITE` → `INFINITE`

### VitS用設定
- `arch: vit_large` → `vit_small`
- `patch_size: 16` → `14`
- `head_n_prototypes: 65536` → `16384`
- `batch_size_per_gpu: 64` → `8`

### MRI画像用調整
- `global_crops_size: 224` → `896` (load_data.pyのTARGET_SIZEに合わせる)
- `mask_sample_probability: 0.5` → `0.3`
- より保守的なマスク比率とクロップ設定

## ファイル構造
```
Vit-SSL/
├── load_data.py                                    # MRIデータローダー
├── train_mri_simple.py                            # 実行スクリプト (新規)
├── README_MRI_SETUP.md                            # このファイル (新規)
├── dinov2/dinov2/configs/
│   └── ssl_mri_vits_config.yaml                   # VitS用設定 (新規)
├── dinov2/dinov2/data/
│   ├── __init___mri.py                            # MRI用データモジュール (修正版)
│   └── loaders_mri.py                             # MRI用ローダー (修正版)
└── dinov2/dinov2/train/
    └── train_mri.py                               # MRI用学習スクリプト (修正版)
```

## 動作確認

学習開始前に以下でMRIデータローダーの動作確認ができます：
```bash
python load_data.py
```

## 注意事項

1. **GPU環境**: CUDA対応GPUが必要です
2. **メモリ**: MRI画像サイズ(896x896)とバッチサイズに応じて十分なGPUメモリが必要
3. **依存関係**: nibabel, PIL, torch等の依存関係が必要
4. **データ形式**: .nii.gz形式のMRIファイルが必要 