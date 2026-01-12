# Parts2Gesture

接触部位形状を介した、全体形状と両手把持ジェスチャの相互検索システム

## 研究背景

物体の把持姿勢認識には、物体全体の形状が重要です。しかし、バスケットとやかんのように形状が異なる物体でも把持姿勢が類似することがあります。本研究では、**手との接触部位形状**に着目し、全体形状のパーツセグメンテーションを通じて、把持姿勢とパーツ形状の関係性を学習します。

## プロジェクト構成

```
Parts2Gesture/
│
├── model_Contratstive_Parts2Gesture/              # 学習済みモデルディレクトリ
│   ├── pointnet_model_*.pth                       # パーツセグメンテーション
│   ├── contrastive_model_*.pth                    # ジェスチャー↔パーツ対比学習
│   └── parts2pts_model_*.pth                      # パーツ↔ポイント写像学習
│
└── utils_ContrastiveLearnig/                      # メイン実装モジュール
    │
    ├── model.py                                   # ニューラルネットワーク定義
    ├── dataset.py                                 # データセット読み込み
    ├── functions.py                               # 共有ユーティリティ関数
    ├── visualization.py                           # 3D可視化関数
    │
    ├── train.py                                   # モデル訓練スクリプト
    ├── calc_mIoU_partseg.py                       # パーツセグメンテーション評価
    ├── calc_mIoU_part2ges.py                      # パーツ→ジェスチャー評価
    │
    ├── show_pts2gesture.py                        # ポイント→ジェスチャー可視化
    ├── show_ges2pts.py                            # ジェスチャー→ポイント可視化
    └── show_cosinsim.py                           # コサイン類似度ヒートマップ
```

### モデル (model.py)

| モデル名 | 役割 | 入力 | 出力 |
|---------|------|------|------|
| **PointNetDenseCls** | パーツセグメンテーション | ポイントクラウド (B×2048×3) | 3クラス分類 (背景/左手/右手) |
| **ContrastiveNet** | ジェスチャー↔パーツ対照学習 | 手データ + パーツ + 全体特徴 | 類似度スコア行列 |
| **PartsToPtsNet** | パーツ↔ポイント対照学習 | 左右パーツ特徴 + 全体特徴 | 類似度スコア行列 |

## データセット詳細

### ディレクトリ構成

```
dataset/
├── train/                   # 訓練用データ（約1000サンプル）
│   ├── pts/                 # ポイントクラウド (CSV形式, 2048×3)
│   ├── pts_label/           # セグメンテーション教師信号 (CSV形式, 2048×1)
│   │                        # 0:背景, 1:左手接触, 2:右手接触
│   └── hands/               # 手ジェスチャー (CSV形式, 2×69)
│                            # 各手23個関節×3軸 
│
├── val/                     # 検証用データ（約200サンプル）
│   ├── pts/
│   ├── pts_label/
│   └── hands/
│
├── search/                  # 検索用データベース（参照セット）
│   ├── pts/
├── pts_label/
│   └── hands/


### データセット分割

| セット | 用途 | サンプル数 | 特徴 |
|--------|------|-----------|------|
| train | モデル訓練 | ~1000 | データ拡張適用 |
| val | ハイパーパラメータ調整 | ~200 | データ拡張なし |
| search | 検索データベース | ~500 | 事前計算特徴用 |

### データ拡張（訓練時のみ）
- 回転: z軸周り 0-360° ランダム回転
- スケーリング: 0.9-1.1倍のランダムスケール
- ノイズ: 標準偏差 0.02 のガウシアンノイズ

### 正規化処理
- ポイント: 中心を原点に、最大距離を1に正規化
- 手: 手首を原点に、各関節を正規化

## 使用方法

### 訓練

```bash
python train.py \
    --dataset path/to/dataset \
    --batchSize 16 \
    --nepoch 100 \
    --model path/to/model_dir  # オプション：既存モデルから続行
```

### 評価

```bash
# パーツセグメンテーション評価
python calc_mIoU_partseg.py \
    --model path/to/model \
    --dataset path/to/dataset

# パーツ→ジェスチャー評価
python calc_mIoU_part2ges.py \
    --model path/to/model \
    --dataset path/to/dataset
```

### 可視化（画面表示）
idxを指定し、任意のジェスチャ・全体形状から相互検索可能
```bash
# ポイント→ジェスチャー可視化
python show_pts2gesture.py \
    --model path/to/model \
    --dataset path/to/dataset \
    --idx 0

# ジェスチャー→ポイント可視化
python show_ges2pts.py \
    --model path/to/model \
    --dataset path/to/dataset \
    --idx 0

# コサイン類似度可視化
python show_cosinsim.py \
    --model path/to/model \
    --dataset path/to/dataset
```

### 可視化（画像保存）

```bash
# ポイント→ジェスチャーの7つの図を保存
python show_pts2gesture.py \
    --model path/to/model \
    --dataset path/to/dataset \
    --save --savedir ./output

# ジェスチャー→ポイントの4つの図を保存
python show_ges2pts.py \
    --model path/to/model \
    --dataset path/to/dataset \
    --save --savedir ./output

# 3つのコサイン類似度ヒートマップを保存
python show_cosinsim.py \
    --model path/to/model \
    --dataset path/to/dataset \
    --save --savedir ./output
```

## 出力形式

### show_pts2gesture.py の保存ファイル
- pts2gesture_ans_{idx}.png - 正解ラベル
- pts2gesture_partseg_{idx}.png - パーツセグメンテーション予測
- pts2gesture_parts_{idx}.png - 抽出されたパーツ
- pts2gesture_output_{idx}.png - 推定ジェスチャー
- pts2gesture_parts_l_ges_l_{idx}.png - 左パーツ↔左ジェスチャー対応
- pts2gesture_parts_r_ges_r_{idx}.png - 右パーツ↔右ジェスチャー対応
- pts2gesture_all_{idx}.png - 全体表示

### show_ges2pts.py の保存ファイル
- ges2pts_ans_{idx}.png - 入力ジェスチャー
- ges2pts_parts_{idx}.png - 推定パーツ
- ges2pts_output_pts_{idx}.png - 推定ポイントクラウド
- ges2pts_all_{idx}.png - 全体表示

### show_cosinsim.py の保存ファイル
- cosinsim_left_hand.png - 左手ジェスチャー↔パーツ類似度ヒートマップ
- cosinsim_right_hand.png - 右手ジェスチャー↔パーツ類似度ヒートマップ
- cosinsim_parts_to_points.png - パーツ↔ポイント写像類似度ヒートマップ

## 主な特徴

- ✅ **PointNetベース**: 順序不変な点群処理
- ✅ **マルチモーダル学習**: ジェスチャー + パーツ + ポイント
- ✅ **対照学習**: 相互検索システム(CLIPを参考に)

## 必要環境

- Python 3.7+
- PyTorch 1.9+
- NumPy, Matplotlib, OpenCV, Pandas, tqdm

## インストール

```bash
pip install torch numpy matplotlib opencv-python pandas plyfile tqdm
```
