# Utils ContrastiveLearning

対照学習を用いた、物体の全体形状と両手把持ジェスチャの相互検索システム

## ファイル構成

### コア実装
- **model.py** - ニューラルネットワークモデル定義
  - PointNetDenseCls: パーツセグメンテーション
  - ContrastiveNet: ジェスチャー↔パーツ対比学習
  - PartsToPtsNet: パーツ↔ポイント写像学習

- **dataset.py** - データセット読み込みと前処理
  - ShapeNetDataset: カスタムデータセットクラス

- **functions.py** - 共有ユーティリティ関数
  - load_models(): モデル一括ロード
  - extract_parts(): パーツ抽出
  - normalize_hand(): 手データ正規化
  - normalize_points(): ポイント正規化

- **visualization.py** - 3D可視化関数
  - drawpts(): ポイントクラウド描画
  - drawhand(): 手スケルトン描画
  - drawparts(): パーツ描画

### 訓練・評価
- **train.py** - モデル訓練スクリプト
  - 3つのモデルの同時訓練
  - 評価指標に基づくモデル保存

- **calc_mIoU_partseg.py** - パーツセグメンテーション評価
- **calc_mIoU_part2ges.py** - パーツ→ジェスチャー評価

### 可視化
- **show_pts2gesture.py** - ポイント→ジェスチャーマッピング可視化
  - 7つの異なる視点から可視化
  - --save フラグで画像保存可能

- **show_ges2pts.py** - ジェスチャー→ポイント逆方向マッピング可視化
  - 4つの異なる視点から可視化
  - --save フラグで画像保存可能

- **show_cosinsim.py** - コサイン類似度ヒートマップ可視化
  - 左手ジェスチャー↔パーツ類似度
  - 右手ジェスチャー↔パーツ類似度
  - パーツ↔ポイント写像類似度
  - 各図を個別に保存可能

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

## 共通パラメータ

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


