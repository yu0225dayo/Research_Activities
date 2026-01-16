"""
共通ユーティリティ関数集

複数のビジュアライゼーションスクリプトで使用される共通関数を定義。
モデルロード、データ処理、描画機能を統合管理。
"""

from typing import Tuple, Optional
import os

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import ShapeNetDataset
from model import PointNet_PartsSeg, Ges2PartsNet, Parts2ShapeNet


def load_models(model_dir: str) -> Tuple:
    """
    学習済みモデルを読み込む。
    
    Args:
        model_dir: モデルディレクトリパス
    
    Returns:
        (PointNet_PartsSeg, Ges2PartsNet, Parts2ShapeNet) のタプル
    """
    # パーツセグメンテーションモデル
    pointnet_path = os.path.join(model_dir, "pointnet_model.pth")
    pointnet_state = torch.load(pointnet_path, weights_only=False)
    model_pointnet = PointNet_PartsSeg(k=3)
    model_pointnet.load_state_dict(pointnet_state)
    model_pointnet.eval()

    # ジェスチャー→パーツ対比学習モデル
    contrastive_path = os.path.join(model_dir, "contrastive_model.pth")
    contrastive_state = torch.load(contrastive_path, weights_only=False)
    model_ges2parts = Ges2PartsNet()
    model_ges2parts.load_state_dict(contrastive_state)
    model_ges2parts.eval()

    # パーツ→ポイント写像モデル
    parts2pts_path = os.path.join(model_dir, "parts2pts_model.pth")
    parts2pts_state = torch.load(parts2pts_path, weights_only=False)
    model_parts2shape = Parts2ShapeNet()
    model_parts2shape.load_state_dict(parts2pts_state)
    model_parts2shape.eval()

    return model_pointnet, model_ges2parts, model_parts2shape


def extract_parts(
    points: np.ndarray,
    pred_choice: np.ndarray,
    sample_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ポイントクラウドからパーツを抽出(centering)。
    
    Args:
        points: ポイントクラウド座標 (N, 3)
        pred_choice: セグメンテーション予測 (N,)
        sample_size: サンプルサイズ (デフォルト: 256)
    
    Returns:
        (左手パーツ, 右手パーツ) のタプル
    """
    parts_l_list = np.array([])
    parts_r_list = np.array([])

    # クラス1: 左手, クラス2: 右手でポイントを分離
    for j in range(len(pred_choice)):
        if pred_choice[j] == 2:
            parts_l_list = np.append(parts_l_list, points[j])
        if pred_choice[j] == 1:
            parts_r_list = np.append(parts_r_list, points[j])

    # サンプル数が不足している場合は補填
    while len(parts_l_list) <= (3 * sample_size):
        add_list = parts_l_list * 1.01
        parts_l_list = np.append(parts_l_list, add_list)

    while len(parts_r_list) <= (3 * sample_size):
        add_list = parts_r_list * 1.01
        parts_r_list = np.append(parts_r_list, add_list)

    # 形状を整形
    parts_l_list = parts_l_list.reshape(int(len(parts_l_list) / 3), 3)
    parts_r_list = parts_r_list.reshape(int(len(parts_r_list) / 3), 3)

    # サンプリング
    choice_l = np.random.choice(int(parts_l_list.shape[0]), sample_size, replace=True)
    choice_r = np.random.choice(int(parts_r_list.shape[0]), sample_size, replace=True)

    pl = parts_l_list[choice_l, :].reshape(1, sample_size, 3).astype(np.float32)
    pr = parts_r_list[choice_r, :].reshape(1, sample_size, 3).astype(np.float32)

    # 中心化
    pl_move = np.expand_dims(np.mean(pl, axis=1), 0)
    pr_move = np.expand_dims(np.mean(pr, axis=1), 0)
    pl = pl - pl_move
    pr = pr - pr_move

    return torch.from_numpy(pl), torch.from_numpy(pr)

def get_parts(
    points: np.ndarray,
    pred_choice: np.ndarray,
    sample_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ポイントクラウドからパーツを抽出。
    
    Args:
        points: ポイントクラウド座標 (N, 3)
        pred_choice: セグメンテーション予測 (N,)
        sample_size: サンプルサイズ (デフォルト: 256)
    
    Returns:
        (左手パーツ, 右手パーツ) のタプル
    """
    parts_l_list = np.array([])
    parts_r_list = np.array([])

    # クラス1: 左手, クラス2: 右手でポイントを分離
    for j in range(len(pred_choice)):
        if pred_choice[j] == 2:
            parts_l_list = np.append(parts_l_list, points[j])
        if pred_choice[j] == 1:
            parts_r_list = np.append(parts_r_list, points[j])

    # サンプル数が不足している場合は補填
    while len(parts_l_list) <= (3 * sample_size):
        add_list = parts_l_list * 1.01
        parts_l_list = np.append(parts_l_list, add_list)

    while len(parts_r_list) <= (3 * sample_size):
        add_list = parts_r_list * 1.01
        parts_r_list = np.append(parts_r_list, add_list)

    # 形状を整形
    parts_l_list = parts_l_list.reshape(int(len(parts_l_list) / 3), 3)
    parts_r_list = parts_r_list.reshape(int(len(parts_r_list) / 3), 3)

    # サンプリング
    choice_l = np.random.choice(int(parts_l_list.shape[0]), sample_size, replace=True)
    choice_r = np.random.choice(int(parts_r_list.shape[0]), sample_size, replace=True)

    pl = parts_l_list[choice_l, :].reshape(1, sample_size, 3).astype(np.float32)
    pr = parts_r_list[choice_r, :].reshape(1, sample_size, 3).astype(np.float32)

    # 中心化
    pl_move = np.expand_dims(np.mean(pl, axis=1), 0)
    pr_move = np.expand_dims(np.mean(pr, axis=1), 0)
    pl = pl 
    pr = pr 

    return torch.from_numpy(pl), torch.from_numpy(pr)

def normalize_hand(hand_data: np.ndarray) -> np.ndarray:
    """
    手ポーズデータを正規化 (手首を原点に)。
    
    Args:
        hand_data: 手関節データ (69,)
    
    Returns:
        正規化された手データ (69,)
    """
    hand = np.split(hand_data, 2, axis=0)
    hand_l = hand[0] - hand[0][0]
    hand_r = hand[1] - hand[1][0]
    hand_l = hand_l.reshape(1, 69)
    hand_r = hand_r.reshape(1, 69)
    return torch.from_numpy(np.concatenate([hand_l, hand_r], axis=0)).float()


def normalize_points(
    point_set: np.ndarray,
    num_points: int = 2048
) -> Tuple[np.ndarray, float]:
    """
    ポイントクラウドを正規化 (中心化とスケーリング)。
    
    Args:
        point_set: ポイントクラウド座標
        num_points: リサンプリング後のポイント数
    
    Returns:
        (正規化されたポイント, スケーリング係数)
    """
    # リサンプリング
    choice = np.random.choice(point_set.shape[0], num_points, replace=True)
    point_set = point_set[choice, :]

    # 中心化
    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)

    # スケーリング
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist

    return point_set.astype(np.float32), dist
