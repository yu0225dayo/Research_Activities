
import numpy as np
from typing import Optional

# 手の骨格接続情報 (関節インデックス)
HAND_SKELETON = [0, 1, 2, 3, 4, 18,
                 0, 5, 6, 7, 19,
                 0, 8, 9, 10, 20,
                 0, 11, 12, 13, 21,
                 0, 14, 15, 16, 17, 22]


def drawpts(data: np.ndarray, label: np.ndarray, ax) -> None:
    """
    セグメンテーション結果を持つポイントクラウドを可視化。
    
    Args:
        data: ポイント座標 (N, 3)
        label: セグメンテーションラベル (N,) - 0: 背景, 1: 左手, 2: 右手
        ax: Matplotlib 3D軸オブジェクト
    """
    # クラスごとにポイントを分離
    zero = np.array([]).reshape(0, 3)  # 背景
    one = np.array([]).reshape(0, 3)   # 左手
    two = np.array([]).reshape(0, 3)   # 右手
    
    for i in range(len(data)):
        if int(label[i]) == 0:
            zero = np.vstack((zero, data[i]))
        elif int(label[i]) == 1:
            one = np.vstack((one, data[i]))
        else:
            two = np.vstack((two, data[i]))
    
    # 各クラスを色分けして描画
    if zero.shape[0] > 0:
        x, y, z = zero[:, 0], zero[:, 1], zero[:, 2]
        ax.scatter(x, y, z, c="green", s=1)  # 背景点

    if one.shape[0] > 0:
        x1, y1, z1 = one[:, 0], one[:, 1], one[:, 2]
        ax.scatter(x1, y1, z1, c="blue", s=1)  # 左手

    if two.shape[0] > 0:
        x2, y2, z2 = two[:, 0], two[:, 1], two[:, 2]
        ax.scatter(x2, y2, z2, c="red", s=1)  # 右手


def drawparts(data: np.ndarray, ax, parts: str = "") -> None:
    """
    パーツの3D点群を描画。
    
    Args:
        data: ポイント座標 (N, 3)
        ax: Matplotlib 3D軸オブジェクト
        parts: パーツ種別 ("left" または "right")
    """
    # 手の種別に応じて色を決定
    if parts == "left":
        color = "red"
    elif parts == "right":
        color = "blue"
    else:
        color = "green"
    
    # 有効なポイントのみを整形
    if data.shape[0] > 0:
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        ax.scatter(x, y, z, c=color, s=1)  # パーツの描画


def drawhand(hand: np.ndarray, color: str, ax, line_width: float = 1.0) -> None:
    """
    手骨格を3Dで描画。
    
    Args:
        hand: 手関節座標 (23, 3)
        color: 描画色 (e.g., "red", "blue")
        ax: Matplotlib 3D軸オブジェクト
        line_width: 線の幅
    """
    if hand.shape[0] == 0:
        return
    
    hx, hy, hz = hand[:, 0], hand[:, 1], hand[:, 2]
    
    # 5本の指ごとに描画ロジックを分離
    segment_indices = [
        (0, 1, 2, 3, 4, 18),  # 親指
        (0, 5, 6, 7, 19),     # 人差し指
        (0, 8, 9, 10, 20),    # 中指
        (0, 11, 12, 13, 21),  # 薬指
        (0, 14, 15, 16, 17, 22)  # 小指
    ]
    
    s = 0
    for i, seg_len in enumerate([6, 5, 5, 5, 6]):
        for k in range(seg_len - 1):
            j = k if i == 0 else s + k
            x = np.array([hx[HAND_SKELETON[j]], hx[HAND_SKELETON[j + 1]]])
            y = np.array([hy[HAND_SKELETON[j]], hy[HAND_SKELETON[j + 1]]])
            z = np.array([hz[HAND_SKELETON[j]], hz[HAND_SKELETON[j + 1]]])
            ax.plot(x, y, z, c=color, linewidth=line_width)
        s += seg_len
