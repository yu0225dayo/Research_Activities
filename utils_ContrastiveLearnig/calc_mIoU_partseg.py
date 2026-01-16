"""
Evaluation script for computing mIoU metrics on part segmentation.

Evaluates the performance of the PointNet-based part segmentation model.
"""

from __future__ import print_function
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import ShapeNetDataset
from model import PointNet_PartsSeg
from functions import load_models

parser = argparse.ArgumentParser(description='mIoU evaluation for part segmentation')
parser.add_argument('--model', type=str, default='Contratstive_Parts2Gesture', help='model directory path')
parser.add_argument('--dataset', type=str, default='dataset', help='dataset path')

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    split='val',
    data_augmentation=False)

database = ShapeNetDataset(
    root=opt.dataset,
    split='search',
    data_augmentation=False)

dloader = DataLoader(d, batch_size=10, shuffle=False)

databaseloader = DataLoader(database, batch_size=len(os.listdir(os.path.join(opt.dataset, "search/pts"))), shuffle=False)

# モデル読み込み
model_pointnet, _, _ = load_models(opt.model)

arr_mIoU_partseg = np.array([])
arr_name = np.array([])

pts_dir = os.path.join(opt.dataset, "search/pts")
label_dir = os.path.join(opt.dataset, "search/label")

for pts_csv in os.listdir(pts_dir):
    pts_path = os.path.join(pts_dir, pts_csv)
    point_set = np.array(pd.read_csv(pts_path, header=None)).astype(np.float32)
    choice = np.random.choice(point_set.shape[0], 2048, replace=True)
    # リサンプリング
    point_set = point_set[choice, :]
    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # 中心化
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist  # スケーリング

    point_set = torch.from_numpy(point_set)

    seg_csv = pts_csv.split(".")[0] + "_label.csv"
    seg_path = os.path.join(label_dir, seg_csv)
    seg_label = np.array(pd.read_csv(seg_path, header=None)).astype(np.int64)
    seg_label = seg_label[choice, :]
    seg_label = torch.from_numpy(seg_label)
    seg_label = seg_label.transpose(1, 0)

    # パーツセグメンテーション
    point = point_set.transpose(1, 0).contiguous()

    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _, all_feat = model_pointnet(point)
    pred_choice = pred.data.max(2)[1]
    print(seg_label.shape, pred_choice.shape)
    mIoU_partseg = torch.sum(torch.eq(seg_label, pred_choice)) / 2048
    print(mIoU_partseg)
    arr_mIoU_partseg = np.append(arr_mIoU_partseg, mIoU_partseg.item())
    arr_name = np.append(arr_name, pts_csv[:2])
    
arr_out = np.stack((arr_mIoU_partseg, arr_name), 1)
df = pd.DataFrame(arr_out)
df.to_csv("mIoU_partseg.csv", header=None, index=None)

classes = np.unique(arr_out[:, 1])
# 各クラスの平均を計算
averages = {}
for cls in classes:
    # クラスに対応する行を取得
    values = arr_out[arr_out[:, 1] == cls, 0].astype(np.float32)
    # 平均を計算
    averages[cls] = np.mean(values)

# 結果を表示
for cls, avg in averages.items():
    print(f"{cls} の平均: {avg}")




    
    
    