"""
Visualization of gesture to point cloud mappings.

Visualizes how gesture sequences are mapped to point cloud parts.
"""

from __future__ import print_function
import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from dataset import ShapeNetDataset
from model import PointNet_PartsSeg, Ges2PartsNet, Parts2ShapeNet
from visualization import drawpts, drawhand, drawparts
from functions import *

# Parser setup
parser = argparse.ArgumentParser(description='Gesture to point cloud visualization')
parser.add_argument('--model', type=str, default='model_Contratstive_Parts2Gesture', help='model directory path')
parser.add_argument('--dataset', type=str, default='dataset', help='dataset path')
parser.add_argument('--idx', type=int, default=0, help='sample index')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--save', action='store_true', help='save visualization images instead of showing')
parser.add_argument('--savedir', type=str, default='./output', help='directory to save images')

opt = parser.parse_args()
print(f"Options: {opt}")

# 保存ディレクトリの作成
if opt.save and not os.path.exists(opt.savedir):
    os.makedirs(opt.savedir)
    print(f"Created save directory: {opt.savedir}")

d = ShapeNetDataset(
    root=opt.dataset,
    split='train',
    data_augmentation=False)

pts_target, seglabel, hand_set, label, batch_weight = d[opt.idx]

hand = np.split(hand_set, 2, axis=0)
hand_l = hand[0] - hand[0][0]
hand_r = hand[1] - hand[1][0]
hand_l = hand_l.reshape(1, 69)
hand_r = hand_r.reshape(1, 69)

# モデル読み込み
model_pointnet, model_ges2parts, model_parts2shape = load_models(opt.model)

min_logit_sk_l = 0
min_logit_parts_l = 0
min_logit_sk_r = 0
min_logit_parts_r = 0

p_l = np.array([])
p_r = np.array([])
ges_l_path = ""
ges_r_path = ""
l_move = np.array([])
r_move = np.array([])

pts_dir = os.path.join(opt.dataset, "search/pts")

for pts_csv in os.listdir(pts_dir):
    pts_path = os.path.join(pts_dir, pts_csv)
    point_set = np.array(pd.read_csv(pts_path, header=None)).astype(np.float32)
    
    # ポイント正規化
    point_set, dist = normalize_points(point_set, num_points=2048)
    point_set = torch.from_numpy(point_set)

    # パーツセグメンテーション
    point = point_set.transpose(1, 0).contiguous()

    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _, all_feat= model_pointnet(point)
    pred_choice = pred.data.max(2)[1].cpu()

    pred_choice = pred_choice[0]

    if (np.count_nonzero(pred_choice==1) != 0) and (np.count_nonzero(pred_choice==2)!=0):
        # パーツ抽出
        pl, pr = extract_parts(point_set.cpu().numpy(), pred_choice.numpy(), sample_size=256)
        pl = pl.transpose(2, 1)
        pr = pr.transpose(2, 1)

        parts_l, parts_r, all_feat = pl, pr, all_feat
        
        # 中心位置を保存
        pl_move = np.expand_dims(np.mean(pl.numpy() if isinstance(pl, torch.Tensor) else pl, axis=2), 0)
        pr_move = np.expand_dims(np.mean(pr.numpy() if isinstance(pr, torch.Tensor) else pr, axis=2), 0)

        # hand_l, hand_r = hand_l.cuda(), hand_r.cuda()
        logit_per_sk_l, logit_per_parts_l, sk_feat_l, parts_feat_l = model_ges2parts(hand_l, parts_l, all_feat)
        logit_per_sk_r, logit_per_parts_r, sk_feat_r, parts_feat_r = model_ges2parts(hand_r, parts_r, all_feat)

        if logit_per_sk_l > min_logit_sk_l:
            pf_l = parts_feat_l
            p_l = parts_l
            l_move = pl_move
            min_logit_sk_l = logit_per_sk_l
            parts_name_l = pts_csv

        if  logit_per_sk_r >min_logit_sk_r:
            pf_r = parts_feat_r
            p_r = parts_r
            r_move = pr_move
            min_logit_sk_r = logit_per_sk_r
            parts_name_r = pts_csv

print("---推定結果---")
print("parts_l:", parts_name_l, "sim:", min_logit_sk_l)
print("parts_r:", parts_name_r, "sim:", min_logit_sk_r)

min_logit_per_p = 0
pred_pts_csv = ""
pred_pts = np.array([])

# 最高スコアの点群を検索

for pts_csv in os.listdir(pts_dir):
    pts_path = os.path.join(pts_dir, pts_csv)
    point_set=np.array(pd.read_csv(pts_path,header=None)).astype(np.float32)
    choice = np.random.choice(point_set.shape[0], 2048, replace=True)
    # リサンプリング
    point_set = point_set[choice, :]
    point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0)  # 中心
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
    point_set = point_set / dist  # スケーリング
    point_set = torch.from_numpy(point_set)
    # パーツセグメンテーション
    point = point_set.transpose(1, 0).contiguous()
    point = Variable(point.view(1, point.size()[0], point.size()[1]))
    pred, _, _, all_feat = model_pointnet(point)
    
    logit_per_p, logit_per_pts = model_parts2shape(pf_l, pf_r, all_feat)

    if logit_per_p > min_logit_per_p:
        pred_pts_csv=pts_path
        min_logit_per_p = logit_per_p
        pred_pts=point_set
    #print(logit_per_p,pts_csv)
    

ges_l = hand_l.cpu().numpy().reshape(23, 3) 
ges_r = hand_r.cpu().numpy().reshape(23, 3) 
print(p_l.shape, p_r.shape)

pred_parts_l = p_l[0].transpose(0, 1).contiguous().cpu().numpy() + l_move[0]
pred_parts_r = p_r[0].transpose(0, 1).contiguous().cpu().numpy() + r_move[0]

# パーツとポイントの比較

# Matplotlib で表示
# pointset, seglabel, predchoice
print("===========")
print("推定結果")
print("全体の形状は", pred_pts_csv)

fig, ax = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax.set_title('入力：ジェスチャー（赤：左手、青：右手）', fontsize=16) 

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

fig1, ax1 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax1.set_title('推定パーツ（赤：左手、青：右手）', fontsize=16) 
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_zlim(-1, 1)

fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax2.set_title('推定ポイントクラウド', fontsize=16) 
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)

fig3, ax3 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax3.set_title('推定ポイント＋入力ジェスチャー（重ね合わせ）', fontsize=16) 
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_zlim(-1, 1)

fig4, ax4 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax4.set_title('参照：教師データのポイントクラウド', fontsize=16) 
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax4.set_zlim(-1, 1)

# 入力
drawhand(hand=hand[0], color="red", ax=ax)
drawhand(hand=hand[1], color="blue", ax=ax)
# 予測

# パーツ左右
drawparts(pred_parts_l, ax=ax1, parts="left")
drawparts(pred_parts_r, ax=ax1, parts="right")

point_set = pred_pts.numpy()
drawparts(point_set, ax=ax2, parts="")

# 出力
# ジェスチャーとポイント
drawhand(hand=hand[0], color="red", ax=ax3)
drawhand(hand=hand[1], color="blue", ax=ax3)
drawparts(point_set, ax=ax3, parts="")

# 教師データ
drawparts(pts_target.numpy(), ax=ax4, parts="")
# 表示または保存
if opt.save:
    fig.savefig(os.path.join(opt.savedir, f'ges2pts_ans_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    fig1.savefig(os.path.join(opt.savedir, f'ges2pts_parts_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    fig2.savefig(os.path.join(opt.savedir, f'ges2pts_output_pts_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    fig3.savefig(os.path.join(opt.savedir, f'ges2pts_all_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    print(f"Images saved to {opt.savedir}")
    plt.show()
else:
    plt.show()



