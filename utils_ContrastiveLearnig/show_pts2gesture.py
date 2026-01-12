"""
Visualization of point cloud to gesture mappings.

Visualizes how point cloud parts are mapped to gesture sequences.
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
from model import PointNetDenseCls, ContrastiveNet
from visualization import drawpts, drawhand, drawparts
from functions import *

# Parser setup
parser = argparse.ArgumentParser(description='Point cloud to gesture visualization')
parser.add_argument('--model', type=str, default='model_Contratstive_Parts2Gesture', help='model directory path')
parser.add_argument('--dataset', type=str, default='dataset', help='dataset path')
parser.add_argument('--idx', type=int, default=300, help='sample index')
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
    split='search',
    data_augmentation=False)

point_set, seglabel, hand_set, label, batch_weight = d[opt.idx]

# モデル読み込み
classifier, sk_parts_classifier, _ = load_models(opt.model)

point = point_set.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))

pred, _, _, all_feat = classifier(point)
pred_choice = pred.data.max(2)[1].cpu()

print("処理中...")

print("推測  ラベル 1 ,2 ,0:", np.count_nonzero(pred_choice==1), np.count_nonzero(pred_choice==2), np.count_nonzero(pred_choice==0))
print("正解  ラベル 1 ,2 ,0:", np.count_nonzero(seglabel==1), np.count_nonzero(seglabel==2), np.count_nonzero(seglabel==0))

print(pred_choice.shape)
pred_choice = pred_choice[0]
print(pred_choice.shape)

# パーツ抽出
pl, pr = get_parts(point_set, pred_choice, sample_size=256)

# トランスポーズして特徴抽出準備
pl = pl.transpose(2, 1)
pr = pr.transpose(2, 1)

# 中心情報を保存（後で手のデータをオフセットするために使用）
pl_move = np.expand_dims(np.mean(pl.numpy(), axis=2), 0) if isinstance(pl, torch.Tensor) else np.expand_dims(np.mean(pl, axis=2), 0)
pr_move = np.expand_dims(np.mean(pr.numpy(), axis=2), 0) if isinstance(pr, torch.Tensor) else np.expand_dims(np.mean(pr, axis=2), 0)

parts_l, parts_r = pl, pr

min_logit_sk_l = 0
min_logit_parts_l = 0
min_logit_sk_r = 0
min_logit_parts_r = 0

ges_l = np.array([])
ges_r = np.array([])

ges_l_path = ""
ges_r_path = ""

hands_dir = os.path.join(opt.dataset, "search/hands")

for hands_csv in os.listdir(hands_dir):
    hands_path = os.path.join(hands_dir, hands_csv)
    hand_set=np.array(pd.read_csv(hands_path,header=None)).astype(np.float32)
    hand=np.split(hand_set,2,axis=0)
    hand_l = hand[0]-hand[0][0]
    hand_r = hand[1]-hand[1][0]
    hand_l=hand_l.reshape(1,69)
    hand_r=hand_r.reshape(1,69)
    hand_l , hand_r = torch.from_numpy(hand_l) , torch.from_numpy(hand_r) 
    #hand_l, hand_r = hand_l.cuda(), hand_r.cuda()
    logit_per_sk_l, logit_per_parts_l, sk_feat_l, parts_feat_l = sk_parts_classifier(hand_l, parts_l, all_feat)
    logit_per_sk_r, logit_per_parts_r, sk_feat_r, parts_feat_r = sk_parts_classifier(hand_r, parts_r, all_feat)

    if logit_per_parts_l > min_logit_parts_l:
        ges_l = hand_l
        min_logit_parts_l = logit_per_parts_l
        ges_l_path = hands_path

    if logit_per_parts_r > min_logit_parts_r:
        ges_r = hand_r
        min_logit_parts_r = logit_per_parts_r
        ges_r_path = hands_path

print("---推定結果---")
print("ges_l:",ges_l_path, "sim:", min_logit_parts_l)
print("ges_r:",ges_r_path, "sim:", min_logit_parts_r)

ges_l = ges_l.cpu().numpy().reshape(23,3) 
ges_l = ges_l - np.expand_dims(np.mean(ges_l, axis = 0), 0) + pl_move[0]

ges_r = ges_r.cpu().numpy().reshape(23,3) 
ges_r = ges_r - np.expand_dims(np.mean(ges_r, axis = 0), 0) + pr_move[0]

#matplotlibで表示しよう。

fig, ax = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax.set_title('入力: 正解ラベル', fontsize=20) 
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

fig1, ax1 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax1.set_title('パーツセグメンテーション予測', fontsize=20) 
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)
ax1.set_zlim(-1,1)

fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax2.set_title('抽出されたパーツ', fontsize=20) 
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)
ax2.set_zlim(-1,1)

fig7, ax7 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax7.set_title('入力ポイントクラウド', fontsize=20) 
ax7.set_xlim(-1,1)
ax7.set_ylim(-1,1)
ax7.set_zlim(-1,1)


fig3, ax3 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax3.set_title('推定されたジェスチャー（左右）', fontsize=20) 
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_zlim(-1, 1)

fig4, ax4 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax4.set_title('左パーツ ← → 左ジェスチャー対応', fontsize=20) 
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax4.set_zlim(-1, 1)

fig5, ax5 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax5.set_title('右パーツ ← → 右ジェスチャー対応', fontsize=20) 
ax5.set_xlim(-1.5, 1.5)
ax5.set_ylim(-1.5, 1.5)
ax5.set_zlim(-1, 1)

fig6, ax6 = plt.subplots(figsize=(6, 6), facecolor='white', 
                       subplot_kw={'projection': '3d'})
ax6.set_title('全体：ポイント＋推定ジェスチャー', fontsize=20) 
ax6.set_xlim(-1.5, 1.5)
ax6.set_ylim(-1.5, 1.5)
ax6.set_zlim(-1, 1)

# 可視化
point_set = point_set.numpy()
drawpts(point_set, seglabel, ax=ax)
drawparts(point_set, ax=ax7, parts="")
# 予測結果
drawpts(point_set, pred_choice, ax=ax1)
# 左右パーツ
drawparts(parts_l, ax=ax2, parts="left")
drawparts(parts_r, ax=ax2, parts="right")

# 出力ジェスチャー
drawhand(hand=ges_l, color="red", ax=ax3)
drawhand(hand=ges_r, color="blue", ax=ax3)

# 左パーツと左ジェスチャー
drawhand(hand=ges_l, color="red", ax=ax4)
drawparts(parts_l, ax=ax4, parts="left")

# 右パーツと右ジェスチャー
drawhand(hand=ges_r, color="blue", ax=ax5)
drawparts(parts_r, ax=ax5, parts="right")

#out pts and ges
drawhand(hand=ges_l,color="red",ax=ax6)
drawhand(hand=ges_r,color="blue",ax=ax6)
drawparts(point_set,ax=ax6,parts="")

# 表示または保存
if opt.save:
    fig.savefig(os.path.join(opt.savedir, f'pts2gesture_ans_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    fig1.savefig(os.path.join(opt.savedir, f'pts2gesture_partseg_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    fig2.savefig(os.path.join(opt.savedir, f'pts2gesture_parts_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    fig3.savefig(os.path.join(opt.savedir, f'pts2gesture_output_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    fig4.savefig(os.path.join(opt.savedir, f'pts2gesture_parts_l_ges_l_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    fig5.savefig(os.path.join(opt.savedir, f'pts2gesture_parts_r_ges_r_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    fig6.savefig(os.path.join(opt.savedir, f'pts2gesture_all_{opt.idx}.png'), dpi=100, bbox_inches='tight')
    print(f"Images saved to {opt.savedir}")
    plt.show()
else:
    plt.show()



