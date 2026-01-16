"""
Visualization of cosine similarity matrices from contrastive learning models.

Generates heatmap images of similarity scores between gestures and point cloud parts.
"""

from __future__ import print_function
import argparse
import os

import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from dataset import ShapeNetDataset
from model import PointNet_PartsSeg, Ges2PartsNet, Parts2ShapeNet
from functions import load_models

parser = argparse.ArgumentParser(description='Cosine similarity visualization')
parser.add_argument('--model', type=str, default='Contratstive_Parts2Gesture', help='model path')
parser.add_argument('--samplesize', type=int, default=16, help='input batch size')
parser.add_argument('--dataset', type=str, default='dataset', help='dataset path')
parser.add_argument('--save', action='store_true', help='save visualization images instead of showing')
parser.add_argument('--savedir', type=str, default='./output', help='directory to save images')

opt = parser.parse_args()
print(opt)

# 保存ディレクトリの作成
if opt.save and not os.path.exists(opt.savedir):
    os.makedirs(opt.savedir)
    print(f"Created save directory: {opt.savedir}")

d = ShapeNetDataset(
    root=opt.dataset,
    split='val',
    data_augmentation=False)

database =  ShapeNetDataset(
    root=opt.dataset,
    split='search',
    data_augmentation=False)

dloader = DataLoader(d, batch_size=10, shuffle=False)
databaseloader = DataLoader(database, batch_size=len(os.listdir(os.path.join(opt.dataset, "search/pts"))), shuffle=False)

# モデル読み込み
model_pointnet, model_ges2parts, model_parts2shape = load_models(opt.model)

pts, seg, hands, label, batch_w = next(iter(dloader))
pts_base, seg_base, hands_base, label_base, batch_w_base = next(iter(databaseloader))

# すべてのジェスチャーの中から最もマッチするものを見つける
def parts2ges_all(p, g, seg):
    outarr_p_size=p.shape[0]
    outarr_g_size=g.shape[0]
    points =p.transpose(2, 1).contiguous()
    pred, _, _,all_feat = model_pointnet(points)
    pred = pred.view(-1, 3)

    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().data.numpy()
    # pred_choice 1: 左手, 2: 右手
    
    points=points.transpose(1, 2).cpu().data.numpy()
    pred_np=pred_np.reshape(outarr_p_size,2048,1)
    pl=np.array([])
    pr=np.array([])
    for batch in range(outarr_p_size):
        count=0
        parts_l_list=np.array([])
        parts_r_list=np.array([])
        #print(batch,np.count_nonzero(pred_np[batch]==1),np.count_nonzero(pred_np[batch]==2) )
        target_l=pred_np
        target_r=pred_np
        if np.count_nonzero(pred_np[batch]==2)<=10:    
            target_l=seg
            print("セグメント失敗",batch)
        else:
            target_l=pred_np
        if np.count_nonzero(pred_np[batch]==1)<=10:
            target_r=seg
            print("セグメント失敗",batch)
        else:
            target_r=pred_np

        for j in range(2048):
            if target_l[batch][j]==2:
                parts_l_list=np.append(parts_l_list,points[batch][j])
            if target_r[batch][j]==1:
                parts_r_list=np.append(parts_r_list,points[batch][j])

        while len(parts_l_list)<=(3 * 256):
            count+=1
            add_list=parts_l_list*1.01
            parts_l_list=np.append(parts_l_list,add_list)

        while len(parts_r_list)<=(3 * 256):
            count+=1
            add_list=parts_r_list*1.01
            parts_r_list=np.append(parts_r_list,add_list)
        
        parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3)             
        choice = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True)                
        pl=np.append(pl,parts_l_list[choice,:])
        parts_r_list=parts_r_list.reshape(int(len(parts_r_list)/3),3)             
        choice = np.random.choice(int(parts_r_list.shape[0]), 256, replace=True)                
        pr=np.append(pr,parts_r_list[choice,:])
        pl = pl - np.expand_dims(np.mean(pl, axis = 0), 0)
        pr = pr - np.expand_dims(np.mean(pr, axis = 0), 0)
    pl=pl.reshape(outarr_p_size,256,3).astype(np.float32)
    pr=pr.reshape(outarr_p_size,256,3).astype(np.float32)

    pl=torch.from_numpy(pl)
    pr=torch.from_numpy(pr)
    pl=pl.transpose(2,1)
    pr=pr.transpose(2,1)
    # ------------------------------------------------

    hand=np.split(g,2,axis=1)
    hand_l=hand[0]
    hand_r=hand[1]
    # 手首を0に正規化
    for k in range(outarr_g_size):
        hand_l[k] = hand_l[k] - hand_l[k][0]
        hand_r[k] = hand_r[k] - hand_r[k][0]

    hand_l=hand_l.reshape(outarr_g_size,69)
    hand_r=hand_r.reshape(outarr_g_size,69)
    sim_pl_gl, _, _, parts_l_feat = model_ges2parts(hand_l, pl, all_feat)
    sim_pr_gr, _, _, parts_r_feat = model_ges2parts(hand_r, pr, all_feat)

    sim_pl_gr, _, _, parts_l_feat = model_ges2parts(hand_l, pr, all_feat)
    sim_pr_gl, _, _, parts_r_feat = model_ges2parts(hand_r, pl, all_feat)
    sim_pl_per_g = torch.cat((sim_pl_gl,sim_pl_gr),dim=1)
    sim_pr_per_g = torch.cat((sim_pr_gl,sim_pr_gr),dim=1)
    
    sim_parts_per_ges = torch.cat((sim_pl_per_g, sim_pr_per_g),dim=0).transpose(1,0)
    sim_parts_per_ges = sim_parts_per_ges.detach().numpy()
    print("デバッグ出力")
    mat=sim_parts_per_ges.shape[1]//sim_parts_per_ges.shape[0]
    if mat==0:
        mat=1
    print("===========")

    img=np.zeros((mat*sim_parts_per_ges.shape[0],sim_parts_per_ges.shape[1]))

    for k in range(mat):
        for h in range(sim_parts_per_ges.shape[0]):
            for w in range(sim_parts_per_ges.shape[1]):
                img[sim_parts_per_ges.shape[0]*k+h][w]= sim_parts_per_ges[h][w]

    block_size = 3
    image_blocked = np.kron(img, np.ones((block_size, block_size)))
    image = image_blocked

    return image

# すべて対すべての類似度マトリックス
# image=parts2ges_all(pts,hands)

def parts2ges(p,g,seg):
    outarr_p_size=p.shape[0]
    outarr_g_size=g.shape[0]
    points =p.transpose(2, 1).contiguous()
    pred, _, _,all_feat = model_pointnet(points)
    pred = pred.view(-1, 3)

    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().data.numpy()
    # pred_choice 1: 左手, 2: 右手
    
    points=points.transpose(1, 2).cpu().data.numpy()
    pred_np=pred_np.reshape(outarr_p_size,2048,1)
    pl=np.array([])
    pr=np.array([])
    for batch in range(outarr_p_size):
        count=0
        parts_l_list=np.array([])
        parts_r_list=np.array([])
        #print(batch,np.count_nonzero(pred_np[batch]==1),np.count_nonzero(pred_np[batch]==2) )
        target_l=pred_np
        target_r=pred_np
        if np.count_nonzero(pred_np[batch]==2)<=10:    
            target_l=seg
            print("セグメント失敗",batch)
        else:
            target_l=pred_np
        if np.count_nonzero(pred_np[batch]==1)<=10:
            target_r=seg
            print("セグメント失敗",batch)
        else:
            target_r=pred_np

        for j in range(2048):
            if target_l[batch][j]==2:
                parts_l_list=np.append(parts_l_list,points[batch][j])
            if target_r[batch][j]==1:
                parts_r_list=np.append(parts_r_list,points[batch][j])

        while len(parts_l_list)<=(3 * 256):
            count+=1
            add_list=parts_l_list*1.01
            parts_l_list=np.append(parts_l_list,add_list)
        while len(parts_r_list)<=(3 * 256):
            count+=1
            add_list=parts_r_list*1.01
            parts_r_list=np.append(parts_r_list,add_list)

        parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3)             
        choice = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True)                
        pl=np.append(pl,parts_l_list[choice,:])
        parts_r_list=parts_r_list.reshape(int(len(parts_r_list)/3),3)             
        choice = np.random.choice(int(parts_r_list.shape[0]), 256, replace=True)                
        pr=np.append(pr,parts_r_list[choice,:])
        pl = pl - np.expand_dims(np.mean(pl, axis = 0), 0)
        pr = pr - np.expand_dims(np.mean(pr, axis = 0), 0)
    pl=pl.reshape(outarr_p_size,256,3).astype(np.float32)
    pr=pr.reshape(outarr_p_size,256,3).astype(np.float32)

    pl=torch.from_numpy(pl)
    pr=torch.from_numpy(pr)
    pl=pl.transpose(2,1)
    pr=pr.transpose(2,1)

    hand=np.split(g,2,axis=1)
    hand_l=hand[0]
    hand_r=hand[1]
    # 手首を0に正規化
    for k in range(outarr_g_size):
        hand_l[k] = hand_l[k] - hand_l[k][0]
        hand_r[k] = hand_r[k] - hand_r[k][0]

    hand_l=hand_l.reshape(outarr_g_size,69)
    hand_r=hand_r.reshape(outarr_g_size,69)
    sim_pl_gl, _, _, parts_l_feat =model_ges2parts(hand_l, pl, all_feat)
    sim_pr_gr, _, _, parts_r_feat =model_ges2parts(hand_r, pr, all_feat)

    sim_parts_per_g_l = sim_pl_gl.transpose(1,0)
    sim_parts_per_ges_l= sim_parts_per_g_l.detach().numpy()
    print(sim_parts_per_ges_l)
    sim_parts_per_g_r = sim_pr_gr.transpose(1,0)
    sim_parts_per_ges_r= sim_parts_per_g_r.detach().numpy()
    print("===========")

    img_l=np.zeros((sim_parts_per_ges_l.shape[0],sim_parts_per_ges_l.shape[1]))
    img_r=np.zeros((sim_parts_per_ges_r.shape[0],sim_parts_per_ges_r.shape[1]))

    for h in range(sim_parts_per_g_l.shape[0]):
        for w in range(sim_parts_per_g_l.shape[1]):
            img_l[h][w]= sim_parts_per_g_l[h][w]

    for h in range(sim_parts_per_g_r.shape[0]):
        for w in range(sim_parts_per_g_r.shape[1]):
            img_r[h][w]= sim_parts_per_g_r[h][w]

    # パーツ→ポイント類似度
    sim_parts_per_pts, _ = model_parts2shape(parts_l_feat, parts_r_feat, all_feat)
    sim_parts_per_pts=sim_parts_per_pts.detach().numpy()
    img_parts_per_pts=np.zeros((sim_parts_per_pts.shape[0],sim_parts_per_pts.shape[1]))
    for h in range(sim_parts_per_pts.shape[0]):
        for w in range(sim_parts_per_pts.shape[1]):
            img_parts_per_pts[h][w]= sim_parts_per_pts[h][w]
    
    block_size = 1
    if block_size !=1:
        image_blocked_l = np.kron(img_l, np.ones((block_size, block_size)))
        image_l = image_blocked_l
        image_blocked_r = np.kron(img_r, np.ones((block_size, block_size)))
        image_r = image_blocked_r
        img_parts_per_pts = np.kron(img_parts_per_pts, np.ones((block_size, block_size)))

    else:
        image_l, image_r, image_parts_per_pts = img_l, img_r, img_parts_per_pts
        
    return image_l, image_r, image_parts_per_pts

count=0
ziku_label=np.array([])
filename_befor=""
for filename in np.array(label_base):
    
    if filename[:2] != filename_befor:
        ziku_label=np.append(ziku_label,count)
        filename_befor=filename[:2]
    count+=1


ziku_label=ziku_label
ziku_label[0]=0
ziku_label=np.append(ziku_label,len(label_base))
print(ziku_label)

import matplotlib.colors as mcolors
image=parts2ges_all(pts_base,hands_base,seg_base)
cate_color=["white","red","pink","orange",""]
il,ir,i_parts2pts=parts2ges(pts_base,hands_base,seg_base)
plt.rcParams["font.size"]=15

plt.figure(1)
plt.figure(figsize=(8, 8)) 
plt.title('左手：ジェスチャー ← → パーツ形状のコサイン類似度', fontsize=16)
plt.imshow(il, cmap='viridis', aspect='equal',vmax=1,vmin=0)
labels =[int(round(tick)) for tick in ziku_label] 
plt.colorbar(label='コサイン類似度')  # カラーバー表示
plt.xticks(ticks=ziku_label-0.5,labels=labels)
plt.xticks(rotation=90)
plt.yticks(ticks=ziku_label-0.5,labels=labels)

plt.xlabel("ジェスチャー特徴")
plt.ylabel("パーツ特徴")
plt.tick_params(labelsize=8.5)

if opt.save:
    plt.savefig(os.path.join(opt.savedir, 'cosinsim_left_hand.png'), dpi=100, bbox_inches='tight')
else:
    plt.savefig('colormap_image.png', dpi=300)

plt.figure(2)
plt.figure(figsize=(8, 8)) 
plt.title('右手：ジェスチャー ← → パーツのコサイン類似度', fontsize=16)
plt.imshow(ir, cmap='viridis', aspect='equal',vmax=1,vmin=0)
labels =[int(round(tick)) for tick in ziku_label] 
plt.colorbar(label='コサイン類似度')  # カラーバー表示
plt.xticks(ticks=ziku_label-0.5,labels=labels)
plt.xticks(rotation=90)
plt.yticks(ticks=ziku_label-0.5,labels=labels)
plt.xlabel("ジェスチャー特徴")
plt.ylabel("パーツ特徴")
plt.tick_params(labelsize=8.5)

if opt.save:
    plt.savefig(os.path.join(opt.savedir, 'cosinsim_right_hand.png'), dpi=100, bbox_inches='tight')

plt.figure(3)
plt.figure(figsize=(8, 8)) 
plt.title('両手のパーツ ← → ポイントのコサイン類似度', fontsize=16)
plt.imshow(i_parts2pts, cmap='viridis', aspect='equal',vmax=1,vmin=0)
labels =[int(round(tick)) for tick in ziku_label] 
plt.colorbar(label='コサイン類似度')  # カラーバー表示
plt.xticks(ticks=ziku_label-0.5,labels=labels)
plt.xticks(rotation=90)
plt.yticks(ticks=ziku_label-0.5,labels=labels)

plt.xlabel("ジェスチャー特徴")
plt.ylabel("パーツ特徴")

plt.tick_params(labelsize=8.5)

if opt.save:
    plt.savefig(os.path.join(opt.savedir, 'cosinsim_parts_to_points.png'), dpi=100, bbox_inches='tight')
    print(f"All images saved to {opt.savedir}")
else:
    plt.show()



