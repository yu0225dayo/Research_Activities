"""
Evaluation script for computing mIoU metrics on part-to-gesture mapping.

This module evaluates the performance of the contrastive learning model
for gesture recognition from 3D point cloud parts.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ShapeNetDataset
from model import ContrastiveNet, PartsToPtsNet, PointNetDenseCls
from functions import load_models

# Configuration
CHOICE = "Mug"
NUM_POINTS = 2048
SAMPLES_PER_PART = 256


# Argument parsing
parser = argparse.ArgumentParser(description='mIoU evaluation for part segmentation')
parser.add_argument('--model', type=str, default='model_Contratstive_Parts2Gesture', help='model directory path')
parser.add_argument('--dataset', type=str, default='dataset', help='dataset path')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

opt = parser.parse_args()
print(f"Arguments: {opt}")

# Load datasets
val_dataset = ShapeNetDataset(
    root=opt.dataset,
    split="val",
    data_augmentation=False
)

search_dataset = ShapeNetDataset(
    root=opt.dataset,
    split="search",
    data_augmentation=False
)

# Create data loaders
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
search_loader = DataLoader(
    search_dataset,
    batch_size=len(os.listdir(os.path.join(opt.dataset, "search/pts"))),
    shuffle=False
)

# Load models
print("Loading models...")

pointnet_classifier, sk_parts_classifier, p2pts_classifier = load_models(opt.model)

# Load sample batches
pts, seg, hands, label, batch_w = next(iter(val_loader))
pts_base, seg_base, hands_base, label_base, batch_w_base = next(iter(search_loader))

def parts2ges_all(parts: np.ndarray, ges: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """
    Find the best matching gesture for given point cloud parts across all gestures.
    
    Args:
        parts: Point cloud parts (batch_size, num_points, 3)
        ges: Gesture sequences (batch_size, num_joints, 3)
        seg: Segmentation labels (batch_size, num_points)
    
    Returns:
        Heatmap image of similarity scores
    """
    outarr_p_size = parts.shape[0]
    outarr_g_size = ges.shape[0]
    points =parts.transpose(2, 1).contiguous()
    pred, _, _,all_feat = pointnet_classifier(points)
    pred = pred.view(-1, 3)

    # pred_choice 1 :left,  pred_choice 2 : right
    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().data.numpy()
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
        #augmentation 検出パーツが少ないとき、性能が変わりそう
        # *1.01は少しだけずらす意味
        while len(parts_l_list)<=(3 * 256):
            add_list=parts_l_list*1.01
            parts_l_list=np.append(parts_l_list,add_list)

        parts_l_list=parts_l_list.reshape(int(len(parts_l_list)/3),3)             
        choice = np.random.choice(int(parts_l_list.shape[0]), 256, replace=True)                
        pl=np.append(pl,parts_l_list[choice,:])

        while len(parts_r_list)<=(3 * 256):
            add_list=parts_r_list*1.01
            parts_r_list=np.append(parts_r_list,add_list)
        
        parts_r_list=parts_r_list.reshape(int(len(parts_r_list)/3),3)             
        choice = np.random.choice(int(parts_r_list.shape[0]), 256, replace=True)                
        pr=np.append(pr,parts_r_list[choice,:])
        pl = pl - np.expand_dims(np.mean(pl, axis = 0), 0)
        pr = pr - np.expand_dims(np.mean(pr, axis = 0), 0)
    pl = pl.reshape(outarr_p_size,256,3).astype(np.float32)
    pr = pr.reshape(outarr_p_size,256,3).astype(np.float32)

    pl = torch.from_numpy(pl)
    pr = torch.from_numpy(pr)
    pl = pl.transpose(2,1)
    pr = pr.transpose(2,1)
    
    hand = np.split(ges,2,axis=1)
    hand_l = hand[0]
    hand_r = hand[1]
    #手首を0に
    for k in range(outarr_g_size):
        hand_l[k] = hand_l[k] - hand_l[k][0]
        hand_r[k] = hand_r[k] - hand_r[k][0]
    hand_l=hand_l.reshape(outarr_g_size,69)
    hand_r=hand_r.reshape(outarr_g_size,69)

    #入力とのコサイン類似度を獲得
    sim_pl_gl, _, _, parts_l_feat =sk_parts_classifier(hand_l, pl, all_feat)
    sim_pr_gr, _, _, parts_r_feat =sk_parts_classifier(hand_r, pr, all_feat)
    sim_pl_gr, _, _, parts_l_feat =sk_parts_classifier(hand_l, pr, all_feat)
    sim_pr_gl, _, _, parts_r_feat =sk_parts_classifier(hand_r, pl, all_feat)

    sim_pl_per_g = torch.cat((sim_pl_gl,sim_pl_gr),dim=1)
    sim_pr_per_g = torch.cat((sim_pr_gl,sim_pr_gr),dim=1)
    
    sim_parts_per_ges = torch.cat((sim_pl_per_g, sim_pr_per_g),dim=0).transpose(1,0)
    sim_parts_per_ges = sim_parts_per_ges.detach().numpy()
    
    mat = sim_parts_per_ges.shape[1]//sim_parts_per_ges.shape[0]

    if mat == 0:
        mat = 1

    #heatmap作成
    img = np.zeros((mat*sim_parts_per_ges.shape[0], sim_parts_per_ges.shape[1]))

    for k in range(mat):
        for h in range(sim_parts_per_ges.shape[0]):
            for w in range(sim_parts_per_ges.shape[1]):
                img[sim_parts_per_ges.shape[0]*k+h][w]= sim_parts_per_ges[h][w]

    block_size = 3
    image_blocked = np.kron(img, np.ones((block_size, block_size)))
    image = image_blocked

    return image


def parts2ges(
    parts: np.ndarray,
    gestures: np.ndarray,
    seg: np.ndarray,
    filenames: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute matching accuracy between parts and gestures.
    
    Args:
        parts: Point cloud parts (batch_size, num_points, 3)
        gestures: Gesture sequences (batch_size, num_joints, 3)
        seg: Segmentation labels (batch_size, num_points)
        filenames: List of filenames for class classification
    
    Returns:
        Tuple of accuracy arrays:
            - same_arr_l: Left hand matching accuracy
            - same_arr_r: Right hand matching accuracy
            - same_arr_class_l: Left hand class accuracy
            - same_arr_class_r: Right hand class accuracy
            - same_p2p: Part-to-points matching accuracy
            - same_class_p2p: Part-to-points class accuracy
    """
    batchsize = parts.shape[0]
    outarr_p_size = parts.shape[0]
    outarr_g_size = gestures.shape[0]
    points = parts.transpose(2, 1).contiguous()
    pred, _, _, all_feat = pointnet_classifier(points)
    pred = pred.view(-1, 3)

    pred_choice = pred.data.max(1)[1]
    pred_np = pred_choice.cpu().data.numpy()
    # Class mapping: 0=background, 1=left hand, 2=right hand
    
    points = points.transpose(1, 2).cpu().data.numpy()
    pred_np = pred_np.reshape(outarr_p_size, NUM_POINTS, 1)
    pl = np.array([])
    pr = np.array([])
    
    for batch in range(outarr_p_size):
        parts_l_list = np.array([])
        parts_r_list = np.array([])
        
        target_l = pred_np
        target_r = pred_np
        
        # Fallback to ground truth if segmentation fails
        if np.count_nonzero(pred_np[batch] == 2) <= 10:
            target_l = seg
            print(f"Warning: Segmentation failed for batch {batch}")
        else:
            target_l = pred_np
            
        if np.count_nonzero(pred_np[batch] == 1) <= 10:
            target_r = seg
            print(f"Warning: Segmentation failed for batch {batch}")
        else:
            target_r = pred_np

        # Extract left and right hand points
        for j in range(NUM_POINTS):
            if target_l[batch][j] == 2:
                parts_l_list = np.append(parts_l_list, points[batch][j])
            if target_r[batch][j] == 1:
                parts_r_list = np.append(parts_r_list, points[batch][j])
        
        # Data augmentation for insufficient points
        while len(parts_l_list) <= (3 * SAMPLES_PER_PART):
            add_list = parts_l_list * 1.01  # Small perturbation
            parts_l_list = np.append(parts_l_list, add_list)

        parts_l_list = parts_l_list.reshape(int(len(parts_l_list) / 3), 3)
        choice = np.random.choice(int(parts_l_list.shape[0]), SAMPLES_PER_PART, replace=True)
        pl = np.append(pl, parts_l_list[choice, :])

        while len(parts_r_list) <= (3 * SAMPLES_PER_PART):
            add_list = parts_r_list * 1.01
            parts_r_list = np.append(parts_r_list, add_list)
        
        parts_r_list = parts_r_list.reshape(int(len(parts_r_list) / 3), 3)
        choice = np.random.choice(int(parts_r_list.shape[0]), SAMPLES_PER_PART, replace=True)
        pr = np.append(pr, parts_r_list[choice, :])
        
        # Normalize
        pl = pl - np.expand_dims(np.mean(pl, axis=0), 0)
        pr = pr - np.expand_dims(np.mean(pr, axis=0), 0)
    
    pl = pl.reshape(outarr_p_size, SAMPLES_PER_PART, 3).astype(np.float32)
    pr = pr.reshape(outarr_p_size, SAMPLES_PER_PART, 3).astype(np.float32)

    pl = torch.from_numpy(pl)
    pr = torch.from_numpy(pr)
    pl = pl.transpose(2, 1)
    pr = pr.transpose(2, 1)

    # Split gestures into left and right hands
    hand = np.split(gestures, 2, axis=1)
    hand_l = hand[0]
    hand_r = hand[1]
    
    # Normalize hand poses (set wrist as origin)
    for k in range(outarr_g_size):
        hand_l[k] = hand_l[k] - hand_l[k][0]
        hand_r[k] = hand_r[k] - hand_r[k][0]

    hand_l = hand_l.reshape(outarr_g_size, 69)
    hand_r = hand_r.reshape(outarr_g_size, 69)
    
    # Compute similarity scores
    sim_pl_gl, _, _, parts_l_feat = sk_parts_classifier(hand_l, pl, all_feat)
    sim_pr_gr, _, _, parts_r_feat = sk_parts_classifier(hand_r, pr, all_feat)

    ans = torch.eye(batchsize, batchsize)
    pl_gl = (torch.argmax(ans, dim=1).eq(torch.argmax(sim_pl_gl, dim=1)).sum()) / (batchsize)
    pr_gr = (torch.argmax(ans, dim=1).eq(torch.argmax(sim_pr_gr, dim=1)).sum()) / (batchsize)

    sim_plgl = torch.argmax(sim_pl_gl, dim=1)
    sim_prgr = torch.argmax(sim_pr_gr, dim=1)
    same_arr_r = np.array([])
    same_arr_l = np.array([])

    for i in range(batchsize):
        if sim_plgl[i] == i:
            same_arr_l = np.append(same_arr_l, 1)
        if sim_prgr[i] == i:
            same_arr_r = np.append(same_arr_r, 1)
        if sim_plgl[i] != i:
            same_arr_l = np.append(same_arr_l, 0)
        if sim_prgr[i] != i:
            same_arr_r = np.append(same_arr_r, 0)

    same_arr_class_l = np.array([])
    same_arr_class_r = np.array([])

    for j in range(batchsize):
        if filenames[j][:2] == filenames[int(torch.argmax(sim_pl_gl, dim=1)[j])][:2]:
            same_arr_class_l = np.append(same_arr_class_l, 1)
        if filenames[j][:2] == filenames[int(torch.argmax(sim_pr_gr, dim=1)[j])][:2]:
            same_arr_class_r = np.append(same_arr_class_r, 1)
        if filenames[j][:2] != filenames[int(torch.argmax(sim_pl_gl, dim=1)[j])][:2]:
            same_arr_class_l = np.append(same_arr_class_l, 0)
        if filenames[j][:2] != filenames[int(torch.argmax(sim_pr_gr, dim=1)[j])][:2]:
            same_arr_class_r = np.append(same_arr_class_r, 0)
    
    # Parts-to-points mapping
    sim_parts_per_pts, _ = p2pts_classifier(parts_l_feat, parts_r_feat, all_feat)
    same_p2p = np.array([])
    sim_p2p = torch.argmax(sim_parts_per_pts, dim=1)

    for k in range(batchsize):
        if sim_p2p[k] == k:
            same_p2p = np.append(same_p2p, 1)
        if sim_p2p[k] != k:
            same_p2p = np.append(same_p2p, 0)
    
    same_class_p2p = np.array([])

    for l in range(batchsize):
        if filenames[l][:2] == filenames[int(torch.argmax(sim_parts_per_pts, dim=1)[l])][:2]:
            same_class_p2p = np.append(same_class_p2p, 1)
        if filenames[l][:2] != filenames[int(torch.argmax(sim_parts_per_pts, dim=1)[l])][:2]:
            same_class_p2p = np.append(same_class_p2p, 0)

    return same_arr_l, same_arr_r, same_arr_class_l, same_arr_class_r, same_p2p, same_class_p2p


def calc_mIoU(results: np.ndarray, metric_name: str) -> None:
    """
    Calculate and print mean Intersection over Union (mIoU) per class.
    
    Args:
        results: Array with [accuracy_score, class_label] for each sample
        metric_name: Name of the metric for display
    """
    classes = np.unique(results[:, 1])
    
    print(f"\n{metric_name} Results:")
    print("-" * 50)
    
    for cls in classes:
        # Extract values for this class
        values = results[results[:, 1] == cls, 0].astype(np.float32)
        
        # Compute statistics
        mean_value = np.mean(values)
        num_samples = len(values)
        
        print(f"  Class {cls}: mean={mean_value:.4f}, samples={num_samples}")


# Main evaluation
if __name__ == "__main__":
    print("\nEvaluating gesture recognition performance...")
    print("=" * 50)
    
    # Extract class labels
    arr_name = np.array([])
    for csv in label_base:
        arr_name = np.append(arr_name, csv[:2])
    
    # Compute results
    (arr_l, arr_r, s_arr_l, s_arr_r, p2p, s_p2p) = parts2ges(
        pts_base, hands_base, seg_base, label_base
    )

    # Stack results with labels
    arr_out_l = np.stack((arr_l, arr_name), 1)
    arr_out_r = np.stack((arr_r, arr_name), 1)

    sarr_out_l = np.stack((s_arr_l, arr_name), 1)
    sarr_out_r = np.stack((s_arr_r, arr_name), 1)

    arr_out_p2p = np.stack((p2p, arr_name), 1)
    sarr_out_p2p = np.stack((s_p2p, arr_name), 1)

    # Print results
    calc_mIoU(arr_out_l, "Left Hand Direct Matching (sim_pl_gl)")
    calc_mIoU(arr_out_r, "Right Hand Direct Matching (sim_pr_gr)")
    calc_mIoU(sarr_out_l, "Left Hand Class Matching (sim_class_pl_gl)")
    calc_mIoU(sarr_out_r, "Right Hand Class Matching (sim_class_pr_gr)")
    calc_mIoU(arr_out_p2p, "Part-to-Points Matching (sim_p2p)")
    calc_mIoU(sarr_out_p2p, "Part-to-Points Class Matching (sim_class_p2p)")
    
    print("\n" + "=" * 50)
    print("Evaluation complete.")



