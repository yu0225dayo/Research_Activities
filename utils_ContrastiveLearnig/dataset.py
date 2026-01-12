"""
Dataset loaders for Parts2Gesture.

Provides dataset classes for loading point clouds, segmentation labels,
and hand pose data for gesture recognition tasks.
"""

from __future__ import print_function
from typing import Tuple, Optional
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm
from plyfile import PlyData


class ShapeNetDataset(data.Dataset):
    """Load ShapeNet point clouds with hand gesture annotations."""
    
    def __init__(
        self,
        root: str,
        npoints: int = 2048,
        classification: bool = False,
        class_choice: Optional[str] = None,
        split: str = 'train',
        data_augmentation: bool = True
    ):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.name = class_choice
        self.classes = {self.name: 20}
        self.seg_classes = self.name
        self.num_seg_classes = 3
        self.datapath = []

        datafol = os.path.join(self.root, split)
        namefol = os.path.join(datafol, "pts")
        
        print(f"Loading dataset from: {datafol}")
        
        for csv in os.listdir(namefol):
            pts = os.path.join(datafol, "pts", csv)
            csvname, ext = os.path.splitext(csv)
            label = os.path.join(datafol, "label", csvname + "_label" + ext)
            hand = os.path.join(datafol, "hands", csvname + "_hand" + ext)
            self.datapath.append([self.name, pts, label, hand, csvname])
        
        print(f"Loaded {len(self.datapath)} samples")

    def _normalize_hand(self, hand_poses: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Normalize hand poses by scaling to middle finger length.
        
        Args:
            hand_poses: Hand joint coordinates (23, 3)
        
        Returns:
            Normalized hand poses and scale factor
        """
        # Calculate middle finger length (joints 0->8->9->10->20)
        middle_len = sum(np.array([
            np.linalg.norm(hand_poses[0] - hand_poses[8]),
            np.linalg.norm(hand_poses[8] - hand_poses[9]),
            np.linalg.norm(hand_poses[9] - hand_poses[10]),
            np.linalg.norm(hand_poses[10] - hand_poses[20])
        ]))
        
        hand_scale = 1.0 / middle_len if middle_len > 0 else 1.0
        return hand_poses * hand_scale / 2, hand_scale

    def __getitem__(self, index: int) -> Tuple:
        """
        Load and preprocess a single sample.
        
        Args:
            index: Sample index
        
        Returns:
            Tuple of (point_set, segmentation, hand_poses, label, batch_weight)
        """
        fn = self.datapath[index]
        point_set = np.array(pd.read_csv(fn[1], header=None)).astype(np.float32)
        cls = self.classes[self.datapath[index][0]]
        hand_set = np.array(pd.read_csv(fn[3], header=None)).astype(np.float32)
        label = fn[4]

        # Split into left and right hand
        hand = np.split(hand_set, 2, axis=0)
        hand_l, hand_scale_l = self._normalize_hand(hand[0])
        hand_r, hand_scale_r = self._normalize_hand(hand[1])
        hand_set = np.vstack((hand_l, hand_r))

        # Load segmentation labels
        seg = pd.read_csv(fn[2], header=None).to_numpy().astype(np.int64)
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]

        # Normalize point cloud
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / (dist + 1e-8)
        
        hand_set_rote = hand_set.copy()

        # Data augmentation
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            point_set[:, [0, 1]] = point_set[:, [0, 1]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)
            hand_set_rote[:, [0, 1]] = hand_set_rote[:, [0, 1]].dot(rotation_matrix)

        seg = seg[choice]
        batch_weight = np.array([
            np.count_nonzero(seg == 0),
            np.count_nonzero(seg == 1),
            np.count_nonzero(seg == 2)
        ]) / len(seg)

        point_set = torch.from_numpy(point_set.astype(np.float32))
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        hand_set_rote = torch.from_numpy(hand_set_rote.astype(np.float32))
        batch_weight = torch.from_numpy(batch_weight.astype(np.float32))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg, hand_set_rote, label, batch_weight

    def __len__(self) -> int:
        return len(self.datapath)
    

class ModelNetDataset(data.Dataset):
    """Load ModelNet point cloud classification dataset."""
    
    def __init__(
        self,
        root: str,
        npoints: int = 2048,
        split: str = 'train',
        data_augmentation: bool = True
    ):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        
        with open(os.path.join(root, f'{self.split}.txt'), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                              '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        self.classes = list(self.cat.keys())
        print(f"Loaded {len(self.fns)} ModelNet samples")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a point cloud and its class label."""
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        
        pts = np.vstack([
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        ]).T
        
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        # Normalize point cloud
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / (dist + 1e-8)

        # Data augmentation
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        
        return point_set, cls

    def __len__(self) -> int:
        return len(self.fns)



if __name__ == '__main__':
    """Test dataset loading."""
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <dataset_type> <data_path>")
        sys.exit(1)
    
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root=datapath, class_choice='Chair')
        print(f"Dataset size: {len(d)}")
        ps, seg, hands, label, batch_w = d[0]
        print(f"Point set: {ps.size()}, {ps.dtype}")
        print(f"Segmentation: {seg.size()}, {seg.dtype}")
        print(f"Hands: {hands.size()}, {hands.dtype}")



