"""
Neural network models for Parts2Gesture.

This module implements various deep learning architectures:
- STN3d: Spatial Transformer Network for 3D
- STNkd: Spatial Transformer Network for k-dimensional features
- PointNetfeat: Feature extraction from point clouds
- PointNetDenseCls: Dense classification on point clouds
- ContrastiveNet: Gesture-to-parts cross-modal contrastive learning
- PartsToPtsNet: Parts-to-points mapping network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
class STN3d(nn.Module):
    """Spatial Transformer Network for 3D point clouds."""
    
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of 3D spatial transformer.
        
        Args:
            x: Input tensor of shape (batch_size, 3, num_points)
        
        Returns:
            Transformation matrix of shape (batch_size, 3, 3)
        """
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Identity matrix initialization
        iden = torch.eye(3, dtype=torch.float32).view(1, 9).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """Spatial Transformer Network for k-dimensional features."""
    
    def __init__(self, k: int = 64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.k = k
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of k-dimensional spatial transformer.
        
        Args:
            x: Input tensor of shape (batch_size, k, num_points)
        
        Returns:
            Transformation matrix of shape (batch_size, k, k)
        """
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Identity matrix initialization
        iden = torch.eye(self.k, dtype=torch.float32).flatten().view(1, self.k * self.k).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
        
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    """Extract global and local features from point clouds using PointNet."""
    
    def __init__(self, global_feat: bool = True, feature_transform: bool = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x: torch.Tensor) -> Tuple:
        """
        Extract features from point clouds.
        
        Args:
            x: Point cloud tensor (batch_size, 3, num_points)
        
        Returns:
            If global_feat=True: (global_features, transform_3d, transform_feature)
            If global_feat=False: (concatenated_features, global_features, transform_3d, transform_feature)
        """
        n_pts = x.size(2)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        all_feat = x
        
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), all_feat, trans, trans_feat


class PointNetCls(nn.Module):
    """Classification network on global point cloud features."""
    
    def __init__(self, k: int = 2, feature_transform: bool = False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Classify point clouds.
        
        Args:
            x: Point cloud tensor (batch_size, 3, num_points)
        
        Returns:
            Tuple of:
            - logits: Classification logits
            - transform_3d: 3D transformation matrix
            - transform_feature: Feature space transformation (if enabled)
        """
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

#segmentation k=number of segmentation classes
class PointNetDenseCls(nn.Module):
    """Dense classification network for part segmentation."""
    
    def __init__(self, k: int = 2, feature_transform: bool = False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Perform dense classification on point clouds.
        
        Args:
            x: Point cloud tensor (batch_size, 3, num_points)
        
        Returns:
            Tuple of:
            - predictions: (batch_size, num_points, k) segmentation logits
            - transform_3d: 3D transformation matrix
            - transform_feature: Feature space transformation (if enabled)
            - global_features: Global point cloud features
        """
        batch_size = x.size(0)
        n_pts = x.size(2)
        x, all_feat, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batch_size, n_pts, self.k)
        
        return x, trans, trans_feat, all_feat

class SkeltonNet(nn.Module):
    """Extract features from skeleton joint positions."""
    
    def __init__(self):
        super(SkeltonNet, self).__init__()
        self.fc1 = nn.Linear(69, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract skeleton features."""
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        return x

class PartsNet(nn.Module):
    """Extract features from point cloud parts."""
    
    def __init__(self, feature_transform: bool = False):
        super(PartsNet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, parts: torch.Tensor, all_feat: torch.Tensor) -> torch.Tensor:
        """Extract and combine part features with global features."""
        x, trans, trans_feat = self.feat(parts)

        #x:1024,all_feat:1024
        x = torch.cat([x,all_feat],dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x)))

        return x

class ContrastiveNet(nn.Module):
    """Compute contrastive similarity between gestures and point cloud parts."""
    
    def __init__(self):
        super(ContrastiveNet, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.sknet = SkeltonNet()
        self.partsnet = PartsNet()

    def forward(
        self, 
        input_sk: torch.Tensor, 
        input_parts: torch.Tensor, 
        input_all_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute similarity between skeleton and part features."""
        sk_feat = self.sknet(input_sk)
        parts_feat = self.partsnet(input_parts, input_all_feat)

        sk_feat = sk_feat / sk_feat.norm(dim=-1, keepdim=True)
        parts_feat = parts_feat / parts_feat.norm(dim=-1, keepdim=True)

        logit_per_sk = sk_feat @ parts_feat.t()
        logit_per_parts = logit_per_sk.t()
        
        return logit_per_sk, logit_per_parts, sk_feat, parts_feat
    

class BothPartsNet2(nn.Module):
    """Combine left and right hand part features."""
    
    def __init__(self):
        super(BothPartsNet2, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Combine left and right hand features."""
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        
        return x



    
class PtsFeatNet(nn.Module):
    """Extract features from point cloud global features."""
    
    def __init__(self):
        super(PtsFeatNet, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract point cloud features."""
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return x


class PartsToPtsNet(nn.Module):
    """Map between parts and point clouds using contrastive learning."""
    
    def __init__(self):
        super(PartsToPtsNet, self).__init__()
        self.bothpartsnet = BothPartsNet2()
        self.parts2pts = PtsFeatNet()

    def forward(
        self, 
        parts_feat_l: torch.Tensor, 
        parts_feat_r: torch.Tensor, 
        all_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity between combined parts and point clouds."""
        p_feat = self.bothpartsnet(parts_feat_l, parts_feat_r)
        pts_feat = self.parts2pts(all_feat)

        p_feat = p_feat / p_feat.norm(dim=-1, keepdim=True)
        pts_feat = pts_feat / pts_feat.norm(dim=-1, keepdim=True)

        logit_per_p = p_feat @ pts_feat.t()
        logit_per_pts = logit_per_p.t()
        
        return logit_per_p, logit_per_pts


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """Regularization for spatial transformer networks."""
    d = trans.size(1)
    batch_size = trans.size(0)
    I = torch.eye(d, dtype=trans.dtype, device=trans.device).unsqueeze(0)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class PartsToPtsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bothpartsnet=BothPartsNet2()
        self.parts2pts=PtsFeatNet()
    
    def forward(self,parts_feat_l,parts_feat_r,all_feat):
        
        p_feat=self.bothpartsnet(parts_feat_l,parts_feat_r)
        pts_feat=self.parts2pts(all_feat)

        p_feat = p_feat / p_feat.norm(dim=-1, keepdim=True)
        pts_feat = pts_feat / pts_feat.norm(dim=-1, keepdim=True)

        logit_per_p = p_feat @ pts_feat.t()
        logit_per_pts = logit_per_p.t()
        return logit_per_p, logit_per_pts

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss
