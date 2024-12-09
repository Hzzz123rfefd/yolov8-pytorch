import math
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from torch import optim
import torch
from transformers import AutoModel,AutoTokenizer
from torch.utils.data import DataLoader

from src.utils import *



class ObjectDetect(nn.Module):
    def __init__(
            self,
            reg_max: int = 16,
            class_num: int = 80,
            use_dfl: float = True,
            device: str = "cpu",
    ):
        self.rea_max = reg_max
        self.class_num = class_num
        self.use_dfl = use_dfl
        self.device = device if torch.cuda.is_available() else "cpu"
        
    def compute_loss(self, input):
        # process data
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = input["preds"]  # [(b, 144, 80, 80), (b, 144, 40, 40), (b, 144, 20, 20)]
        """
            pred_scores: (b, 80, 8400)     # 用于计算分类损失
            pred_distri: (b, 64, 8400)    # 用于计算分类损失
        """
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        
        # generator anchors
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        """ 
            anchor_points:(8400, 2)   xy坐标
            stride_tensor:(8400, 1)  每个anchor的步长
        """
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # get true label
        gt_labels = input["gt_labels"]         # (16, 28, 1)
        gt_bboxes = input["gt_bboxes"]     # (16, 28, 4)
        mask_gt = input["mask_gt"]           # (16, 28, 1)
        
        # get predict bboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri) 
        
        # Positive sample matching
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
        
        
        
        
        
