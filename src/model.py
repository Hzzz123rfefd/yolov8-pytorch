import torch.nn as nn
import torch

from src.modules import *
from src.utils import *



class ObjectDetect(nn.Module):
    def __init__(
            self,
            d: float = 0.33,
            w: float = 0.25,
            r: float = 2,
            base_channels: int = 64,
            reg_max: int = 16,
            class_num: int = 80,
            top_k: int = 10,
            alpha: float = 0.5,
            beta: float = 6,
            cls_weight: float = 0.5,
            clou_weight: float = 7.5,
            dfl_weight:float = 1.5,
            use_dfl: float = True,
            
            device: str = "cpu",
    ):
        super().__init__()
        self.rea_max = reg_max
        self.class_num = class_num
        self.clou_weight = clou_weight
        self.dfl_weight = dfl_weight
        self.cls_weight = cls_weight
        self.use_dfl = use_dfl
        self.device = device if torch.cuda.is_available() else "cpu"
    
        self.backbone  = Backbone(base_channels, w, d, r, d_rate = 3)
        self.neck = Neck(base_channels, w, d, r, d_rate = 3)
        self.head = Head(class_num, base_channels, w, d, r,reg_max = 16)
        
        """ loss """
        self.assigner = TaskAlignedAssigner(
            topk = top_k, 
            num_classes = class_num, 
            alpha = alpha, 
            beta = beta
        )
    
    def forward(self,input):
        x = input["image"]
        P3, P4, P5 = self.backbone(x)
        T1, T2, T3 = self.neck(P3, P4, P5)
        dbox, cls, x = self.head(T1, T2, T3)
        output = {
            "predict":x
        }
        return output
        
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
        
        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.clou_weight  # box gain
        loss[1] *= self.cls_weight  # cls gain
        loss[2] *= self.dfl_weight  # dfl gain
        output = {
            "total_loss": loss.sum() * batch_size
        }
        return output
        
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
        
        
        
        
        
