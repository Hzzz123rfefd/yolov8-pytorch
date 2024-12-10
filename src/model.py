import torch.nn as nn
import torch
from tqdm import tqdm
import os
from torch import optim
from torch.utils.data import DataLoader

from src.modules import *
from src.loss import *
from src.utils import *

class ObjectDetect(nn.Module):
    def __init__(
            self,
            model_size = "n",
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
            device: str = "cuda",
    ):
        super().__init__()
        self.model_size = model_size
        self.reg_max = reg_max
        self.class_num = class_num
        self.no = self.reg_max * 4 + self.class_num
        self.clou_weight = clou_weight
        self.dfl_weight = dfl_weight
        self.cls_weight = cls_weight
        self.use_dfl = use_dfl
        self.stride = [8, 16, 32]
        self.proj = torch.arange(self.reg_max, dtype = torch.float, device = device)
        self.device = device if torch.cuda.is_available() else "cpu"
        if model_size == "n":
            self.d = 0.33
            self.w = 0.25
            self.r = 2
        elif model_size == "s":
            self.d = 0.33
            self.w = 0.5
            self.r = 2
            
        self.backbone  = Backbone(base_channels, self.w, self.d, self.r, d_rate = 3).to(self.device)
        self.neck = Neck(base_channels,  self.w, self.d, self.r, d_rate = 3).to(self.device)
        self.head = Head(class_num, base_channels, self.w, self.d, self.r,reg_max = 16).to(self.device)
        
        """ loss """
        self.assigner = TaskAlignedAssigner(
            topk = top_k, 
            num_classes = class_num, 
            alpha = alpha, 
            beta = beta
        )
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
    
    def forward(self,input):
        x = input["image"].to(self.device)
        P3, P4, P5 = self.backbone(x)
        T1, T2, T3 = self.neck(P3, P4, P5)
        dbox, cls, x = self.head(T1, T2, T3)
        output = {
            "predict":x,
            "gt_bboxes":input["gt_bboxes"].to(self.device),
            "gt_labels":input["gt_labels"].to(self.device),
            "gt_mask":input["gt_mask"].to(self.device)
        }
        return output
        
    def compute_loss(self, input):
        # process data
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = input["predict"]  # [(b, 144, 80, 80), (b, 144, 40, 40), (b, 144, 20, 20)]
        """
            pred_scores: (b, 80, 8400)     # 用于计算分类损失
            pred_distri: (b, 64, 8400)    # 用于计算分类损失
        """
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.class_num), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
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
        mask_gt = input["gt_mask"]           # (16, 28, 1)
        
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
        bce = nn.BCEWithLogitsLoss(reduction="none")
        loss[1] = bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

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
            "total_loss": loss.sum() * batch_size,
            "cls_loss": loss[1],
            "ciou_loss": loss[0],
            "dfl_loss": loss[2]
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
    
    def trainning(
        self,
        train_dataloader:DataLoader = None,
        test_dataloader:DataLoader = None,
        optimizer_name:str = "Adam",
        weight_decay:float = 1e-4,
        clip_max_norm:float = 0.5,
        factor:float = 0.3,
        patience:int = 15,
        lr:float = 1e-4,
        total_epoch:int = 1000,
        save_checkpoint_step:str = 10,
        save_model_dir:str = "models"
    ):
        ## 1 trainning log path 
        first_trainning = True
        check_point_path = save_model_dir  + "/checkpoint.pth"
        log_path = save_model_dir + "/train.log"

        ## 2 get net pretrain parameters if need 
        """
            If there is  training history record, load pretrain parameters
        """
        if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
            self.load_pretrained(save_model_dir)  
            first_trainning = False

        else:
            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)
            with open(log_path, "w") as file:
                pass


        ##  3 get optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer, 
            mode = "min", 
            factor = factor, 
            patience = patience
        )

        ## init trainng log
        if first_trainning:
            best_loss = float("inf")
            last_epoch = 0
        else:
            checkpoint = torch.load(check_point_path, map_location=self.device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint["loss"]
            last_epoch = checkpoint["epoch"] + 1

        try:
            for epoch in range(last_epoch,total_epoch):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                loss = train_loss + test_loss
                lr_scheduler.step(loss)
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                check_point_path = save_model_dir  + "/checkpoint.pth"
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": None,
                        "lr_scheduler": None
                    },
                    check_point_path
                )

                if epoch % save_checkpoint_step == 0:
                    os.makedirs(save_model_dir + "/" + "chaeckpoint-"+str(epoch))
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        save_model_dir + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
                    )
                if is_best:
                    self.save_pretrained(save_model_dir)

        # interrupt trianning
        except KeyboardInterrupt:
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict()
                    },
                    check_point_path
                )

    def train_one_epoch(self, epoch,train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train()
        self.to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        cls_loss =  AverageMeter()
        ciou_loss = AverageMeter()
        dfl_loss = AverageMeter()
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device == "cpu" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())
            cls_loss.update(out_criterion["cls_loss"].item())
            ciou_loss.update(out_criterion["ciou_loss"].item())
            dfl_loss.update(out_criterion["dfl_loss"].item())

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = 0 if self.device == "cpu" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "total_loss: {:.4f},cls_loss:{:.4f},ciou_loss:{:.4f},dfl_loss:{:.4f},use_memory: {:.1f}G".format(
                total_loss.avg, 
                cls_loss.avg,
                ciou_loss.avg,
                dfl_loss.avg,
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg

    def test_epoch(self,epoch, test_dataloader,trainning_log_path = None):
        total_loss = AverageMeter()
        cls_loss =  AverageMeter()
        ciou_loss = AverageMeter()
        dfl_loss = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"].item())
                cls_loss.update(out_criterion["cls_loss"].item())
                ciou_loss.update(out_criterion["ciou_loss"].item())
                dfl_loss.update(out_criterion["dfl_loss"].item())

            postfix_str = "total_loss: {:.4f},cls_loss:{:.4f},ciou_loss:{:.4f},dfl_loss:{:.4f}".format(
                total_loss.avg, 
                cls_loss.avg,
                ciou_loss.avg,
                dfl_loss.avg,
            )
        print(postfix_str)
        with open(trainning_log_path, "a") as file:
            file.write(postfix_str + "\n")
        return total_loss.avg

    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self,  save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")
        
        
        
        
        
