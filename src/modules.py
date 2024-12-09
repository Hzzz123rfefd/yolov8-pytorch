import torch
import torch.nn as nn
from src.layers import *

"""
    init:
        base_channels: first channel of conv output
        base_depth:
        deep_mul:


    input:
        x(B,3,H,W): images

    output:
        feat1(B,base_channels * 4 ,H/8,W/8): feather one 
        feat2(B,base_channels * 8 ,H/16,W/16): feather two 
        feat3(B,base_channels * 16 *  deep_mul,H/32,W/32): feather three 
"""

class Backbone(nn.Module):
    def __init__(self, b = 64, w = 1, d = 1, r = 1, d_rate = 3):
        super().__init__()

        self.stem_layer = Conv(in_ch = 3, 
                               out_ch = (int)(b * w), 
                               k = 3, 
                               s = 2)
        
        self.stage_layer1 = nn.Sequential(
            Conv(in_ch = (int)(b * w), 
                 out_ch = (int)(b * 2 * w), 
                 k = 3, 
                 s = 2),
            C2f(in_ch = (int)(b * 2 * w), 
                out_ch = (int)(b * 2 * w), 
                n = (int)(d_rate * d), 
                shortcut = True)
        )

        self.stage_layer2 = nn.Sequential(
            Conv(in_ch = (int)(b * 2 * w), 
                 out_ch = (int)(b * 4 * w), 
                 k = 3, 
                 s = 2),
            C2f(in_ch = (int)(b * 4 * w), 
                out_ch = (int)(b * 4 * w), 
                n = (int)(2 * d_rate * d), 
                shortcut = True)
        )

        self.stage_layer3 = nn.Sequential(
            Conv(in_ch = (int)(b * 4 * w) , 
                 out_ch = (int)(b * 8 * w), 
                 k = 3, 
                 s = 2),
            C2f(in_ch = (int)(b * 8 * w), 
                out_ch = (int)(b * 8 * w), 
                n = (int)(2 * d_rate * d), 
                shortcut = True)
        )

        self.stage_layer4 = nn.Sequential(
            Conv(in_ch = (int)(b * 8 * w) , 
                 out_ch = (int)(b * 8 * w * r), 
                 k = 3, 
                 s = 2),
            C2f(in_ch = (int)(b * 8 * w * r), 
                out_ch = (int)(b * 8 * w * r), 
                n = (int)(1 * d_rate * d), 
                shortcut = True),
            SPPF(in_ch = (int)(b * 8 * w * r), 
                 out_ch = (int)(b * 8 * w * r),
                 k=5)
        )

    def forward(self, x):
        x = self.stem_layer(x)
        x = self.stage_layer1(x)

        x = self.stage_layer2(x)
        P3 = x

        x = self.stage_layer3(x)
        P4 = x

        x = self.stage_layer4(x)
        P5 = x
        return P3, P4, P5
    

class Head(nn.Module):
    def __init__(self, class_num, b = 64, w = 1, d = 1, r = 1,reg_max = 16):
        super().__init__()
        self.reg_max = reg_max 
        self.class_num = class_num
        # number of object : number class + 4  * reg_max
        self.no = class_num + 4  * reg_max
        self.box_branch1 = nn.Sequential(Conv(in_ch = (int)(b * 4 * w),
                                              out_ch = (int)(b * 4 * w),
                                              k = 3,
                                              s = 1), 
                                        nn.Conv2d(in_channels= (int)(b * 4 * w),
                                                  out_channels= (int)(4 * self.reg_max), 
                                                  kernel_size = 1,
                                                  stride = 1,
                                                  padding = 0)
                                        )
        self.cls_branch1 = nn.Sequential(Conv(in_ch = (int)(b * 4 * w),
                                        out_ch = (int)(b * 4 * w),
                                        k = 3,
                                        s = 1), 
                                        nn.Conv2d(in_channels= (int)(b * 4 * w),
                                                out_channels= class_num, 
                                                kernel_size = 1,
                                                stride = 1,
                                                padding = 0)
                                        )

        self.box_branch2 = nn.Sequential(Conv(in_ch = (int)(b * 8 * w),
                                              out_ch = (int)(b * 8 * w),
                                              k = 3,
                                              s = 1), 
                                        nn.Conv2d(in_channels= (int)(b * 8 * w),
                                                  out_channels= (int)(4 * self.reg_max), 
                                                  kernel_size = 1,
                                                  stride = 1,
                                                  padding = 0)
                                        )
        
        self.cls_branch2 = nn.Sequential(Conv(in_ch = (int)(b * 8 * w),
                                        out_ch = (int)(b * 8 * w),
                                        k = 3,
                                        s = 1), 
                                        nn.Conv2d(in_channels= (int)(b * 8 * w),
                                                    out_channels = class_num, 
                                                    kernel_size = 1,
                                                    stride = 1,
                                                    padding = 0)
                                        )
        
        self.box_branch3 = nn.Sequential(Conv(in_ch = (int)(b * 8 * w * r),
                                              out_ch = (int)(b * 8 * w * r),
                                              k = 3,
                                              s = 1), 
                                        nn.Conv2d(in_channels= (int)(b * 8 * w * r),
                                                  out_channels= (int)(4 * self.reg_max), 
                                                  kernel_size = 1,
                                                  stride = 1,
                                                  padding = 0)
                                        )

        self.cls_branch3 = nn.Sequential(Conv(in_ch = (int)(b * 8 * w * r),
                                            out_ch = (int)(b * 8 * w * r),
                                            k = 3,
                                            s = 1), 
                                        nn.Conv2d(in_channels= (int)(b * 8 * w * r),
                                                    out_channels = class_num, 
                                                    kernel_size = 1,
                                                    stride = 1,
                                                    padding = 0)
                                        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    def forward(self,T1,T2,T3):
        batch_num = T1.shape[0]

        bbox1          = self.box_branch1(T1)     
        cls1           = self.cls_branch1(T1)
        bbox2          = self.box_branch2(T2)
        cls2           = self.cls_branch2(T2)
        bbox3          = self.box_branch3(T3)
        cls3           = self.cls_branch3(T3)
        x              = [torch.cat([bbox1,cls1],1), torch.cat([bbox2,cls2],1), torch.cat([bbox3,cls3],1)]

        box, cls       = torch.cat([xi.view(batch_num, self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.class_num), 1)

        dbox           = self.dfl(box)

        return dbox, cls, x
    

class Neck(nn.Module):
    def __init__(self,b = 64, w = 1, d = 1, r = 1, d_rate = 3):
        super().__init__()

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_layer1 = C2f((int)(b * 12 * w), 
                                    (int)(b * 4 * w), 
                                    (int)(d_rate * d), 
                                    False)
        self.top_down_layer2 = C2f((int)(b * 8 * w * (1 + r)), 
                                   (int)(b * 8 * w), 
                                   (int)(d_rate * d), 
                                   False)
        self.bottom_up_layer0 = C2f((int)(b * 12 * w), 
                                    (int)(b * 8 * w), 
                                    (int)(d_rate * d), 
                                    False)
        self.bottom_up_layer1 = C2f((int)(b * 8 * w * (1 + r)), 
                                    (int)(b * 8 * w * r), 
                                    (int)(d_rate * d), 
                                    False)
        

        self.down_sample0 = Conv(in_ch = (int)(b * 4 * w), 
                                   out_ch = (int)(b * 4 * w), 
                                    k = 3, 
                                    s = 2)
        self.down_sample1 = Conv(in_ch = (int)(b * 8 * w), 
                            out_ch = (int)(b * 8 * w), 
                            k = 3, 
                            s = 2)




    def forward(self,P3,P4,P5):
        P5_upsample = self.upsample(P5)                   
        F1          = self.top_down_layer2(torch.cat([P5_upsample, P4], 1))
        F1_upsample = self.upsample(F1)        
        T1          = self.top_down_layer1(torch.cat([F1_upsample, P3], 1))

        t1_cov      = self.down_sample0(T1)
        F2          = torch.cat([t1_cov, F1], 1)               
        T2          = self.bottom_up_layer0(F2) 

        t2_cov      = self.down_sample1(T2)
        F3          = torch.cat([t2_cov, P5],1)               
        T3          = self.bottom_up_layer1(F3)                     
        return T1,T2,T3 

