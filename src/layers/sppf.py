import torch
import torch.nn as nn
from .conv import *

class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        c_          = in_ch // 2
        self.cv1    = Conv(in_ch, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, out_ch, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))