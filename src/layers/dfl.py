import torch.nn as nn
import torch



class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, reg_max = 16):
        super().__init__()
        self.conv   = nn.Conv2d(reg_max, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(reg_max, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, reg_max, 1, 1))
        self.reg_max     = reg_max

    def forward(self, x):
        b, c, a = x.shape
        x_hat = x.view(b, 4, self.reg_max, a).transpose(2, 1).softmax(1)
        return self.conv(x_hat).view(b, 4, a)