from src.model import ObjectDetect

import torch

x = torch.rand(2,3,640,640)
model = ObjectDetect()
y = model(x)