import timm
import torch
from config import *
from torch import nn


class Model(nn.Module):
    def __init__(self,model_name):
        super().__init__()
        self.base = timm.create_model(model_name, pretrained=True)
        self.base_cfg = model.default_cfg
        self.fc = nn.Linear(self.base_cfg["num_classes"],1 ,bias=True)

    def forward(self, x):
        x = self.base(x)
        return nn.Softmax(self.fc(x))

model = Model(model_name)
input_size = model.base_cfg["input_size"][1]
base_model = torch.nn.DataParallel(model, device_ids=[0, 1])
base_model.to(DEVICE)