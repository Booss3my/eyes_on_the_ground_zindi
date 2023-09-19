import timm
import torch
from config import *
from torch import nn

# Import the required library for TPU
import torch_xla
import torch_xla.core.xla_model as xm

class Model(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base = timm.create_model(model_name, pretrained=True)
        self.base_cfg = self.base.default_cfg
        self.fc = nn.Linear(self.base_cfg["num_classes"], 1, bias=True)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.base(x)
        return self.fc(x)

model = Model(model_name)
input_size = model.base_cfg["input_size"][1]

# Get the TPU device
device = xm.xla_device()

base_model = model
base_model.to(device)