import timm
import torch
from config import *
from torch import nn


class Model(nn.Module):
    def __init__(self,model_name):
        super().__init__()
        self.base = timm.create_model(model_name, pretrained=True)
        self.base_cfg = self.base.default_cfg
        self.fc = nn.Linear(self.base_cfg["num_classes"],1 ,bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.base(x)
        return self.sigmoid(self.fc(x))


def init_model():
    model = Model(model_name)
    input_size = model.base_cfg["input_size"][1]
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus))
    if num_gpus > 0:
        base_model = torch.nn.DataParallel(model, device_ids=device_ids)
        model_parameters  = base_model.module.parameters()
    else:
        base_model = model
        model_parameters  = base_model.parameters()
    base_model.to(DEVICE)
    return base_model,model_parameters,input_size