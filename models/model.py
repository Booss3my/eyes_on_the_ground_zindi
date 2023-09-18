import timm
import torch
from config import *


model= timm.create_model(model_name, pretrained=True)
model_cfg = model.default_cfg
input_size = model_cfg["input_size"][1]
in_feats = model.get_classifier().in_features


setattr(model,model_cfg["classifier"],torch.nn.Linear(in_features=in_feats,out_features=1,bias=True))

base_model = torch.nn.DataParallel(model, device_ids=[0, 1])
base_model.to(DEVICE)