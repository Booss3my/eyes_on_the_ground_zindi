import timm
import torch
from config import *

model= timm.create_model('efficientnet_b1_pruned', pretrained=True)
model.classifier = torch.nn.Linear(in_features=1280,out_features=1,bias=True)
base_model = torch.nn.DataParallel(model, device_ids=[0, 1])
base_model.to(DEVICE)