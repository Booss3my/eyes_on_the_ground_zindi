import timm
import torch
from config import *

base_model= timm.create_model('efficientnet_b1_pruned', pretrained=True)
base_model.classifier = torch.nn.Linear(in_features=1280,out_features=1,bias=True)
base_model = torch.nn.DataParallel(base_model, device_ids=[0, 1])
for param in base_model.parameters():
    param.requires_grad = False

optimisable_params =  base_model.module.classifier.parameters()

for param in optimisable_params:
    param.requires_grad = True

base_model.to(DEVICE)