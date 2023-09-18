import timm
import torch
from config import *

model_name = 'efficientnet_b1_pruned' 
model= timm.create_model(model_name, pretrained=True)
model.classifier = torch.nn.Linear(in_features=1280,out_features=1,bias=True)
base_model = torch.nn.DataParallel(model, device_ids=[0, 1])
base_model.to(DEVICE)