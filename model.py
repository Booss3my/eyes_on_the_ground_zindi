import timm
import torch
from config import *

base_model= timm.create_model('efficientnet_b1_pruned', pretrained=True)
base_model.classifier = torch.nn.Linear(in_features=1280,out_features=1,bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_model = torch.nn.DataParallel(base_model, device_ids=[0, 1])
base_model.to(device)