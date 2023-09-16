import timm
from torch import nn

base_model= timm.create_model('efficientnet_b1_pruned', pretrained=True)
base_model.classifier = nn.Linear(in_features=1280,out_features=1,bias=True)