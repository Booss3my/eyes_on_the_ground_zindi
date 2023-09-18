import os
import sys
import albumentations as A
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import * 


class eog_Dataset(Dataset):
    def __init__(self, image_paths, labels=None, tfs=None, size=224):
        super().__init__()

        self.tfs = tfs
        self.image_paths = image_paths
        self.labels = labels
        self.resize_ts = A.Compose([
            A.SmallestMaxSize(int(size), p=1.0),
            A.CenterCrop(size, size, p=1.0)])


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        pillow_image = Image.open(self.image_paths[index])
        out = np.array(pillow_image)
        out = self.resize_ts(image=out)['image']

        if self.tfs is not None:
            out = self.tfs(image=out)['image']

        if self.labels is None:
            return out
        return out, torch.tensor(self.labels[index]/100,dtype=torch.float32)
