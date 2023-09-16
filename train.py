import pandas as pd 
from config import *
from utils import display_image_tensor
import os
from dataset.data import eog_Dataset
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
train_data  = pd.read_csv(os.path.join(DATA_ROOT_PATH,"train.csv"))
image_paths = [os.path.join(IMAGE_PATH,filename) for filename in train_data.filename]

dataset = eog_Dataset(image_paths, labels = train_data.extent,tfs=TRAIN_TFS)

DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)
display_image_tensor(dataset.__getitem__(100)[0])

