import pandas as pd 
from config import *
from utils import display_image_tensor
import os
from dataset.data import eog_Dataset
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

data  = pd.read_csv(os.path.join(DATA_ROOT_PATH,"train.csv"))

train_im, val_im, train_lab, val_lab =train_test_split(data.filename,data.extent,test_size=0.33,random_state=10,shuffle=True)

train_image_paths = [os.path.join(IMAGE_PATH,filename) for filename in train_im]
val_image_paths =  [os.path.join(IMAGE_PATH,filename) for filename in val_im]

train_dataset = eog_Dataset(train_image_paths, labels = train_lab.values,tfs=TRAIN_TFS)
val_dataset = eog_Dataset(val_image_paths, labels = val_lab.values,tfs=TRAIN_TFS)

train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)
val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)


