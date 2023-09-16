import pandas as pd 
from config import *
import os
from dataset.data import eog_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import base_model
import torch
from torch import nn
from tqdm import tqdm

data  = pd.read_csv(os.path.join(DATA_ROOT_PATH,"train.csv"))

train_im, val_im, train_lab, val_lab =train_test_split(data.filename,data.extent,test_size=0.33,random_state=10,shuffle=True)

train_image_paths = [os.path.join(IMAGE_PATH,filename) for filename in train_im]
val_image_paths =  [os.path.join(IMAGE_PATH,filename) for filename in val_im]

train_dataset = eog_Dataset(train_image_paths, labels = train_lab.values,tfs=TRAIN_TFS)
val_dataset = eog_Dataset(val_image_paths, labels = val_lab.values,tfs=TRAIN_TFS)

train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)
val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lr=LR,params=base_model.parameters())
#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    


for i in range(NUM_EPOCHS):
    running_loss = 0    
    for images,labels in tqdm(train_dataloader,f'Iterating through {len(train_dataloader)} batches'):   
        y = base_model(images)
        loss  = criterion(y,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss+=loss
    print(f'epoch {i}/{NUM_EPOCHS}: Training MSE {running_loss/len(train_dataloader)}')
