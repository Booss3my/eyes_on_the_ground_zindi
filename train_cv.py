import pandas as pd 
from config import *
import os
from dataset.data import EogDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from models.model import *
import torch
from torch import nn
from tqdm import tqdm
import wandb
from utils import EarlyStopper
import joblib
from lion_pytorch import Lion
from train_one_ep import one_epoch

n_splits = 5
data  = pd.read_csv(os.path.join(DATA_ROOT_PATH,"train.csv"))
mask = data['filename'].apply(lambda x: len(x.split(" ")) <= 1)
data = data.loc[mask].sample(frac=SAMPLE_FRAC,random_state= 10).reset_index(drop=True)


skf = StratifiedKFold(n_splits=n_splits,shuffle=True)
for i, (train_index, val_index) in enumerate(skf.split(data.index,data.extent)):
    
    train_image_paths = [os.path.join(IMAGE_PATH,data.filename[filename_idx]) for filename_idx in train_index]
    val_image_paths =  [os.path.join(IMAGE_PATH,data.filename[filename_idx]) for filename_idx in val_index]

    train_dataset = EogDataset(train_image_paths, labels = data.extent[train_index].values,size=input_size, tfs=TRAIN_TFS)
    val_dataset = EogDataset(val_image_paths, labels =  data.extent[val_index].values,size=input_size,tfs=VAL_TFS)

    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)
    val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)
    
    criterion = nn.MSELoss()
    optimizer = Lion(base_model.module.parameters(), lr=LR, weight_decay=1e-2)
    val_losses = []
    for j in range(NUM_EPOCHS):
        train_loss = one_epoch(base_model, train_dataloader, criterion, optimizer,type="train")
        print(f"Epoch {j+1}/{NUM_EPOCHS} --- Training loss :{train_loss}")
        if j%5==4 | j+1==NUM_EPOCHS:
            val_loss = one_epoch(base_model, train_dataloader, criterion, optimizer,type="validation")
            print(f"Epoch {j+1}/{NUM_EPOCHS} --- Validation loss :{val_loss}")
        if j+1==NUM_EPOCHS:
            val_losses.append(val_loss)

    
        
    


