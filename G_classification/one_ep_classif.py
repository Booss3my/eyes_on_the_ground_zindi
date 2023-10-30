from config import *
import torch
from tqdm import tqdm
from models.model import *
import pandas as pd
import os
from dataset.data import EogDataset
from torch.utils.data import DataLoader


def one_epoch_classif(base_model, dataloader, criterion, optimizer,type="train"):
    running_loss=0
    if type=="train":
        for j,(images,labels) in tqdm(enumerate(dataloader),f'Iterating through {len(dataloader)} batches'):   
            y = base_model(images.to(DEVICE)).squeeze()
            labels =(100*labels).to(DEVICE)
            loss  = criterion(y,labels)
            (loss/N_GRAD_CUMUL).backward()

            if j%N_GRAD_CUMUL==N_GRAD_CUMUL-1 or j==len(dataloader)-1: 
                optimizer.step()
                optimizer.zero_grad()
            running_loss+=loss
        
    elif type=="validation":       
        for images,labels in tqdm(dataloader,f'Iterating through {len(dataloader)} batches'):
            with torch.no_grad():
                y = base_model(images.to(DEVICE)).squeeze()
                loss  = criterion(y,labels.to(DEVICE))
                running_loss+=loss
        
    lb_loss = running_loss/len(dataloader) 
    return lb_loss


def WithModelPredictClass(base_model,dataloader):
    test_output=torch.tensor([]).to(DEVICE)
    for images in tqdm(dataloader,f'Iterating through {len(dataloader)} batches'):
            with torch.no_grad():
                test_output = torch.cat((test_output,base_model.eval()(images.to(DEVICE)).squeeze()))
    return test_output.cpu()

def Predict_(base_model,input_size,submission_df,text):
    data  = pd.read_csv("/kaggle/input/subm-file/Test.csv")
    pred_fnames = submission_df.merge(data, how="inner", on="ID").filename
    pred_image_paths = [os.path.join(TEST_IMAGE_PATH,filename) for filename in pred_fnames]
    pred_dataset = EogDataset(pred_image_paths,size=input_size,tfs=VAL_TFS)
    pred_dataloader = DataLoader(pred_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_DL_WORKERS)
    prediction = WithModelPredictClass(base_model,pred_dataloader)       
    submission_df[f"extent_{text}"] = prediction
