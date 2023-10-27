from config import *
import torch
from tqdm import tqdm
from models.model import *
def one_epoch(base_model, dataloader, criterion, optimizer,type="train"):
    running_loss=0
    if type=="train":
        for j,(images,labels) in tqdm(enumerate(dataloader),f'Iterating through {len(dataloader)} batches'):   
            y = base_model(images.to(DEVICE)).squeeze()
            loss  = torch.sqrt(criterion(y,labels.to(DEVICE)))
            (loss/N_GRAD_CUMUL).backward()

            if j%N_GRAD_CUMUL==N_GRAD_CUMUL-1 or j==len(dataloader)-1: 
                optimizer.step()
                optimizer.zero_grad()
            running_loss+=loss
        
    elif type=="validation":       
        for images,labels in tqdm(dataloader,f'Iterating through {len(dataloader)} batches'):
            with torch.no_grad():
                y = base_model(images.to(DEVICE)).squeeze()
                loss  = torch.sqrt(criterion(y,labels.to(DEVICE)))
                running_loss+=loss
        
    lb_loss = 100*running_loss/len(dataloader) 
    return lb_loss
