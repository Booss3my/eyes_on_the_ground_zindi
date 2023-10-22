import pandas as pd 
from config import *
import os
from dataset.data import EogDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.model import *
import torch
from torch import nn
from tqdm import tqdm
import wandb
from utils import EarlyStopper
import joblib
from lion_pytorch import Lion


data  = pd.read_csv(os.path.join(DATA_ROOT_PATH,"train.csv"))
mask = data['filename'].apply(lambda x: len(x.split(" ")) <= 1)
data = data.loc[mask].sample(frac=SAMPLE_FRAC,random_state= 10)
label = (data.extent==0).astype("uint8")

train_im_idx, val_im_idx, train_lab, val_lab =train_test_split(data.index,label,test_size=0.2,random_state=10,shuffle=True)

train_image_paths = [os.path.join(IMAGE_PATH,data.filename[filename_idx]) for filename_idx in train_im_idx]
val_image_paths =  [os.path.join(IMAGE_PATH,data.filename[filename_idx]) for filename_idx in val_im_idx]

train_dataset = EogDataset(train_image_paths, labels = train_lab.values,size=input_size, tfs=TRAIN_TFS)
val_dataset = EogDataset(val_image_paths, labels = val_lab.values,size=input_size,tfs=VAL_TFS)

train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)
val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)

criterion = nn.CrossEntropyLoss()
optimizer = Lion(model_parameters, lr=LR, weight_decay=1e-2)
# optimizer = torch.optim.Adam(lr=LR,params=base_model.module.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

wandb.login(key=WANDB_KEY)
config = dict(test_name = "classification", learning_rate=LR, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, frac_data_used=SAMPLE_FRAC, Image_size = input_size, model_name=model_name, train_indexes = train_im_idx,val_indexes=val_im_idx)
joblib.dump(config, "train_config.pkl")
wandb.init(project="eyes_on_the_ground", config=config)

best_val_loss = 1e10
early_stopper = EarlyStopper(patience=3, min_delta=0.5)
for i in range(NUM_EPOCHS):
    #train
    running_loss = 0    
    for j,(images,labels) in tqdm(enumerate(train_dataloader),f'Iterating through {len(train_dataloader)} batches'):   
        y = base_model(images.to(DEVICE)).squeeze()
        loss  = criterion(y,labels.to(DEVICE))
        (loss/N_GRAD_CUMUL).backward()
        # nn.utils.clip_grad_norm_(base_model.module.parameters(), 1.0)
        if j%N_GRAD_CUMUL==N_GRAD_CUMUL-1 or j==len(train_dataloader)-1: 
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss+=loss
    
    wandb.log({"train loss": running_loss/len(train_dataloader), "epoch": i})
    print(f'epoch {i}/{NUM_EPOCHS}: Training cross_entropy {running_loss/len(train_dataloader)}')
    print("training classification report",classification_report(labels.detach().cpu().type(torch.uint8),(y.detach().cpu()>0.5).type(torch.uint8)))
    
    #val
    if i%3==2: 
        val_running_loss=0
        for images,labels in tqdm(val_dataloader,f'Iterating through {len(val_dataloader)} batches'):
            with torch.no_grad():
                 y = base_model(images.to(DEVICE)).squeeze()
                 loss  = criterion(y,labels.to(DEVICE))
                 val_running_loss+=loss
        
        lb_loss = val_running_loss/len(val_dataloader) 
        if lb_loss<best_val_loss:
            torch.save(base_model.module.state_dict(), MODEL_SAVE_PATH)
            best_val_loss = lb_loss
        
        if early_stopper.early_stop(val_running_loss/len(val_dataloader) ):             
            break

        wandb.log({"val loss": lb_loss, "epoch": i})
        print(f'epoch {i}/{NUM_EPOCHS}: Validation cross entropy {lb_loss}')
        print("validation classification report",classification_report(labels.detach().cpu(),(y.detach().cpu()>0.5).type(torch.uint8)))

    scheduler.step()