import pandas as pd 
from config import *
import os
from dataset.data import EogDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.model import *
import torch
from torch import nn
from tqdm import tqdm
import wandb
from utils import EarlyStopper
from lion_pytorch import Lion
import torch_xla
import torch_xla.core.xla_model as xm


device = xm.xla_device()

# ... (keep the data loading and splitting part same) ...

base_model.to(device)

data  = pd.read_csv(os.path.join(DATA_ROOT_PATH,"train.csv"))
mask = data['filename'].apply(lambda x: len(x.split(" ")) <= 1)
data = data.loc[mask].sample(frac=SAMPLE_FRAC,random_state= 10)

train_im, val_im, train_lab, val_lab =train_test_split(data.filename,data.extent,test_size=0.2,random_state=10,shuffle=True)

train_image_paths = [os.path.join(IMAGE_PATH,filename) for filename in train_im]
val_image_paths =  [os.path.join(IMAGE_PATH,filename) for filename in val_im]

train_dataset = EogDataset(train_image_paths, labels = train_lab.values,size=input_size, tfs=TRAIN_TFS)
val_dataset = EogDataset(val_image_paths, labels = val_lab.values,size=input_size,tfs=VAL_TFS)

train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)
val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)


criterion = nn.MSELoss()

optimizer = Lion(base_model.module.parameters(), lr=LR, weight_decay=1e-2)
# optimizer = torch.optim.Adam(lr=LR,params=base_model.module.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

wandb.login(key=WANDB_KEY)
config = dict(learning_rate=LR, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, frac_data_used=SAMPLE_FRAC, Image_size = input_size)
wandb.init(project="eyes_on_the_ground", config=config)


best_val_loss = 1e10
early_stopper = EarlyStopper(patience=3, min_delta=0.5)

for i in range(NUM_EPOCHS):
    # train
    running_loss = 0    
    for j, (images, labels) in tqdm(enumerate(train_dataloader), f'Iterating through {len(train_dataloader)} batches'):   
        y = base_model(images.to(device)).squeeze()  # Change DEVICE to device
        loss = torch.sqrt(criterion(y, labels.to(device)))  # Change DEVICE to device
        (loss/N_GRAD_CUMUL).backward()
        # nn.utils.clip_grad_norm_(base_model.module.parameters(), 1.0)
        if j%N_GRAD_CUMUL==N_GRAD_CUMUL-1 or j==len(train_dataloader)-1: 
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss+=loss
    
    wandb.log({"train loss": 100*running_loss/len(train_dataloader), "epoch": i})
    print(f'epoch {i}/{NUM_EPOCHS}: Training RMSE {100*running_loss/len(train_dataloader)}')
    
    #val
    if i%3==2: 
        val_running_loss=0
        for images,labels in tqdm(val_dataloader,f'Iterating through {len(val_dataloader)} batches'):
            with torch.no_grad():
                 y = base_model(images.to(DEVICE)).squeeze()
                 loss  = torch.sqrt(criterion(y,labels.to(DEVICE)))
                 val_running_loss+=loss
        
        lb_loss = 100*val_running_loss/len(val_dataloader) 
        if lb_loss<best_val_loss:
            torch.save(base_model.module.state_dict(), MODEL_SAVE_PATH)
            best_val_loss = lb_loss
        
        if early_stopper.early_stop(100*val_running_loss/len(val_dataloader) ):             
            break

        wandb.log({"val loss": lb_loss, "epoch": i})
        print(f'epoch {i}/{NUM_EPOCHS}: Validation RMSE {lb_loss}')


    scheduler.step()
