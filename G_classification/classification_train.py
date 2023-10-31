import pandas as pd 
import os
import sys
R_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(R_path)

from config import *
from dataset.data import EogDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from models.model import *
from torch import nn
from lion_pytorch import Lion
import wandb
from G_classification.one_ep_classif import Predict_,one_epoch_classif
from utils import seed_everything

seed_everything(SEED)
n_splits = 3
average_losses = {"cv_train_loss":0, "cv_val_loss":0}
data  = pd.read_csv(os.path.join(DATA_ROOT_PATH,"train.csv"))

submission_df= pd.read_csv("/kaggle/input/subm-file/SampleSubmission.csv")  #too late to parametrize -- to do ^^
mask = data['filename'].apply(lambda x: len(x.split(" ")) <= 1)
data = data.loc[mask].sample(frac=SAMPLE_FRAC,random_state= 10).reset_index(drop=True)
label = (data.extent>20).astype("uint8")
skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=SEED)
test_prediction_stack = []

#training
for i, (train_index, val_index) in enumerate(skf.split(data.index,data.extent)):
    
    base_model,model_parameters,input_size = init_model()
    train_image_paths = [os.path.join(TRAIN_IMAGE_PATH,data.filename[filename_idx]) for filename_idx in train_index]
    val_image_paths =  [os.path.join(TRAIN_IMAGE_PATH,data.filename[filename_idx]) for filename_idx in val_index]

    train_dataset = EogDataset(train_image_paths, labels = label[train_index].values,size=input_size, tfs=TRAIN_TFS)
    val_dataset = EogDataset(val_image_paths, labels =  label[val_index].values,size=input_size,tfs=VAL_TFS)

    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)
    val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_DL_WORKERS)

    criterion = nn.BCELoss()
    optimizer = Lion(model_parameters, lr=LR, weight_decay=1e-2)
    
    print(f"######## Fold {i+1}/{n_splits}#####")
    for j in range(NUM_EPOCHS):
        train_loss = one_epoch_classif(base_model, train_dataloader, criterion, optimizer,type="train")
        print(f"Epoch {j+1}/{NUM_EPOCHS} --- Training loss :{train_loss}")
        if j%5==4 or j+1==NUM_EPOCHS:
            val_loss = one_epoch_classif(base_model, train_dataloader, criterion, optimizer,type="validation")
            print(f"Epoch {j+1}/{NUM_EPOCHS} --- Validation loss :{val_loss}")
        if j+1==NUM_EPOCHS:
            average_losses["cv_val_loss"]+=val_loss/n_splits
            average_losses["cv_train_loss"]+=train_loss/n_splits
    #predict for test
    Predict_(base_model,input_size,submission_df,f"fold_{i+1}")
        
print(f"######## train loss {average_losses['cv_train_loss']} -- validation loss {average_losses['cv_val_loss']}#####")

#generate submission
submission_df.to_csv("predictions.csv")

#logging
if wandb_flag:
    wandb.login(key=WANDB_KEY)
    config = dict(learning_rate=LR, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, frac_data_used=SAMPLE_FRAC, model_name=model_name)
    wandb.init(project="eyes_on_the_ground", config=config)
    wandb.log(average_losses)