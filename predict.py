from dataset.data import eog_Dataset
from torch.utils.data import DataLoader
from config import *
from models.model import *
from tqdm import tqdm
import pandas as pd 
import os
import torch

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
base_model = torch.nn.DataParallel(model, device_ids=[0, 1])
base_model.to(DEVICE)


data  = pd.read_csv(os.path.join(DATA_ROOT_PATH,"Test.csv"))
submission_file= pd.read_csv(os.path.join(DATA_ROOT_PATH,"SampleSubmission.csv"))
pred_fnames = submission_file.merge(data, how="inner", on="ID").filename

pred_image_paths = [os.path.join(IMAGE_PATH,filename) for filename in pred_fnames]
averaged_output=torch.zeros(len(pred_image_paths)).to(DEVICE)
for _ in range(5):
    pred_dataset = eog_Dataset(pred_image_paths,size=input_size,tfs=TRAIN_TFS)
    pred_dataloader = DataLoader(pred_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_DL_WORKERS)

    pred_iter=iter(pred_dataloader)
    test_output=torch.tensor([]).to(DEVICE)
    for data in tqdm(pred_iter):
        with torch.no_grad():
            test_output = torch.cat((test_output,base_model.eval()(data.to(DEVICE)).squeeze()))
    
    averaged_output+=test_output/5

submission = submission_file.copy()        
submission["extent"] = (100*abs(averaged_output)).type(torch.int).cpu()
submission.to_csv("submission_ob.csv",index=False)