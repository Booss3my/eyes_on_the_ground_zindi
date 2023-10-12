from dataset.data import EogDataset
from torch.utils.data import DataLoader
from config import *
from models.model import *
from tqdm import tqdm
import torch
import pandas as pd
import os
import argparse

def predict(model_path,pred_image_paths,averaging_iter = 5,tfs = VAL_TFS):

    #load model
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus))
    if num_gpus > 0:
        base_model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        base_model = model
    base_model.to(DEVICE)
    
    #inference
    averaged_output=torch.zeros(len(pred_image_paths)).to(DEVICE)
    for _ in range(averaging_iter):
        pred_dataset = EogDataset(pred_image_paths,size=input_size,tfs=tfs)
        pred_dataloader = DataLoader(pred_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_DL_WORKERS)

        pred_iter=iter(pred_dataloader)
        test_output=torch.tensor([]).to(DEVICE)
        for data in tqdm(pred_iter):
            with torch.no_grad():
                test_output = torch.cat((test_output,base_model.eval()(data.to(DEVICE)).squeeze()))
        averaged_output+=test_output/averaging_iter
    return (100*abs(averaged_output)).type(torch.int).cpu()


if __name__ == "__main__":
    pred_image_info = pd.read_csv(prediction_image_paths)
    pred_image_paths = pred_image_info.filename  
    #averaging predictions only if TTA
    averaged_predictions = predict(model_path,pred_image_paths,averaging_iter=1,tfs=VAL_TFS)
    pred_image_info["predicted_extent"] = averaged_predictions
    pred_image_info.to_csv("predictions.csv",index=False)  
    