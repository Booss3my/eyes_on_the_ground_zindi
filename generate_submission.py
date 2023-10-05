from dataset.data import EogDataset
from torch.utils.data import DataLoader
from config import *
from models.model import *
from tqdm import tqdm
import pandas as pd 
import os
import torch
from predict import predict


data  = pd.read_csv(os.path.join(DATA_ROOT_PATH,"Test.csv"))
submission_file= pd.read_csv(os.path.join(DATA_ROOT_PATH,"SampleSubmission.csv"))
pred_fnames = submission_file.merge(data, how="inner", on="ID").filename
pred_image_paths = [os.path.join(IMAGE_PATH,filename) for filename in pred_fnames]

#averaging predictions only if TTA
averaged_predictions = predict(model_path,pred_image_paths,averaging_iter=1,tfs=VAL_TFS)
submission = submission_file.copy()        
submission["extent"] = (100*abs(averaged_predictions)).type(torch.int).cpu()
submission.to_csv("submission_ob.csv",index=False)