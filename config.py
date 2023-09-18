import albumentations as A 
from albumentations.pytorch import ToTensorV2
import argparse
import torch 

parser = argparse.ArgumentParser(description="Configurations for my project")

parser.add_argument('--DATA_ROOT_PATH', type=str, default=r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi\eog_dataset", help="The root path where your data is stored")
parser.add_argument('--IMAGE_PATH', type=str, default=r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi\eog_dataset\images", help="The path to the directory where your images are stored")
parser.add_argument('--ROOT_PATH', type=str, default=r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi", help="The main root path for the project")
parser.add_argument('--BATCH_SIZE', type=int, default=32, help="The batch size to be used during training")
parser.add_argument('--LR', type=float, default=1e-3, help="The learning rate to be used during training")
parser.add_argument('--NUM_EPOCHS', type=int, default=10, help="The number of epochs to train for")
parser.add_argument('--NUM_DL_WORKERS', type=int, default=0, help="Num workers for dataloader")
parser.add_argument('--SAMPLE_FRAC', type=float, default=1.0, help="fraction of data to keep")
parser.add_argument('--WANDB_KEY', type=str, help="API key for wandb")
parser.add_argument('--SCHEDULER_STEP', type=int, default=10, help="steplr scheduler step param")
parser.add_argument('--SCHEDULER_GAMMA', type=float, default=0.3, help="Gamma scheduler")
parser.add_argument('--MODEL_SAVE_PATH', type=str, default="best_model.ckpt", help="model save path")

#predict only parameters
parser.add_argument('--model_path', type=str, help="Path to model")  
parser.add_argument('--submission_path', type=str, help="Path to submission file")

args = parser.parse_args()

DATA_ROOT_PATH = args.DATA_ROOT_PATH
IMAGE_PATH = args.IMAGE_PATH
ROOT_PATH = args.ROOT_PATH
BATCH_SIZE = args.BATCH_SIZE
LR = args.LR
NUM_EPOCHS = args.NUM_EPOCHS
NUM_DL_WORKERS=args.NUM_DL_WORKERS
SAMPLE_FRAC = args.SAMPLE_FRAC
WANDB_KEY=args.WANDB_KEY
SCHEDULER_STEP = args.SCHEDULER_STEP
SCHEDULER_GAMMA = args.SCHEDULER_GAMMA
MODEL_SAVE_PATH = args.MODEL_SAVE_PATH
model_path = args.model_path
submission_path = args.submission_path

TRAIN_TFS = A.Compose([
    A.Transpose(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(transpose_mask=True)
])

VAL_TFS = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(transpose_mask=True)
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


 