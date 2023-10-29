import albumentations as A 
from albumentations.pytorch import ToTensorV2
import argparse
import torch 
from utils import seed_everything

parser = argparse.ArgumentParser(description="Configurations for my project")

parser.add_argument('--DATA_ROOT_PATH', type=str, default=r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi\eog_dataset", help="The root path where your data is stored")
parser.add_argument('--TRAIN_IMAGE_PATH', type=str, default=r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi\eog_dataset\images", help="The path to the directory where your train images are stored")
parser.add_argument('--TEST_IMAGE_PATH', type=str, default=r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi\eog_dataset\images", help="The path to the directory where your test images are stored")
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
parser.add_argument('--IM_SIZE', type=int, default=240, help="image size")
parser.add_argument('--N_GRAD_CUMUL', type=int, default=4, help="image size")
#predict only parameters
parser.add_argument('--model_path', type=str, help="Path to model")  
parser.add_argument('--submission_path', type=str, help="Path to submission file")
parser.add_argument('--model_name', type=str,default='efficientnet_b1_pruned', help="Model name")
parser.add_argument('--prediction_image_paths', type=str, help="Paths to images used in predictions, saved as csv file with filename column")
parser.add_argument('--wandb_flag', type=bool, default=False, help="Wandb logging flag")
parser.add_argument('--only_dr', type=bool, default=False, help="keep only Drought damage flag")

args = parser.parse_args()
wandb_flag = args.wandb_flag
DATA_ROOT_PATH = args.DATA_ROOT_PATH
TRAIN_IMAGE_PATH = args.TRAIN_IMAGE_PATH
TEST_IMAGE_PATH = args.TEST_IMAGE_PATH
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
IM_SIZE = args.IM_SIZE
N_GRAD_CUMUL = args.N_GRAD_CUMUL
model_path = args.model_path
submission_path = args.submission_path
model_name = args.model_name
prediction_image_paths = args.prediction_image_paths
only_dr = args.only_dr
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

SEED=10

 