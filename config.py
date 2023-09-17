import albumentations as A 
from albumentations.pytorch import ToTensorV2
import argparse

parser = argparse.ArgumentParser(description="Configurations for my project")

parser.add_argument('--DATA_ROOT_PATH', type=str, default=r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi\eog_dataset", help="The root path where your data is stored")
parser.add_argument('--IMAGE_PATH', type=str, default=r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi\eog_dataset\images", help="The path to the directory where your images are stored")
parser.add_argument('--ROOT_PATH', type=str, default=r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi", help="The main root path for the project")
parser.add_argument('--BATCH_SIZE', type=int, default=32, help="The batch size to be used during training")
parser.add_argument('--LR', type=float, default=1e-3, help="The learning rate to be used during training")
parser.add_argument('--NUM_EPOCHS', type=int, default=10, help="The number of epochs to train for")
parser.add_argument('--NUM_DL_WORKERS', type=int, default=1, help="Num workers for dataloader")

args = parser.parse_args()

DATA_ROOT_PATH = args.DATA_ROOT_PATH
IMAGE_PATH = args.IMAGE_PATH
ROOT_PATH = args.ROOT_PATH
BATCH_SIZE = args.BATCH_SIZE
LR = args.LR
NUM_EPOCHS = args.NUM_EPOCHS
NUM_DL_WORKERS=args.NUM_DL_WORKERS

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
