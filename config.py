import albumentations as A 
from albumentations.pytorch import ToTensorV2


DATA_ROOT_PATH =  r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi\eog_dataset"
IMAGE_PATH = r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi\eog_dataset\images"
ROOT_PATH = r"C:\Users\oussa\OneDrive\Desktop\eyes_on_the_ground_zindi"



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


BATCH_SIZE = 32
NUM_DL_WORKERS = 0 