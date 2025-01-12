import torch
import pickle
import pandas as pd
import os

from data.process_data import load_data, augment_data, preprocess_data, to_tensors, split_batch
from models.quickdraw_cnn import QuickDrawCNN_V1, QuickDrawCNN_V2
from training.model_train import train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
image_dir = "data/numpy_bitmap/"
categories = os.listdir(image_dir)
p1, p2 = categories[:len(categories)//2], categories[len(categories)//2:]
bmap_p1 = load_data(p1, image_dir, file_standardize=False)
bmap_p2 = load_data(p2, image_dir, file_standardize=False)