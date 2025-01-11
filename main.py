import torch
from torch.utils.data import DataLoader
from data.process_data import load_data, augment_data, preprocess_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
image_dir = "data/numpy_bitmap/"
df = load_data(image_dir, file_standardize=True, sample=True)
processed = preprocess_data(df)
augmented = augment_data(processed, rot=True, angle=15, h_flip=True, v_flip=True)

print(augmented.head())
print(augmented.shape)