########################################
# NOTE THIS IS FULL VERSION WORK IN PROGRESS
# IM ATTEMPTING DIFFERENT METHODS AS I KEEP RUNNING INTO A PROBLEM
# NOT ENOUGH VRAM!

# THE notebook.ipynb IS COMPLETE SO PLEASE REFER TO THAT FOR NOW
# AND NOT main.py   utils/   data/
########################################


import torch
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

from utils.model_train import train
from utils.processing import load_data, process_data, augment_data, to_tensors, split_batch, incremental_save
from utils.quickdraw_cnn import QuickDrawCNN_V1, QuickDrawCNN_V2

# base varaiables
image_dir = "data/numpy_bitmap/"
categories = os.listdir(image_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels_map = {label: i for i, label in enumerate(set(categories))}
parent_path = 'saves'
model_path = f'{parent_path}/model'
var_path = f'{parent_path}/vars'

# pipeline function
def pipeline(image_dir: str, categories: list, device: torch.device, labels_map: dict):
    
    '''
    A pipeline function to run the entire process of loading, processing, augmenting, and splitting the data onto each category at a time.
    '''
    
    train_datasets, test_datasets, val_datasets = [], [], []

    for cat in tqdm(categories, desc="Processing categories"):
        
        #load, process, augment, and split the data
        features, label = load_data(image_dir, cat, file_standardize=False)
        features, label = process_data(features, label)
        features, label = augment_data(features, label, rot=0, h_flip=False, v_flip=False)
        features, labels = to_tensors(features, label, labels_map, device=device)
        
        #split the data into train, test, and validation sets
        train_loader, test_loader, val_loader = split_batch(features, labels, batch_size=64)
        train_datasets.append(train_loader.dataset)
        test_datasets.append(test_loader.dataset)
        val_datasets.append(val_loader.dataset)
    
    # Concatenate the dataloaders
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=64, shuffle=True)
    test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=64, shuffle=False)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=64, shuffle=False)
    
    return train_loader, test_loader, val_loader

# run the pipeline
train_loader, test_loader, val_loader = pipeline(image_dir, categories, device, labels_map)

# define the imported V2 CNN model
model = QuickDrawCNN_V2().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
try:
    train_loss, val_loss, train_acc, val_acc = train(model=model,
                                                     train_loader=train_loader,
                                                     val_loader=val_loader,
                                                     epochs=10,
                                                     criterion=criterion,
                                                     optimizer=optimizer,
                                                     device=device)
except:
    print("Training interrupted manually...")
    torch.save(model.state_dict(), "model_state.pth")
    print("Model state saved.")

# save the model and variables incrementally
varaibles_saved = incremental_save(var_path)
with open(varaibles_saved, "wb") as f:
    pickle.dump(labels_map, f)
    pickle.dump(train_loss, f)
    pickle.dump(val_loss, f)
    pickle.dump(train_acc, f)
    pickle.dump(val_acc, f)

model_saved = incremental_save(model_path, data=model)
