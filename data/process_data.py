import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

tqdm.pandas()

def load_data(categories: list, dir: str, file_standardize: bool = False, sample: int = 0):
    '''
    Load data from every .npy files in `dir` directory, and stack them onto one dataframe.
    
    args:
    - categories: list of directories where the files are stored
    - dir: directory where the files are store
    - file_standardize: if True, the function will standarize the file    
    - sample: instead of doin the whole dataset, take a sample
    
    returns:
    - features: numpy array with the features
    - labels: list with the labels
    '''
    
    num_categories = len(categories)

    # if needed standarize the file names
    if file_standardize:
        for filename in os.listdir(dir):
            os.rename(dir + filename, dir + filename.replace(" ", "_").lower())
            
    if sample:
        categories = categories[: (num_categories//100 * sample) ]
        print(f"Loading {num_categories//100 * sample} categories...")
    else:
        print(f"Loading {num_categories} categories...")
    
    # iterate over all files in the dirs and add to a dataframe
    
    features = []
    labels = []

    for cat in tqdm(categories, desc="Loading data"):
        data = np.load(os.path.join(dir + cat))
        features.append(data)
        labels.append(cat)
        
    features = np.vstack(features)
    labels = np.array(labels)
    
    print(f"Data loaded, size: {features.shape}")
    return features, labels





def preprocess_data(features: np.array, labels: np.array, reshape_size: tuple = (28, 28)):
    
    '''
    Preprocess the data by reshaping the features and removing the .npy extension from the labels
    
    args:
    - features: numpy array with the features
    - labels: numpy array with the labels
    - reshape_size: size to reshape the features
    
    returns: preprocessed dataframe
    '''
    
    print(f"Preprocessing {len(labels)} images...")
    
    features = features.reshape(-1, *reshape_size) # leaves first dimension as is, reshapes the rest to 28x28
    labels = [labels[i].split(".")[0] for i in tqdm(range(len(labels)), desc="Processing labels")] # remove .npy extension

    return features, labels





def to_tensors(features: np.array, labels: np.array, labels_map: dict, device: str = 'cpu'):
    
    '''
    Normalize objects convert the features and labels into tensor objects
    
    args:
    - features: numpy array with the features
    - labels: numpy array with the labels
    - labels_map: dictionary with the labels mapping
    - device: device to use
    
    returns: features, labels, labels_map
    '''
    
    print(f"Converting {len(labels)} images to tensors, using {device}...")
        
    labels = [labels_map[label] for label in labels] # convert labels to integers
    
    features = torch.tensor(features).to(device) / 255.0 # normalizing 0-255 -> 0-1
    features = features.unsqueeze(1) # adding channel dimension for CNN, 1 channel for grayscale
    labels = torch.tensor(labels).to(device)
    
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    return features, labels





def augment_data(features: np.array, labels: np.array, rot: int = 0, h_flip: bool = False, v_flip: bool = False) -> tuple:
    
    '''
    Augment the data using the following optional techniques:
    - Rotating the image by `angle` degrees
    - Flipping the image horizontally
    - Flipping the image vertically
    
    args:
    features: numpy array with the features
    labels: numpy array with the labels
    rot: rotate the image by angle
    h_flip: flip the image horizontally
    v_flip: flip the image vertically
    
    returns: augmented features and labels
    '''
    
    print(f'Starting size: {features.shape}')
    
    augmented_features = [features]
    augmented_labels = [labels]

    
    # rotating image by `rot` degrees
    if rot:
        print(f"Rotating images by {rot} degrees...")
        rotated_features = np.array([rotate(img, rot, reshape=False) for img in tqdm(features, desc="Rotating images...")])
        augmented_features.append(rotated_features) # append the rotated images to the list of augmented
        augmented_labels.append(labels) # append the labels to the list of augmented
        
    # flipping image horizontally
    if h_flip:
        print("Flipping images horizontally...")
        hflipped_features = np.array([np.fliplr(img) for img in tqdm(features, desc="Flipping images horizontally...")])
        augmented_features.append(hflipped_features)
        augmented_labels.append(labels)
        
    # flipping image vertically
    if v_flip:
        print("Flipping images vertically...")
        vflipped_features = np.array([np.flipud(img) for img in tqdm(features, desc="Flipping images vertically...")])
        augmented_features.append(vflipped_features)
        augmented_labels.append(labels)
        
    augmented_features = np.concatenate(augmented_features)
    augmented_labels = np.concatenate(augmented_labels)
    
    print(f'Augmented size: {augmented_features.shape}')
    return augmented_features, augmented_labels




def split_batch(df: pd.DataFrame, batch_size: int = 64):
    
    '''
    Split the dataframe into train, test, and validation sets
    Turn the sets into DataLoader objects
    
    args:
    - df: dataframe to split
    - batch_size: size of the batch
    
    returns: train_loader, test_loader, val_loader
    '''
    
    print(f"Splitting {df.shape[0]} images into train, test, and validation sets...")
    
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=0)
    val_data, test_data = train_test_split(test_data, test_size=0.5, train_size=0.5, random_state=0)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader


