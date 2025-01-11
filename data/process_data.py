import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

from torch.utils.data import DataLoader

import torch

def load_data(dir: str, file_standardize: bool = False, sample: bool = False, sample_size: int = 1) -> pd.DataFrame:
    '''
    Load data from every .npy files in `dir` directory, and stack them onto one dataframe.
    
    args:
    - dir: directory where the files are stored
    - file_standardize: if True, the function will standarize the file    
    - sample: if True, the function will return a sample of the data
    - sample_size: size of the sample, as a percentage of the total data
    
    returns: dataframe: pd.DataFrame
    '''
    
    assert os.path.exists(dir), f"Directory {dir} does not exist."
    
    # if needed standarize the file namess
    if file_standardize:
        for filename in os.listdir(dir):
            os.rename(dir + filename, dir + filename.replace(" ", "_").lower())
            
    categories = os.listdir(dir)
    if sample:
        categories = categories[: (len(categories)//100 * sample_size) ]
    
    print(f"Loading {len(categories)} categories...")
    
    # itarate over all files in the dir and add to a dataframe
    dataframe = pd.DataFrame()
    temp_df = []

    for cat in categories:
        data = pd.DataFrame(np.load(dir + cat))
        data["category"] = cat
        temp_df.append(data)
        
    dataframe = pd.concat(temp_df, ignore_index=True)
    
    return dataframe

def preprocess_data(df: pd.DataFrame):
    
    processed = pd.DataFrame()
    processed["labels"] = df["category"].apply(lambda x: x.split(".")[0]) # remove .npy extension
    processed["features"] = df.iloc[:, :-1].apply(lambda x: np.array(x).reshape(28,28), axis=1) # reshape to 28x28
    
    return processed

def augment_data(df: pd.DataFrame, rot: bool = False, angle: int = 15, h_flip: bool = False, v_flip: bool = False) -> pd.DataFrame:
    
    '''
    Augment the data using the following optional techniques:
    - Rotating the image by `angle` degrees
    - Flipping the image horizontally
    - Flipping the image vertically
    
    args:
    df: dataframe to augment
    rot: rotate the image
    angle: degrees to rotate the image
    h_flip: flip the image horizontally
    v_flip: flip the image vertically
    
    returns: augmented dataframe
    '''
    
    print(f'Starting size: {df.shape}')
    
    augmented = pd.DataFrame()
    
    # rotating image by `angle` degrees
    if rot:
        print(f"Rotating images by {angle} degrees...")
        df_rot = df.copy()
        df_rot['features'] = df_rot['features'].apply(lambda x: rotate(x, 15, reshape=False))
        augmented = pd.concat([augmented, df_rot], ignore_index=True)
        
    # flipping image horizontally
    if h_flip:
        print("Flipping images horizontally...")
        df_hflip = df.copy()
        df_hflip['features'] = df_hflip['features'].apply(lambda x: x[:, ::-1])
        augmented = pd.concat([augmented, df_hflip], ignore_index=True)
        
    # flipping image vertically
    if v_flip:
        print("Flipping images vertically...")
        df_vflip = df.copy()
        df_vflip['features'] = df_vflip['features'].apply(lambda x: x[::-1, :])
        augmented = pd.concat([augmented, df_vflip], ignore_index=True)
        
    print(f'Augmented size: {augmented.shape}')
    return augmented

def to_tensors_batch(df: pd.DataFrame, X: str = 'features', y: str = 'labels'):
    
    '''
    Normalize objects convert the dataframe into tensor objects
    
    args:
    - df: dataframe to convert
    - X: column name for features
    - y: column name for labels
    
    returns: features, labels, labels_map
    '''
    
    unique = df[y].unique()
    labels_map = {}
    
    for n, cat in enumerate(unique):
        labels_map[cat] = n # mapping category to integer, eg: alarm_clock -> 0, shoe -> 1
        
    df[y] = df[y].map(labels_map)
    
    features = torch.tensor(np.array(df[X].tolist()))
    labels = torch.tensor([x for x in df[y]])
    
    features = features.float() / 255.0 # normalizing 0-255 -> 0-1
    features = features.unsqueeze(1) # adding channel dimension for CNN
    
    return features, labels, labels_map

def split_batch(df: pd.DataFrame, batch_size: int = 64):
    
    '''
    Split the dataframe into train, test, and validation sets
    Turn the sets into DataLoader objects
    
    args:
    - df: dataframe to split
    - batch_size: size of the batch
    
    returns: train_loader, test_loader, val_loader
    '''
    
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=0)
    val_data, test_data = train_test_split(test_data, test_size=0.5, train_size=0.5, random_state=0)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader


