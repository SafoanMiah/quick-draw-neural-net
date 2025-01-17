import os
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch
from tqdm import tqdm




def load_data(dir: str, category: str, file_standardize: bool = False, verbose: bool = False):
    
    '''
    Load the bitmaps data from the directory
    
    args:
    - dir: directory path
    - category: category of the data
    - file_standardize: standardize the file names
    
    returns: features, label
    '''

    # if needed standarize the file names
    if file_standardize:
        os.rename(dir + category, dir + category.replace(" ", "_").lower())

    features = np.load(os.path.join(dir + category))
    label = category
    
    if verbose:
        print(f"Loaded              Label: {label} | Features: ({len(features)})")
        
    return features, label




def process_data(features: np.array, label: str, reshape_size: tuple = (-1, 28, 28), portion: float = None, verbose: bool = False):

    '''
    Process data by portioning if needed and reshaping
    
    args:
    - features: features
    - label: label
    - reshape_size: reshape size
    - portion: portion of the data
    
    returns: features, label
    '''

    if portion:
        # mask for datast, portion false, masked to remove data
        mask = np.random.rand(len(features)) <= portion 
        features = features[mask]
        print(f"Portioned size:     Features: {len(features)}")

    # reshaping the features
    features = features.reshape(reshape_size)

    if verbose:
        print(f"Processed           Label: {label} | Features: {features.shape}")
    
    return features, label




def augment_data(features: np.array, label: str, rot: int = 0, h_flip: bool = False, v_flip: bool = False, verbose: bool = False):
    
    '''
    Augment the data by rotating, flipping horizontally and vertically
    
    args:
    - features: features
    - label: label
    - rot: rotation angle
    - h_flip: horizontal flip
    - v_flip: vertical flip
    
    returns: augmented features, label
    '''
    
    augmented_features = [features]
    
    # rotating image by `rot` degrees
    if rot:
        print(f"Rotating images by {rot} degrees...")
        rotated_features = np.array(rotate(features, rot, reshape=False))
        augmented_features.append(rotated_features) # append the rotated images to the list of augmented
        
    # flipping image horizontally
    if h_flip:
        print("Flipping images horizontally...")
        hflipped_features = np.array(np.fliplr(features))
        augmented_features.append(hflipped_features)
        
    # flipping image vertically
    if v_flip:
        print("Flipping images vertically...")
        vflipped_features = np.array(np.flipud(features))
        augmented_features.append(vflipped_features)
    
    augmented_features = np.concatenate(augmented_features)
    
    if verbose:
        print(f'Augmented           Original: {features.shape} | Augmented: {len(augmented_features)}')
        
    return augmented_features, label




def to_tensors(features: np.array, label: str, labels_map: dict, device: str = 'cpu', verbose: bool = False):
    
    '''
    Transform the features and labels to tensors
    
    args:
    - features: features
    - label: label
    - labels_map: labels map
    - device: device to use
    
    returns: features, labels
    '''
    
    label = labels_map[label]
    
    features = torch.tensor(features).to(device) / 255.0 # normalizing 0-255 -> 0-1
    features = features.unsqueeze(1) # adding channel dimension for CNN, 1 channel for grayscale
    
    labels = np.array([label] * len(features))
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    
    if verbose:
        print(f"Tensored            Label: {labels} {labels.shape}| Features: {features.shape} | Device: {device}")
    
    return features, labels




def split_batch(features: np.array, labels: np.array, batch_size: int = 64, verbose: bool = False):
    
    '''
    Split the data into train, test and validation sets, then wrap them in DataLoader
    
    args:
    - features: features
    - labels: labels
    - batch_size: batch size
    
    returns: train_loader, test_loader, val_loader
    '''
    
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

    # Wrap each split in TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if verbose:
        print(f"Split               Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
    
    return train_loader, test_loader, val_loader




def incremental_save(base_path, data = None, seperator="."):
    '''
    Save the data with incremental versioning
    
    args:
    - data: data to save
    - base_path: path to save the data
    - seperator: seperator for the versioning
    
    returns: path of save data
    '''
    
    version = 0
    save_path = f"{base_path}{seperator}{version}.pth"

    # create dir
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # incremental versioning
    while os.path.exists(save_path): 
        version += 1
        save_path = f"{base_path}{seperator}{version}.pth"

    if data:
        torch.save(data.state_dict(), save_path)
        print(f"Saved to {save_path}.")
    else:
        print(f"Path: {save_path}")
        
    return save_path