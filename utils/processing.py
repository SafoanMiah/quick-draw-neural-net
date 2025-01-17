import os
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

from torch.utils.data import DataLoader, TensorDataset
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
        labels.extend([cat] * len(data)) # repeat the label for each image, add list to list
        
    features = np.vstack(features)
    labels = np.array(labels)
    
    print(f"Data loaded, size: {features.shape}")
    return features, labels




def process_data(features_p1: np.array, features_p2: np.array, labels_p1: np.array, labels_p2: np.array, reshape_size: tuple = (-1, 28, 28), portion: float = None):
    
    '''
    Process the data
    
    args:
    - features_p1: numpy array with the features of the first dataset
    - features_p2: numpy array with the features of the second dataset
    - labels_p1: numpy array with the labels of the first dataset
    - labels_p2: numpy array with the labels of the second dataset
    - reshape_size: size to reshape the features (this reshapes the flat array into a 2D array (image) with the specified size)
    - portion: portion of the data to use (this portions the data, removing some layers, leave empty to use all the data)
    
    returns: features, labels
    '''
    
    assert len(features_p1) == len(labels_p1)
    assert len(features_p2) == len(labels_p2)

    if portion:
        mask_p1 = np.random.rand(len(features_p1)) <= portion # mask for datast, portion false
        mask_p2 = np.random.rand(len(features_p2)) <= portion
        
        features_p1 = features_p1[mask_p1]
        features_p2 = features_p2[mask_p2]
        labels_p1 = labels_p1[mask_p1]
        labels_p2 = labels_p2[mask_p2]

    # keeping masked data
    features = np.concatenate([features_p1, features_p2], axis=0)
    labels = np.concatenate([labels_p1, labels_p2], axis=0)

    # reshaping the features
    features = features.reshape(reshape_size)
    
    # label renaming remove .npy
    labels = [label.split(".")[0] for label in labels]
    

    print(f"Original size: {len(labels_p1) + len(labels_p2)}")
    print(f"Reduced size: {len(labels)}")
    
    return features, labels





def to_tensors_split(features: np.array, labels: np.array, device: str = 'cpu'):
    '''
    Normalize objects convert the features and labels into tensor objects
    
    args:
    - features: numpy array with the features
    - labels: numpy array with the labels
    - device: device to use
    
    returns: features, labels
    '''
    
    print(f"Converting {len(labels)} images to tensors, using {device}...")
    
    labels_map = {label: i for i, label in enumerate(set(labels))}
        
    labels = [labels_map[label] for label in labels] # convert labels to integers
    
    features = torch.tensor(features).to(device) / 255.0 # normalizing 0-255 -> 0-1
    features = features.unsqueeze(1) # adding channel dimension for CNN, 1 channel for grayscale
    labels = torch.tensor(labels).to(device)
    
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    return features, labels, labels_map





def to_tensors(features: np.array, labels: np.array, device: str = 'cpu'):
    '''
    Normalize objects convert the features and labels into tensor objects
    
    args:
    - features: numpy array with the features
    - labels: numpy array with the labels
    - device: device to use
    
    returns: features, labels
    '''
    
    print(f"Converting {len(labels)} images to tensors, using {device}...")
    
    labels_map = {label: i for i, label in enumerate(set(labels))}
        
    labels = [labels_map[label] for label in labels] # convert labels to integers
    
    features = torch.tensor(features).to(device) / 255.0 # normalizing 0-255 -> 0-1
    features = features.unsqueeze(1) # adding channel dimension for CNN, 1 channel for grayscale
    labels = torch.tensor(labels).to(device)
    
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    return features, labels, labels_map





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




def split_batch(features: np.array, labels: np.array, batch_size: int = 64):
    
    '''
    Split the dataframe into train, test, and validation sets
    Turn the sets into DataLoader objects
    
    args:
    - features: numpy array with the features
    - labels: numpy array with the labels
    - batch_size: size of the batch
    
    returns: train_loader, test_loader, val_loader
    '''
    
    print(f"Splitting {features.shape} images into train, test, and validation sets...")
    
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
    
    #sizes
    print(f"Train size: {len(train_loader.dataset)} \nValidation size: {len(val_loader.dataset)} \nTest size: {len(test_loader.dataset)}")
    
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