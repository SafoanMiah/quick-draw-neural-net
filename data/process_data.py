import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

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