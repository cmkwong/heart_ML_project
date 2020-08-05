import numpy as np
import pandas as pd
import torch

path = "/data/heart.csv"

def batch_gen(arr, size):
    required_rows = np.random.choice(arr.shape[0], size)
    return torch.tensor(arr[required_rows, :-1], dtype=torch.float32).to(torch.device("cuda")), \
           torch.tensor(arr[required_rows, -1], dtype=torch.long).to(torch.device("cuda"))

def split_data(arr, percentage):
    offset = int(arr.shape[0] * percentage)
    training = arr[:offset,:]
    testing = arr[offset:,:]
    return training, testing

def shuffle(arr):
    np.random.shuffle(arr)
    return arr

def df2array(df):
    return df.values

def array2df(arr, col_list):
    df = pd.DataFrame(arr, columns=col_list)
    return df

def append_col(df_col, df):
    return pd.concat([df,df_col],axis=1)

def normalize_data(series, key):
    max_value = max(series)
    min_value = min(series)
    normalised_series = (series - min_value) / (max_value - min_value)
    df_col = pd.DataFrame({key: normalised_series})
    return df_col

def read_csv(path):
    return pd.read_csv(path)

def out_csv(path, df):
    df.to_csv(path, index=False)
    return True
