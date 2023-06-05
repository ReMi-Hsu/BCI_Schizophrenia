import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

npy_files = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]

def load_data(dataset_root = fr"/local/SSD/bci_dataset_npy/", data_type="filter_asr", split_ratio = 0.8):
    '''
    x shape: [num, time_state, ch]
    y shape: [num]
    '''
    print("="*80)
    print("Load Data")
    ## load data ##
    npy_path = os.path.join(dataset_root, data_type + '_' + npy_files[0])
    x = np.load(npy_path)

    npy_path = os.path.join(dataset_root, data_type + '_' + npy_files[1])
    x_test = np.load(npy_path)

    npy_path = os.path.join(dataset_root, data_type + '_' + npy_files[2])
    y = np.load(npy_path)

    npy_path = os.path.join(dataset_root, data_type + '_' + npy_files[3])
    y_test = np.load(npy_path)
    ## ##

    ## split train val ##
    x_train, x_val = train_val_split(dataset=x, split_ratio=split_ratio)
    y_train, y_val = train_val_split(dataset=y, split_ratio=split_ratio)
    ## ##

    ## normalize ##
    x_train, x_val, x_test = normalize(x_train, x_val, x_test)
    ## ##

    ## average n rows ##

    ## ##

    info(x_train, x_val, x_test, y_train, y_val, y_test)
    print("Load Data Finish")
    print("="*80)
    return x_train, x_val, x_test, y_train, y_val, y_test

def train_val_split(dataset, split_ratio=0.8):
    num = len(dataset)
    split_arr = np.split(dataset, [int(num*split_ratio)], axis=0)

    # for i in split_arr:
    #     print(i.shape)

    return split_arr[0], split_arr[1]

def normalize(x_train, x_val, x_test):
    train_mean = x_train.mean()
    train_std = x_train.std()

    train_normalize = (x_train - train_mean) / train_std
    val_normalize = (x_val - train_mean) / train_std
    test_normalize = (x_test - train_mean) / train_std
    return train_normalize, val_normalize, test_normalize


def averaged_by_N_rows(a, n=8):
    shape = a.shape
    assert len(shape) == 2
    assert shape[0] % n == 0
    b = a.reshape(shape[0] // n, n, shape[1])
    mean_vec = b.mean(axis=1)
    return mean_vec

def info(x_train, x_val, x_test, y_train, y_val, y_test):
    print('x_train shape: ', x_train.shape)
    print('x_val shape: ', x_val.shape)
    print('x_test shape: ', x_test.shape)
    print('y_train shape: ', y_train.shape)
    print('y_val shape: ', y_val.shape)
    print('y_test shape: ', y_test.shape)

if __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    
