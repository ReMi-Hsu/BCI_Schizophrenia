"""
This file is to split dataset into training and testing.
the total number of data is 3571 (training: 3391, testing: 180)
each shape of data is (time_step=9216, num_channels=65)
"""
import os
import sys
import mne
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dir_root = fr'/local/SSD/bci_preprocess/'
save_root = fr'/local/SSD/bci_dataset_npy/'
data = list()

df = pd.read_csv(fr'/local/SSD/archive_schizophrenia/demographic.csv')
label_list = df[df.columns[1]].tolist()
subject_list = list(range(1,82))
label_dict = dict(zip(subject_list, label_list))


person_dirs = os.listdir(dir_root)
data_HC = np.empty((1, 9216, 65))
data_SZ = np.empty((1, 9216, 65))
label_HC = list()
label_SZ = list()


for person_dir in person_dirs:
    person_idx = person_dir.split('.')[0]
    print('name_idx = ', person_idx)
    
    fname = os.path.join(dir_root, person_dir, "raw_{name}.npy".format(name=person_dir))
    npArr_EEGdata = np.load(fname)
    print("the shape of npArr_EEGdata: ", npArr_EEGdata.shape) #(num_channels=65, num_times=9216*num_trail)
    
    npArr_EEGdata_swap = np.swapaxes(npArr_EEGdata, 0, 1)
    print("the shape of npArr_EEGdata_swap: ", npArr_EEGdata_swap.shape)  #(num_times=9216*num_trail, num_channels=65)
    
    npArr_trials = npArr_EEGdata.reshape((-1, 9216, npArr_EEGdata_swap.shape[1]))
    print("the shape of npArr_trials: ", npArr_trials.shape) #(num_trail, 9216, 65)
    
    if (label_dict[int(person_idx)] == 0):  #HC
        tmp = [label_dict[int(person_idx)]] * npArr_trials.shape[0]
        label_HC.extend(tmp)
        data_HC = np.vstack((data_HC, npArr_trials))
    else:  #SZ
        tmp = [label_dict[int(person_idx)]] * npArr_trials.shape[0]
        label_SZ.extend(tmp)
        data_SZ = np.vstack((data_SZ, npArr_trials))
    
    
    print('-'*20)
    
data_HC = np.asarray(data_HC) 
data_HC = data_HC[1:,:,:] 
data_SZ = np.asarray(data_SZ) 
data_SZ = data_SZ[1:,:,:] 

print('-'*30)
 
print('total HC data shape: ', data_HC.shape)
print('length of HC label: ', len(label_HC))
print('total SZ data shape: ', data_SZ.shape)
print('length of SZ label: ', len(label_SZ))

label_SZ = np.asarray(label_SZ)
label_HC = np.asarray(label_HC)


# randomly spilt data
X_train_HC, X_test_HC, y_train_HC, y_test_HC = train_test_split(data_HC, label_HC, test_size=0.05, random_state=21)
X_train_SZ, X_test_SZ, y_train_SZ, y_test_SZ = train_test_split(data_SZ, label_SZ, test_size=0.05, random_state=7)

X_train = np.vstack((X_train_HC, X_train_SZ))
X_test = np.vstack((X_test_HC, X_test_SZ))
y_train = np.append(y_train_HC, y_train_SZ)
y_test = np.append(y_test_HC, y_test_SZ)

# shuffle
index_train = np.array(range(0, X_train.shape[0]))
np.random.shuffle(index_train)
index_test = np.array(range(0, X_test.shape[0]))
np.random.shuffle(index_test)

X_train = X_train[index_train]
X_test = X_test[index_test]
y_train = y_train[index_train]
y_test = y_test[index_test]

print('-'*30)
print('y_test: ', y_test)
print('-'*30)

print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('y_train shape: ', y_train.shape)
print('y_test shape: ', y_test.shape)

np.save(os.path.join(save_root, 'raw_X_train.npy'), X_train)
np.save(os.path.join(save_root, 'raw_X_test.npy'), X_test)
np.save(os.path.join(save_root, 'raw_y_train.npy'), y_train)
np.save(os.path.join(save_root, 'raw_y_test.npy'), y_test)
