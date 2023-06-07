import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
import mne_icalabel
import asrpy

DATASET_ROOT = fr"/local/SSD/archive_schizophrenia/"
SAVE_ROOT = fr"/local/SSD/bci_preprocess"

os.makedirs(SAVE_ROOT, exist_ok=True)

sfreq=1024
col_names = pd.read_csv(os.path.join(DATASET_ROOT, "columnLabels.csv"))
ch_names = list(col_names.columns[4:])
col_names = list(col_names)

def generate_RawArrayData(npdata, info):
    npEGG = np.swapaxes(npdata, 0, 1)
    EEGData = mne.io.RawArray(npEGG, info)
    EEGData.pick("eeg")
    EEGData = EEGData.set_eeg_reference("average")
    return EEGData

def plot_FFT(origin_rawArrayData, filtered_rawArrayData):
    # FFT
    fft_ori = np.abs(mne.time_frequency.stft(origin_rawArrayData, wsize=240))
    fft_fir = np.abs(mne.time_frequency.stft(filtered_rawArrayData, wsize=240))
    Fz_fft_ori_100 = fft_ori[0,:100,0]
    Fz_fft_fir_100 = fft_fir[0,:100,0]
    Fz_fft_fir_100 = np.abs(Fz_fft_fir_100)
    Fz_fft_ori_100 = np.abs(Fz_fft_ori_100)

    x = np.arange(100)
    plt.title('time frequency')
    plt.xlabel('frequcy(HZ)')
    plt.ylabel('amplitude')
    plt.plot(x,Fz_fft_ori_100)
    plt.plot(x,Fz_fft_fir_100)
    plt.legend(['origin', 'After FIR'])
    plt.show()

'''
high pass filter
input: npArr (np.npdarry; shape: (n_channel(70), n_times))
return: filtered_npEGG (np.npdarry; shape: same as input)
'''
def high_pass_filter(npArr):
    print('-'*80)
    print("high_pass_filter")

    ica_low_cut = 1.0
    hi_cut  = None

    npEEG = np.copy(npArr) # shape(n_channel, n_times) = (70, ...)
    filtered_npEGG = mne.filter.filter_data(data = npEEG, sfreq = sfreq, l_freq = ica_low_cut, h_freq = hi_cut, fir_design='firwin')
    # plot_FFT(npEGG, filtered_npEGG)

    print("filtered_npEGG shape: ", filtered_npEGG.shape)
    print("high_pass_filter Done")
    return filtered_npEGG


# reference: https://github.com/DiGyt/asrpy
def ASR_denoise(rawArray_eegData):
    print('-'*80)
    print("ASR_denoise")
    asr = asrpy.ASR(sfreq=rawArray_eegData.info["sfreq"], cutoff=20)
    asr.fit(rawArray_eegData)
    asr_eegData = asr.transform(rawArray_eegData)
    print("Done")
    return asr_eegData

# reference: https://blog.csdn.net/qq_44930039/article/details/127734713
def compute_ICLABEL(rawArray_eegData):
    print('-'*80)
    print("compute_ICLABEL")

    ica = mne.preprocessing.ICA(n_components=.99, random_state=38)
    ica.fit(rawArray_eegData)

    ica_labels = mne_icalabel.label_components(rawArray_eegData, ica, method="iclabel")
    print("Done")
    return ica, ica_labels

def reconstruct_EEGData(rawArray_eegData, ica, ic_labels):
    print('-'*80)
    print("reconstruct_EEGData")

    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    print("Exclude these ICA components: {exclude_idx}".format(exclude_idx=exclude_idx))

    reconstruct_eegData = rawArray_eegData.copy()
    ica.apply(reconstruct_eegData, exclude=exclude_idx)
    print("Done")
    return reconstruct_eegData


'''
read .csv with pandas, sorted, select trails
input: file_name, column_names, channel_names
return: npdarray (shape=(n_channel(70), n_times)),  info
'''
def loadingData(fname, col_names, ch_names):
    # basic info
    ch_type = ['eeg']*64 + ['eog']*4 + ['misc'] + ['eeg']
    event_id = dict(button_tone=1, playback_tone=2, button_alone=3)  # 3 conditions
    info = mne.create_info(ch_names, sfreq, ch_type)
    montage = mne.channels.make_standard_montage('standard_1005')
    info.set_montage(montage=montage, match_case=True)
    info['description'] = 'dataset from ' + fname

    # read csv and sorted
    df = pd.read_csv(os.path.join(DATASET_ROOT, fname, fname), header=None, names=col_names)
    df.sort_values(by=['trial', 'condition', 'sample'], inplace=True)  # sorted 
    
    condition_list = set(df.condition)
    trials_list = set(df.trial)

    # we have to select the trials because not every trial has three conditions
    # we just want the trail which has three conditions.
    X = []
    for _, trial_number in enumerate(trials_list):
        number_of_trial = len(df[df.trial == trial_number])
        if number_of_trial == 9216.0:
            current_sample_matrix = df[df.trial == trial_number][ch_names].values
            X.append(current_sample_matrix)

    npEGG_full_condition = np.asarray(X)
    npEGG_full_condition = npEGG_full_condition.reshape(npEGG_full_condition.shape[0]*npEGG_full_condition.shape[1], -1)
    
    # swap axes  from (n_times, n_channel(70)) to (n_channel(70), n_times)
    npEGG_full_condition = np.swapaxes(npEGG_full_condition, 0, 1)    
    print("the shape of ", fname, "'s data: ", npEGG_full_condition.shape)

    return npEGG_full_condition, info


def npArray2RawArray(npArr, info):
    # change data type from npdarry to mne.RqwArray
    # remain only "eeg" channels (65 channels left)
    rawArr_EGGdata = mne.io.RawArray(npArr, info)
    rawArr_EGGdata = rawArr_EGGdata.pick(picks=["eeg"])
    rawArr_EGGdata = rawArr_EGGdata.set_eeg_reference("average")

    return rawArr_EGGdata

    
if __name__ == "__main__":
    print("run program file: {program_fname}".format(program_fname = os.path.basename(__file__)))
    start_time = time.time()

    for dir_name in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, dir_name)):
            print("="*80)
            print(dir_name)
            
            # make save DIR 
            save_path = os.path.join(SAVE_ROOT, dir_name)
            os.makedirs(save_path, exist_ok=True)

            # load .csv
            npArr_EEGdata, info = loadingData(fname=dir_name, col_names=col_names, ch_names=ch_names) #70 channels
        
            # Raw data (no filter, no ASR)
            raw_EEGdata = npArray2RawArray(npArr=npArr_EEGdata, info=info) # 65 "eeg" channels left
            raw_ica, raw_ic_labels = compute_ICLABEL(raw_EEGdata)
            raw_ica_reconstruct = reconstruct_EEGData(raw_EEGdata, raw_ica, raw_ic_labels)
            raw_reconstruct_ndarray, _ = raw_ica_reconstruct[:]
            # store 
            raw_npy_save_path = os.path.join(save_path, "raw_{fname}.npy".format(fname=dir_name))
            np.save(raw_npy_save_path, raw_reconstruct_ndarray)
            raw_ica_reconstruct.save(os.path.join(save_path, "raw_{fname}.fif".format(fname=dir_name)))


            # filterd data
            filter_npArr_EEGdata = high_pass_filter(npArr_EEGdata)
            filter_raw_EEGdata = npArray2RawArray(npArr=filter_npArr_EEGdata, info=info) # 65 "eeg" channels left
            filter_ica, filter_ic_labels = compute_ICLABEL(filter_raw_EEGdata)  
            filter_ica_reconstruct = reconstruct_EEGData(filter_raw_EEGdata, filter_ica, filter_ic_labels)
            filter_reconstruct_ndarray, _ = filter_ica_reconstruct[:]
            # store 
            filter_npy_save_path = os.path.join(save_path, "filter_{fname}.npy".format(fname=dir_name))
            np.save(filter_npy_save_path, filter_reconstruct_ndarray)
            filter_ica_reconstruct.save(os.path.join(save_path, "filter__{fname}.fif".format(fname=dir_name)))


            # ASR data (w/o high_pass_filter)
            asr_EEGData = ASR_denoise(raw_EEGdata)
            asr_ica, asr_ic_labels = compute_ICLABEL(asr_EEGData)
            asr_reconstruct = reconstruct_EEGData(asr_EEGData, asr_ica, asr_ic_labels)
            asr_reconstruct_ndarray, _ = asr_reconstruct[:]
            # store 
            asr_npy_save_path = os.path.join(save_path, "asr_{fname}.npy".format(fname=dir_name))
            np.save(asr_npy_save_path, asr_reconstruct_ndarray)
            asr_reconstruct.save(os.path.join(save_path, "asr_{fname}.fif".format(fname=dir_name)))


            # ASR data (with high_pass_filter)
            filter_asr_EEGData = ASR_denoise(filter_raw_EEGdata)
            filter_asr_ica, filter_asr_ic_labels = compute_ICLABEL(filter_asr_EEGData)
            filter_asr_reconstruct = reconstruct_EEGData(asr_EEGData, filter_asr_ica, filter_asr_ic_labels)
            filter_asr_reconstruct_ndarray, _ = filter_asr_reconstruct[:]
            # store 
            filter_asr_npy_save_path = os.path.join(save_path, "filter_asr_{fname}.npy".format(fname=dir_name))
            np.save(filter_asr_npy_save_path, filter_asr_reconstruct_ndarray)
            filter_asr_reconstruct.save(os.path.join(save_path, "filter_asr_{fname}.fif".format(fname=dir_name)))    


            # store the ic label
            print("="*80)
            ic_label_txt_save_path = os.path.join(save_path, "{fname}_iclabel.txt".format(fname=dir_name))
            with open(ic_label_txt_save_path, 'w') as f:
                f.write("raw: {raw_iclabel}".format(raw_iclabel=raw_ic_labels))
                f.write("highpass-filter: {filter_iclabel}".format(filter_iclabel=filter_ic_labels))
                f.write("asr (w/o high_pass_filter): {asr_iclabel}".format(asr_iclabel=asr_ic_labels))
                f.write("asr (with high_pass_filter): {asr_iclabel}".format(asr_iclabel=filter_asr_ic_labels))

    end_time = time.time()

    print("program file: {program_fname} is finished".format(program_fname = os.path.basename(__file__)))
    print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "mins")