# Data Preprocessing
### Dataset Description
- The datasets has 40 people (25 is HC, 15 is SZ)
- In each person's files(.csv)  
    - there are 74 colums (channels)
        - **basic information:** (the first 4 columns, will be eliminated)
            - *subject	/ trial / 	condition /	sample*
                - one subject has 100 trials.
                - one trials at most has 3 conditions
                    - that is, button_tone, playback_tone, button_alone
                    - trials that has only 1 or 2 conditions will be eliminated
                - one condition has 3072 samples
                - the trial which has three conditions has $3072 \times 3 = 9216$ samples
        - **EEG channels:** (65 channels)
            - *Fp1, AF7, AF3, F1, F3, F5, F7, FT7, FC5, FC3, FC1, C1, C3, C5, T7, TP7, CP5, CP3, CP1, P1, P3, P5, P7, P9, PO7, PO3, O1, Iz, Oz, POz, Pz, CPz, Fpz, Fp2, AF8, AF4, AFz, Fz, F2, F4, F6, F8, FT8, FC6, FC4, FC2, FCz, Cz, C2, C4, C6, T8, TP8, CP6, CP4, CP2, P2, P4, P6, P8, P10, PO8, PO4, O2, TP10*
        - **unkown channels:** (the last 6 colums, will be eliminated)
            - *VEOa, VEOb, HEOL, HEOR, Nose*
        <!-- montage 的圖 -->

### After preprocessing
the dataset will remains **EEG channels** (65 channels)
- each person's data has 4 versions (and each versions has two data structures with the same values) 
    - 4 versions:
        - Raw data (no filter, no ASR): use this as baseline when training deep learning model
        - filterd data (Raw data through high-pass filter)
        - ASR data (w/o high_pass_filter)
        - ASR data (w/ high_pass_filter): has best signal quality, so use this for training deep learning model
    - 2 data structures:
        - .fif (mne.RawArray)
        - npy (np.array)
    - each data shape
        - no matter which data stuctures, the shape: (num_channels, num_times)  
           - num_channels = 65 (only EEG channels)
           - num_times = 3\*3072\* num_trails (with three condition)
                - the trails which less than three conditions will be eliminated
        - **however, how do we cut out the all datas into training and testing is essential.**
        - **Data Split Method** 
            - In order to reasonably split the data into training and testing, each trail should be divided into one data
                - **the total number of datas of one person is num_trails**
            - the rate of training and testing is 95:5 (randomly spilt)
    - When training, we can use the data
        - **each data's shape can be (times_step=3*3072, channels=65)**
        - or you can swap (channels=65, times_step=3*3072)

- **The total number of data is** 
    - training:  3391
    - testing:   180

### Project Setup
- Environment
    ```bash
    $ conda create --name bci python=3.10.11
    $ conda activate bci
    $ conda install -c anaconda numpy
    $ conda install -c conda-forge scikit-learn
    $ conda install -c conda-forge matplotlib
    $ conda install -c anaconda pandas
    $ conda install -c anaconda pip
    $ conda install -c conda-forge gcc=12.1.0
    $ conda install -c anaconda jupyter
    $ pip install mne==1.3.1 -q
    $ pip install asrpy -q
    $ pip install autoreject -q
    $ pip install pyriemann -q
    ```
- Preprocessing.py
    - Self-defined variable
        ```python
        # in Preprocessing.py
        DATASET_ROOT = fr"/local/SSD/archive_schizophrenia/" # your dataset directory path
        SAVE_ROOT = fr"/local/SSD/bci_preprocess" # your result directory path
        ```
    - How to use
        ```bash
        $ python Preprocessing.py
        ```
- data_split.py
    - Self-defined variable
        ```python
        # in data_split.py
        dir_root = fr'/local/SSD/bci_preprocess/' # your dataset directory path after preprocessing
        save_root = fr'/local/SSD/bci_dataset_npy/' # your result directory path
        dataset_types = ['raw', 'asr', 'filter', 'filter_asr']
        dataset_type = dataset_types[0] # your dataset type version
        ```
    - How to use
        ```bash
        $ python data_split.py
        ```
### Methods for Preprocessing
- 1.0 Hz high-pass filter
    ```python
    # in Preprocessing.py
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
    ```
- Change data type from np.array to mne.io.RawArray for furthur preprocessing
- ICA
    - Find ICLABEL
        ```python
        # in Preprocessing.py
        def compute_ICLABEL(rawArray_eegData):
            print('-'*80)
            print("compute_ICLABEL")

            ica = mne.preprocessing.ICA(n_components=.99, random_state=38)
            ica.fit(rawArray_eegData)

            ica_labels = mne_icalabel.label_components(rawArray_eegData, ica, method="iclabel")
            print("Done")
            return ica, ica_labels
        ```
    - Reconstruct Data: reserve the labels which is "brain" or "other"
        ```python
        # in Preprocessing.py
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
        ```
### Methods for Data Split 
- total eeg signal shape for all subjects: [trail_num, timestamp, channel_nums] = [3571, 9216, 65]
- split to train set and test set for Deep Learning. 
    - $\text{Train}:\text{Test}=95:5$ for a person eeg signal
    ```python
    # in data_split.py
    # randomly spilt data
    X_train_HC, X_test_HC, y_train_HC, y_test_HC = train_test_split(data_HC, label_HC, test_size=0.05, random_state=21)
    X_train_SZ, X_test_SZ, y_train_SZ, y_test_SZ = train_test_split(data_SZ, label_SZ, test_size=0.05, random_state=7)
    ```