# BCI_Schizophrenia
NTHU 11120ISA557300 Brain Computer Interfaces: Fundamentals and Application Final Group Project
- Project: Schizophrenia Disorder Detection
- Teammate
    - 108062224 廖珞珽
    - 111062574 徐瑞憫
    - 111062575 許詠晴
### Dataset
- Dataset Source
    - Name: EEG data from basic sensory task in Schizophrenia ([link](https://www.kaggle.com/datasets/broach/button-tone-sz))
    - Paper: Did I Do That? Abnormal Predictive Processes in Schizophrenia When Button Pressing to Deliver a Tone
    - [Project Details](https://reporter.nih.gov/project-details/9187052)
    - [Kaggle Competition](https://www.kaggle.com/datasets/broach/button-tone-sz)
- Dataset Description (there are two datasets)
    - *ERPdataset*
        - The ERP dataset has 81 people (32 is Healthy controls, 49 is Schizophrenia)
        - The datasets is provided by [Kaggle Competition](https://www.kaggle.com/datasets/broach/button-tone-sz) and is pre-processed.
            - the pre-processing detail is descripted in the Data Card of the kaggle website.   
        - 9 channels
            - Fz, FCz, Cz, FC3, FC4, C3, C4, CP3, CP4
    - *raw datasets*
        - The raw datasets (without any preprocessing) has 40 people (25 is HC, 15 is SZ)
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
### Project enviroment Setup
- Data Preprosessing
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
- Deep Learning
    - reference: [install tensorflow](https://www.tensorflow.org/install/pip#software_requirements)
    ``` bash
    $ conda create --name bci_tf python=3.8
    $ conda activate bci_tf
    $ conda install -c anaconda tensorflow-gpu
    $ conda install -c anaconda pip
    $ conda install -c anaconda numpy==1.18.5
    $ conda install -c conda-forge matplotlib
    $ conda install -c conda-forge scikit-learn
    ```
- Project Description: click to see the README of these parts
    - [Preprocessing](/Preprocessing/README.md)
    - [MLWithERPdata](/MLWithERPdata/README.md)
    - [Deep Learning](/DeepLearning/README.md)
    
### More detailed content of our entire project can be viewed in ppt and mp4
