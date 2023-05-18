# BCI_Schizophrenia
NTHU 11120ISA557300 Brain Computer Interfaces: Fundamentals and Application Final Group Project
- Project: Schizophrenia Disorder Detection
- Teammate
    - 108062224 廖珞珽
    - 111062574 徐瑞憫
    - 111062575 許詠晴
- Setup
    - data preprosessing
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
- Dataset Source
    - Name: EEG data from basic sensory task in Schizophrenia ([link](https://www.kaggle.com/datasets/broach/button-tone-sz))
    - Paper: Did I Do That? Abnormal Predictive Processes in Schizophrenia When Button Pressing to Deliver a Tone
    - [Project Details](https://reporter.nih.gov/project-details/9187052)s
