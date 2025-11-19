"""
BCI-2b dataset preprocessing module
Based on EEG-DCNet repository: https://github.com/Kanyooo/EEG-DCNet

This module provides functions to load and preprocess the BCI Competition IV-2b dataset
for LOSO (Leave One Subject Out) cross-validation.

Dataset information:
- 9 subjects
- 3 EEG channels (C3, Cz, C4)  
- 2 classes (left hand vs right hand motor imagery)
- Sampling rate: 250 Hz
- Time window: 1.5s to 6s after cue onset (4.5s, 1125 samples)
"""

import numpy as np
import scipy.io as sio
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def load_data_LOSO(data_path, subject, dataset='BCI2b'):
    """
    Load BCI-2b data for LOSO cross-validation
    
    Parameters
    ----------
    data_path : str
        Path to the BCI-2b dataset directory
    subject : int
        Subject index (0-8) to be used as test subject
    dataset : str
        Dataset name (default: 'BCI2b')
        
    Returns
    -------
    X_train : ndarray
        Training data (all subjects except test subject)
    y_train : ndarray
        Training labels
    X_test : ndarray
        Test data (test subject only)
    y_test : ndarray
        Test labels
    """
    X_train, y_train = None, None
    n = 9  # Number of subjects in BCI-2b
    
    for sub in range(0, n):
        path = data_path
        X1, y1 = load_BCI2b_data(path, sub+1, True)   # Training session
        X2, y2 = load_BCI2b_data(path, sub+1, False)  # Evaluation session
        
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
                   
        if (sub == subject):
            X_test = X
            y_test = y
        elif X_train is None:
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test


def load_BCI2b_data(data_path, subject, training):
    """
    Load BCI Competition IV-2b data for a specific subject
    
    Parameters
    ----------
    data_path : str
        Path to the BCI-2b dataset directory
    subject : int
        Subject number (1-9)
    training : bool
        True for training session, False for evaluation session
        
    Returns
    -------
    data_return : ndarray
        EEG data with shape (trials, channels, time_samples)
    class_return : ndarray
        Class labels (0 for left hand, 1 for right hand)
    """
    n_channels = 3
    n_tests = 120 + 120 + 160  # Total possible trials
    window_length = 8 * 250    # 8 seconds at 250 Hz

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_length))

    n_valid_trial = 0
    t1 = int(1.5 * 250)  # Start time_point (1.5s after cue)
    t2 = int(6 * 250)    # End time_point (6s after cue)
    
    if training:
        a = sio.loadmat(data_path + 'B0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'B0' + str(subject) + 'E.mat')
        
    a_data = a["data"]
    
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if a_artifacts[trial] == 0:  # Only use artifact-free trials
                data_return[n_valid_trial, :, :] = np.transpose(
                    a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_length), :n_channels]
                )
                class_return[n_valid_trial] = int(a_y[trial])
                n_valid_trial += 1
                
    class_return = (class_return - 1).astype(int)  # Convert to 0-indexed
    return data_return[0:n_valid_trial, :, t1:t2], class_return[0:n_valid_trial]


def standardize_data(X_train, X_test, channels): 
    """
    Standardize EEG data channel-wise
    
    Parameters
    ----------
    X_train : ndarray
        Training data
    X_test : ndarray
        Test data  
    channels : int
        Number of channels
        
    Returns
    -------
    X_train : ndarray
        Standardized training data
    X_test : ndarray
        Standardized test data
    """
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test


def get_data(path, subject, dataset='BCI2b', LOSO=False, isStandard=True, isShuffle=True):
    """
    Main function to load and preprocess BCI-2b data
    
    Parameters
    ----------
    path : str
        Path to the dataset
    subject : int
        Subject index
    dataset : str
        Dataset name (default: 'BCI2b')
    LOSO : bool
        Whether to use LOSO cross-validation (default: False)
    isStandard : bool
        Whether to standardize the data (default: True)
    isShuffle : bool
        Whether to shuffle the data (default: True)
        
    Returns
    -------
    X_train : ndarray
        Training data
    y_train : ndarray
        Training labels
    y_train_onehot : ndarray
        One-hot encoded training labels
    X_test : ndarray
        Test data
    y_test : ndarray
        Test labels
    y_test_onehot : ndarray
        One-hot encoded test labels
    """
    # Load and split the dataset
    if LOSO:
        # LOSO evaluation approach
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        # Subject-specific approach
        X_train, y_train = load_BCI2b_data(path, subject+1, True)
        X_test, y_test = load_BCI2b_data(path, subject+1, False)

    # Shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)
    
    # Prepare testing data 
    N_te, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_te, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)
    
    # Standardize the data
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


def get_BCI2b_dataset_info():
    """
    Get BCI-2b dataset configuration parameters
    
    Returns
    -------
    dataset_conf : dict
        Dataset configuration parameters
    """
    dataset_conf = {
        'name': 'BCI2b',
        'n_classes': 2,
        'n_channels': 3,
        'in_samples': 1125,  # 4.5s * 250Hz
        'n_sub': 9,
        'classes_labels': ['Left Hand', 'Right Hand'],
        'isStandard': True,
        'LOSO': True  # Enable LOSO cross-validation
    }
    
    return dataset_conf