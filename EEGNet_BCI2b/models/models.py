"""
EEGNet model implementation for BCI-2b dataset
Based on EEG-DCNet repository: https://github.com/Kanyooo/EEG-DCNet

References:
- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). 
  EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. 
  Journal of neural engineering, 15(5), 056013.
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, AveragePooling2D
from keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization, Flatten 
from keras.layers import Input, Permute
from keras.constraints import max_norm
from keras import backend as K


def EEGNet_classifier(n_classes, Chans=3, Samples=1125, F1=8, D=2, kernLength=64, dropout_eeg=0.25):
    """
    EEGNet classifier model for BCI-2b dataset
    
    Parameters
    ----------
    n_classes : int
        Number of classes (2 for BCI-2b)
    Chans : int
        Number of EEG channels (3 for BCI-2b)
    Samples : int
        Number of time samples (1125 for BCI-2b, 4.5s * 250Hz)
    F1 : int
        Number of temporal filters
    D : int
        Depth multiplier
    kernLength : int
        Length of temporal convolution
    dropout_eeg : float
        Dropout rate
        
    Returns
    -------
    model : keras.Model
        Compiled EEGNet model
    """
    input1 = Input(shape=(1, Chans, Samples))   
    input2 = Permute((3, 2, 1))(input1) 
    regRate = 0.25

    eegnet = EEGNet(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    eegnet = Flatten()(eegnet)
    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(regRate))(eegnet)
    softmax = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


def EEGNet(input_layer, F1=8, kernLength=64, D=2, Chans=3, dropout=0.25):
    """
    EEGNet core model implementation
    
    This implementation follows the original EEGNet paper exactly as described in:
    Lawhern et al. (2018) EEGNet: a compact convolutional neural network for 
    EEG-based brain–computer interfaces.
    
    Parameters
    ----------
    input_layer : keras.Layer
        Input layer with shape (samples, channels, time)
    F1 : int
        Number of temporal filters (default: 8)
    kernLength : int
        Length of temporal convolution (default: 64)
    D : int
        Depth multiplier for depthwise convolution (default: 2)
    Chans : int
        Number of EEG channels (default: 3 for BCI-2b)
    dropout : float
        Dropout rate (default: 0.25)
        
    Returns
    -------
    block3 : keras.Layer
        Output layer of EEGNet
    """
    F2 = F1 * D
    
    # Block 1: Temporal convolution
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    
    # Block 2: Depthwise convolution
    block2 = DepthwiseConv2D((1, Chans), use_bias=False, 
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)

    # Block 3: Separable convolution
    block3 = SeparableConv2D(F2, (16, 1),
                             data_format='channels_last',
                             use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    
    return block3


def get_EEGNet_model(n_classes=2, n_channels=3, in_samples=1125):
    """
    Create and return EEGNet model with BCI-2b specific parameters
    
    Parameters
    ----------
    n_classes : int
        Number of classes (default: 2 for BCI-2b)
    n_channels : int  
        Number of EEG channels (default: 3 for BCI-2b)
    in_samples : int
        Number of time samples (default: 1125 for BCI-2b, 4.5s * 250Hz)
        
    Returns
    -------
    model : keras.Model
        Compiled EEGNet model ready for training
    """
    model = EEGNet_classifier(
        n_classes=n_classes, 
        Chans=n_channels, 
        Samples=in_samples,
        F1=8,           # As used in original EEG-DCNet experiments
        D=2,            # As used in original EEG-DCNet experiments  
        kernLength=64,  # As used in original EEG-DCNet experiments
        dropout_eeg=0.25 # As used in original EEG-DCNet experiments
    )
    
    return model