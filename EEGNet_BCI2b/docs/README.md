# EEGNet for BCI Competition IV-2b Dataset with LOSO Cross-Validation

This repository contains a clean implementation of EEGNet for the BCI Competition IV-2b dataset using Leave-One-Subject-Out (LOSO) cross-validation. The implementation is extracted from the [EEG-DCNet repository](https://github.com/Kanyooo/EEG-DCNet) and focuses specifically on EEGNet performance on BCI-2b dataset.

## Overview

This implementation reproduces the exact experimental setup used in the EEG-DCNet paper for EEGNet on BCI-2b dataset:

- **Model**: EEGNet (Lawhern et al., 2018)
- **Dataset**: BCI Competition IV-2b (9 subjects, 2 classes)
- **Evaluation**: LOSO cross-validation
- **Parameters**: Exactly matching those reported in EEG-DCNet paper

## Dataset Information

**BCI Competition IV-2b Dataset:**
- **Subjects**: 9 subjects
- **Classes**: 2 (Left Hand vs Right Hand motor imagery)  
- **Channels**: 3 EEG channels (C3, Cz, C4)
- **Sampling Rate**: 250 Hz
- **Time Window**: 1.5s to 6s after cue onset (4.5s, 1125 samples)
- **Sessions**: 2 sessions per subject (training and evaluation)

## Model Parameters

**EEGNet Configuration (matching EEG-DCNet paper):**
- F1 = 8 (number of temporal filters)
- D = 2 (depth multiplier)
- kernLength = 64 (temporal convolution length)
- dropout = 0.25

**Training Parameters:**
- Batch size: 64
- Epochs: 500
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Categorical crossentropy

## Expected Performance

Based on the EEG-DCNet paper, EEGNet achieves:
- **Accuracy**: 86.08%
- **Kappa Score**: 0.7213

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download BCI-2b Dataset

Download the BCI Competition IV-2b dataset from: 
http://bnci-horizon-2020.eu/database/data-sets

The dataset should contain files like:
- B01T.mat, B01E.mat (Subject 1 training and evaluation)
- B02T.mat, B02E.mat (Subject 2 training and evaluation)
- ...
- B09T.mat, B09E.mat (Subject 9 training and evaluation)

### 2. Run Training and Evaluation

```bash
python main_EEGNet_BCI2b_LOSO.py
```

The script will prompt you to enter the path to your BCI-2b dataset directory.

### 3. Results

Results will be saved in the `results_EEGNet_BCI2b_LOSO/` directory:
- Training logs and performance metrics
- Saved model weights
- Confusion matrices
- Performance plots

## File Structure

```
EEGNet_BCI2b/
├── main_EEGNet_BCI2b_LOSO.py    # Main training and evaluation script
├── models.py                     # EEGNet model implementation
├── preprocess.py                 # BCI-2b data loading and preprocessing
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Implementation Details

### Data Preprocessing
- **Standardization**: Channel-wise z-score normalization
- **Time Window**: 1.5s to 6s after cue onset (matching original paper)
- **Artifact Handling**: Only artifact-free trials are used
- **Data Shuffling**: Random shuffling with fixed seed for reproducibility

### LOSO Cross-Validation
- Each of the 9 subjects is used as test subject once
- Remaining 8 subjects are used for training
- Training data is further split into train/validation (80%/20%)
- Final performance is averaged across all 9 folds

### Model Architecture
The EEGNet implementation follows the original paper exactly:
1. **Temporal Convolution**: F1 filters of length kernLength
2. **Depthwise Convolution**: Spatial filtering with depth multiplier D
3. **Separable Convolution**: Feature extraction with F2 = F1 * D filters
4. **Classification**: Dense layer with softmax activation

## Reproducibility

This implementation ensures reproducibility by:
- Using fixed random seeds for TensorFlow and NumPy
- Exact parameter matching with the original EEG-DCNet experiments
- Identical preprocessing pipeline
- Same evaluation methodology (LOSO cross-validation)

## References

1. **EEGNet Original Paper:**
   - Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. Journal of neural engineering, 15(5), 056013.

2. **EEG-DCNet Paper (source of experimental setup):**
   - Altaheri, H., Muhammad, G., Alsulaiman, M., Amin, S. U., Altuwaijri, G. A., Abdul, W., ... & Faisal, M. (2022). Physics-informed attention temporal convolutional network for EEG-based motor imagery classification. IEEE Transactions on Industrial Informatics, 19(2), 2249-2258.

3. **BCI Competition IV-2b Dataset:**
   - http://bnci-horizon-2020.eu/database/data-sets

## License

This code is based on the EEG-DCNet repository which is licensed under Apache-2.0 License.

## Acknowledgments

This implementation is extracted and adapted from the [EEG-DCNet repository](https://github.com/Kanyooo/EEG-DCNet) by Hamdi Altaheri et al. All parameters and preprocessing steps are maintained exactly as described in their paper to ensure reproducible results.