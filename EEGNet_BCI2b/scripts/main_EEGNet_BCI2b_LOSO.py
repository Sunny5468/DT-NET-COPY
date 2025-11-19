"""
EEGNet training script for BCI-2b dataset with LOSO cross-validation
Based on EEG-DCNet repository: https://github.com/Kanyooo/EEG-DCNet

This script implements exactly the same training methodology used in the original
EEG-DCNet paper for EEGNet on BCI Competition IV-2b dataset with LOSO cross-validation.

Parameters match those reported in the paper:
- EEGNet with F1=8, D=2, kernLength=64, dropout=0.25
- LOSO cross-validation (9 subjects)
- Training parameters: batch_size=64, epochs=500, lr=0.001
- Data preprocessing: standardization, 1.5-6s time window
"""

import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Use keras directly for better compatibility
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, Callback

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

# 添加父目录到路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.models import get_EEGNet_model
from utils.preprocess import get_data, get_BCI2b_dataset_info


def draw_learning_curves(history, sub, results_path, stopped_epoch=None, best_epoch=None):
    """绘制训练/验证曲线并可视化早停信息

    Parameters
    ----------
    history : keras.callbacks.History
        训练历史对象
    sub : int
        subject 编号（1-based）
    results_path : str
        保存图像的结果目录
    stopped_epoch : int or None
        EarlyStopping 的停止 epoch（0-based），如果没有触发早停则为 None
    best_epoch : int or None
        最佳 validation loss 对应的 epoch（0-based）
    """
    train_acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(1, len(train_acc)+1)

    plt.figure(figsize=(12, 4))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    if best_epoch is not None:
        plt.axvline(best_epoch+1, color='green', linestyle='--', linewidth=1.2, label='Best Val Loss Epoch')
    if stopped_epoch is not None and stopped_epoch+1 < len(train_acc):
        plt.axvline(stopped_epoch+1, color='red', linestyle='--', linewidth=1.2, label='Early Stop Epoch')
    plt.title(f'Subject {sub} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    if best_epoch is not None:
        plt.axvline(best_epoch+1, color='green', linestyle='--', linewidth=1.2, label='Best Val Loss Epoch')
    if stopped_epoch is not None and stopped_epoch+1 < len(train_loss):
        plt.axvline(stopped_epoch+1, color='red', linestyle='--', linewidth=1.2, label='Early Stop Epoch')
    plt.title(f'Subject {sub} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.suptitle(f'Subject {sub} Training Curves')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(results_path, f'subject_{sub}_learning_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.close()


def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    """Generate confusion matrix plot"""
    display_labels = classes_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, 
                                  display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix of Subject: ' + str(sub))
    plt.savefig(results_path + '/subject_' + str(sub) + '.png')
    plt.show()


def draw_performance_barChart(num_sub, metric, label):
    """Draw performance bar chart for all subjects"""
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model ' + label + ' per subject')
    ax.set_ylim([0, 1])
    plt.show()


def train(dataset_conf, train_conf, results_path):
    """
    Train EEGNet model using LOSO cross-validation
    
    Parameters
    ----------
    dataset_conf : dict
        Dataset configuration parameters
    train_conf : dict
        Training configuration parameters
    results_path : str
        Path to save results
    """
    # Remove the 'result' folder before training
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)        

    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    
    # Create files to store results
    best_models = open(results_path + "/best_models.txt", "w")
    log_write = open(results_path + "/log.txt", "w")
    
    # Get dataset parameters
    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    
    # Get training hyperparameters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    n_train = train_conf.get('n_train')
    from_logits = train_conf.get('from_logits')
    live_epoch_logging = train_conf.get('live_epoch_logging', True)

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))
    
    # Iteration over subjects for LOSO cross-validation
    for sub in range(n_sub):
        print(f'\nTraining on subject {sub+1} (LOSO: test subject {sub+1})')
        log_write.write(f'\nTraining on subject {sub+1} (LOSO: test subject {sub+1})\n')
        
        # Variables to save the best subject accuracy among multiple runs
        BestSubjAcc = 0 
        bestTrainingHistory = []
        bestStoppedEpoch = None
        bestValBestEpoch = None
        
        # Get training and test data using LOSO
        X_train, _, y_train_onehot, _, _, _ = get_data(
            data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)
        
        # Divide the training data into training and validation
        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
            X_train, y_train_onehot, test_size=0.2, random_state=42)       
        
        # Iteration over multiple runs 
        for train_run in range(n_train):
            # Set random seeds for reproducibility
            tf.random.set_seed(train_run+1)
            np.random.seed(train_run+1)
            
            # Get the current 'IN' time to calculate the 'run' training time
            in_run = time.time()
            
            # Create folders and files to save trained models for all runs
            # 统一权重文件命名: subject-{sub+1}.weights.h5
            run_dir = os.path.join(results_path, 'saved_models', f'run-{train_run+1}')
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            filepath = os.path.join(run_dir, f'subject-{sub+1}.weights.h5')
            
            # Create the EEGNet model
            model = get_EEGNet_model(
                n_classes=dataset_conf.get('n_classes'),
                n_channels=dataset_conf.get('n_channels'),
                in_samples=dataset_conf.get('in_samples')
            )
            
            # Compile the model
            model.compile(
                loss=CategoricalCrossentropy(from_logits=from_logits),
                optimizer=Adam(learning_rate=lr), 
                metrics=['accuracy']
            )          

            # Define CSV log file per subject-run
            csv_log_path = os.path.join(run_dir, f'subject-{sub+1}_training_log.csv')
            csv_logger = CSVLogger(csv_log_path, append=False)

            # 实时 epoch 输出回调
            class EpochLogger(Callback):
                def __init__(self, log_fh, subject_idx, seed_idx):
                    super().__init__()
                    self.log_fh = log_fh
                    self.subject_idx = subject_idx
                    self.seed_idx = seed_idx
                    self.start_time = None
                def on_train_begin(self, logs=None):
                    self.start_time = time.time()
                def on_epoch_end(self, epoch, logs=None):
                    if logs is None: return
                    msg = (f"[Sub {self.subject_idx} Seed {self.seed_idx}] Epoch {epoch+1:03d} "
                           f"loss={logs.get('loss'):.4f} val_loss={logs.get('val_loss'):.4f} "
                           f"acc={logs.get('accuracy'):.4f} val_acc={logs.get('val_accuracy'):.4f}")
                    print(msg)
                    self.log_fh.write(msg + '\n')
                def on_train_end(self, logs=None):
                    total = time.time() - self.start_time
                    self.log_fh.write(f"[Sub {self.subject_idx} Seed {self.seed_idx}] Train finished in {total/60:.2f} min\n")

            early_stop_cb = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True,
                                          verbose=0)
            callbacks = [
                ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                save_best_only=True, save_weights_only=True, mode='min'),
                ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20,
                                  verbose=0, min_lr=0.0001),
                csv_logger,
                early_stop_cb
            ]
            if live_epoch_logging:
                callbacks.append(EpochLogger(log_write, sub+1, train_run+1))
            
            # Train the model
            history = model.fit(
                X_train, y_train_onehot, 
                validation_data=(X_val, y_val_onehot), 
                epochs=epochs, 
                batch_size=batch_size,
                callbacks=callbacks, 
                verbose=0
            )
           
            # Load the best weights and evaluate
            model.load_weights(filepath)
            y_pred = model.predict(X_val)

            if from_logits:
                y_pred = tf.nn.softmax(y_pred).numpy().argmax(axis=-1)
            else:
                y_pred = y_pred.argmax(axis=-1)
                
            labels = y_val_onehot.argmax(axis=-1)
            acc[sub, train_run] = accuracy_score(labels, y_pred)
            kappa[sub, train_run] = cohen_kappa_score(labels, y_pred)
                        
            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            
            # Print & write performance measures for each run
            info = f'Subject: {sub+1}   seed {train_run+1}   time: {(out_run-in_run)/60:.1f} m   '
            info = info + f'valid_acc: {acc[sub, train_run]:.4f}   valid_loss: {min(history.history["val_loss"]):.3f}'
            print(info)
            log_write.write(info +'\n')
            
            # If current training run is better than previous runs, save the history
            if(BestSubjAcc < acc[sub, train_run]):
                BestSubjAcc = acc[sub, train_run]
                bestTrainingHistory = history
                # 记录早停停止 epoch（若未触发早停，stopped_epoch 可能为 0 或 = epochs-1）
                stopped_epoch = getattr(early_stop_cb, 'stopped_epoch', None)
                bestStoppedEpoch = stopped_epoch if stopped_epoch != 0 else None
                # 找到最优 val_loss 的 epoch
                if 'val_loss' in history.history:
                    bestValBestEpoch = int(np.argmin(history.history['val_loss']))
        
        # Store the path of the best model among several runs (写入完整路径 + 验证最佳准确率)
        best_run = np.argmax(acc[sub,:])
        best_model_path = os.path.join(results_path, 'saved_models', f'run-{best_run+1}', f'subject-{sub+1}.weights.h5')
        best_models.write(f'{best_model_path}\tbest_valid_acc={acc[sub, best_run]:.4f}\n')

        # Plot Learning curves 
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory, sub+1, results_path,
                                 stopped_epoch=bestStoppedEpoch,
                                 best_epoch=bestValBestEpoch)
          
    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
           
    # Print & write the validation performance using all seeds
    head1 = head2 = '         '
    for sub in range(n_sub): 
        head1 = head1 + f'sub_{sub+1}   '
        head2 = head2 + '-----   '
    head1 = head1 + '  average'
    head2 = head2 + '  -------'
    
    info = '\n---------------------------------\nValidation performance (acc %):'
    info = info + '\n---------------------------------\n' + head1 +'\n'+ head2
    
    for run in range(n_train): 
        info = info + f'\nSeed {run+1}:  '
        for sub in range(n_sub): 
            info = info + f'{acc[sub, run]*100:.2f}   '
        info = info + f'  {np.average(acc[:, run])*100:.2f}   '
        
    info = info + '\n---------------------------------\nAverage acc - all seeds: '
    info = info + f'{np.average(acc)*100:.2f} %\n\nTrain Time  - all seeds: {(out_exp-in_exp)/60:.1f}'
    info = info + ' min\n---------------------------------\n'
    print(info)
    log_write.write(info+'\n')

    # Close open files 
    best_models.close()   
    log_write.close() 


def test(dataset_conf, results_path):
    """
    Test the trained models using LOSO cross-validation
    
    Parameters
    ----------
    dataset_conf : dict
        Dataset configuration parameters
    results_path : str
        Path to saved models
    """
    # Open the "Log" file to write the evaluation results 
    log_write = open(results_path + "/log.txt", "a")
    
    # Get dataset parameters
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    classes_labels = dataset_conf.get('classes_labels')
     
    # Test the performance based on several runs (seeds)
    runs = sorted(os.listdir(results_path+"/saved_models"))  # 保证加载顺序稳定
    
    # Initialize variables
    acc = np.zeros((n_sub, len(runs)))
    kappa = np.zeros((n_sub, len(runs)))
    cf_matrix = np.zeros([n_sub, len(runs), n_classes, n_classes])

    all_inference_times = []  # 收集所有 (subject, seed) 的单 trial 推理时间
    
    # Iteration over subjects 
    for sub in range(n_sub):
        print(f'\nTesting subject {sub+1} (LOSO)')
        
        # Load test data (the subject that was left out during training)
        _, _, _, X_test, _, y_test_onehot = get_data(
            data_path, sub, dataset, LOSO=LOSO, isStandard=isStandard)     

        # Create model
        model = get_EEGNet_model(
            n_classes=dataset_conf.get('n_classes'),
            n_channels=dataset_conf.get('n_channels'),
            in_samples=dataset_conf.get('in_samples')
        )

        # Iteration over runs (seeds) 
        for seed in range(len(runs)): 
            # Load the model of the seed (统一权重文件名)
            model.load_weights(f'{results_path}/saved_models/{runs[seed]}/subject-{sub+1}.weights.h5')

            # Measure inference time per trial
            start = time.time()
            y_logits = model.predict(X_test)
            single_trial_time = (time.time() - start)/X_test.shape[0]
            all_inference_times.append(single_trial_time)

            # 预测一致性（当前网络输出已是 softmax 概率，若 from_logits=True 则再 softmax）
            from_logits = False  # 模型构建阶段以 softmax 结尾，可根据需要改成 dataset_conf 或传参
            if from_logits:
                y_probs = tf.nn.softmax(y_logits).numpy()
            else:
                y_probs = y_logits
            y_pred = y_probs.argmax(axis=-1)
            
            # Calculate accuracy and K-score          
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, seed] = accuracy_score(labels, y_pred)
            kappa[sub, seed] = cohen_kappa_score(labels, y_pred)
            
            # Calculate confusion matrix
            cf_matrix[sub, seed, :, :] = confusion_matrix(labels, y_pred, normalize='true')
        
    # Print & write the average performance measures for all subjects     
    head1 = head2 = '         '
    for sub in range(n_sub): 
        head1 = head1 + f'sub_{sub+1}   '
        head2 = head2 + '-----   '
    head1 = head1 + '  average'
    head2 = head2 + '  -------'
    
    info = '\n---------------------------------\nTest performance (acc & k-score):\n'
    info = info + '---------------------------------\n' + head1 +'\n'+ head2
    
    for run in range(len(runs)): 
        info = info + f'\nSeed {run+1}: '
        info_acc = '(acc %)   '
        info_k = '        (k-sco)   '
        for sub in range(n_sub): 
            info_acc = info_acc + f'{acc[sub, run]*100:.2f}   '
            info_k = info_k + f'{kappa[sub, run]:.3f}   '
        info_acc = info_acc + f'  {np.average(acc[:, run])*100:.2f}   '
        info_k = info_k + f'  {np.average(kappa[:, run]):.3f}   '
        info = info + info_acc + '\n' + info_k
        
    avg_infer = np.mean(all_inference_times) if all_inference_times else 0.0
    info = info + f'\n----------------------------------\nAverage - all seeds (acc %): '
    info = info + f'{np.average(acc)*100:.2f}\n                       (k-sco): '
    info = info + f'{np.average(kappa):.3f}\n\nAverage inference time: {avg_infer * 1000:.2f}'
    info = info + ' ms per trial (averaged over subjects & seeds)\n----------------------------------\n'
    print(info)
    log_write.write(info+'\n')
         
    # Draw performance bar charts for all subjects 
    draw_performance_barChart(n_sub, acc.mean(1), 'Accuracy')
    draw_performance_barChart(n_sub, kappa.mean(1), 'k-score')
    
    # Draw confusion matrix for all subjects (average)
    draw_confusion_matrix(cf_matrix.mean((0,1)), 'All', results_path, classes_labels)
    
    # Close opened file    
    log_write.close() 


def run():
    """Main function to run EEGNet training and testing on BCI-2b with LOSO"""
    
    # Get BCI-2b dataset configuration
    dataset_conf = get_BCI2b_dataset_info()
    
    # Set data path (user needs to modify this path)
    data_path = "C:/Users/35696/Desktop/BCI_2b"
    if not data_path.endswith('/'):
        data_path += '/'
    dataset_conf['data_path'] = data_path
    
    # Create a folder to store the results
    results_path = os.getcwd() + "/results_EEGNet_BCI2b_LOSO"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # Set training hyperparameters (matching EEG-DCNet paper)
    train_conf = {
        'batch_size': 64, 
        'epochs': 200, 
        'patience': 10, 
        'lr': 0.001,
        'n_train': 1,  # Number of training runs per subject
        'LearnCurves': True, 
        'from_logits': False,
        'live_epoch_logging': True  # 实时打印每个 epoch 指标
    }
    
    print("="*60)
    print("EEGNet Training on BCI Competition IV-2b Dataset")
    print("LOSO Cross-Validation (Leave One Subject Out)")
    print("="*60)
    print(f"Dataset: {dataset_conf['name']}")
    print(f"Subjects: {dataset_conf['n_sub']}")
    print(f"Classes: {dataset_conf['n_classes']} ({', '.join(dataset_conf['classes_labels'])})")
    print(f"Channels: {dataset_conf['n_channels']}")
    print(f"Samples: {dataset_conf['in_samples']} ({dataset_conf['in_samples']/250:.1f}s)")
    print(f"LOSO: {dataset_conf['LOSO']}")
    print("="*60)
    
    # Train the model
    print("Starting training...")
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model
    print("Starting evaluation...")
    test(dataset_conf, results_path)
    
    print("Training and evaluation completed!")
    print(f"Results saved in: {results_path}")


if __name__ == "__main__":
    run()