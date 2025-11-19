"""
DANN-EEGNet 训练脚本 - 用于 BCI-2b 数据集的跨被试迁移学习
使用域对抗神经网络 (Domain Adversarial Neural Network) 实现 LOSO 交叉验证

主要特性:
1. 源域（8个被试）训练标签分类器和域分类器
2. 通过梯度反转学习域不变特征
3. 目标域（1个被试）用于测试泛化性能
4. Lambda 参数动态调度，控制域对抗强度
"""

import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

# 添加父目录到路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.models_dann import get_dann_model, DomainAdaptationSchedule, GradientReversalLayer
from utils.preprocess import get_data, get_BCI2b_dataset_info


def prepare_dann_data(data_path, target_subject, dataset='BCI2b', isStandard=True):
    """
    为 DANN 准备源域和目标域数据
    
    Parameters
    ----------
    data_path : str
        数据集路径
    target_subject : int
        目标被试索引 (0-8)
    dataset : str
        数据集名称
    isStandard : bool
        是否标准化
        
    Returns
    -------
    X_source : ndarray
        源域数据（其他8个被试）
    y_source : ndarray
        源域标签
    domain_source : ndarray
        源域域标签（0-7，表示来自哪个源域被试）
    X_target : ndarray
        目标域数据（测试被试）
    y_target : ndarray
        目标域标签
    """
    from utils.preprocess import load_data_LOSO
    
    # 使用 LOSO 加载数据
    X_source, y_source, X_target, y_target = load_data_LOSO(
        data_path, target_subject, dataset)
    
    # 为源域数据生成域标签
    # 需要知道每个样本来自哪个源域被试
    from utils.preprocess import load_BCI2b_data, shuffle as sk_shuffle
    
    X_source_list = []
    y_source_list = []
    domain_source_list = []
    
    n_subjects = 9
    domain_idx = 0
    
    for sub in range(n_subjects):
        if sub == target_subject:
            continue  # 跳过目标被试
        
        # 加载该源域被试的数据
        X1, y1 = load_BCI2b_data(data_path, sub+1, True)   # 训练会话
        X2, y2 = load_BCI2b_data(data_path, sub+1, False)  # 评估会话
        
        X_sub = np.concatenate((X1, X2), axis=0)
        y_sub = np.concatenate((y1, y2), axis=0)
        
        # 为该被试的所有样本分配域标签
        domain_labels = np.full(len(X_sub), domain_idx, dtype=int)
        
        X_source_list.append(X_sub)
        y_source_list.append(y_sub)
        domain_source_list.append(domain_labels)
        
        domain_idx += 1
    
    # 合并所有源域数据
    X_source = np.concatenate(X_source_list, axis=0)
    y_source = np.concatenate(y_source_list, axis=0)
    domain_source = np.concatenate(domain_source_list, axis=0)
    
    # Shuffle 源域数据
    from sklearn.utils import shuffle
    X_source, y_source, domain_source = shuffle(
        X_source, y_source, domain_source, random_state=42)
    
    # Reshape 数据
    N_source, N_ch, T = X_source.shape
    X_source = X_source.reshape(N_source, 1, N_ch, T)
    
    N_target, N_ch, T = X_target.shape
    X_target = X_target.reshape(N_target, 1, N_ch, T)
    
    # 标准化
    if isStandard:
        from utils.preprocess import standardize_data
        X_source, X_target = standardize_data(X_source, X_target, N_ch)
    
    # One-hot 编码
    y_source_onehot = to_categorical(y_source)
    y_target_onehot = to_categorical(y_target)
    domain_source_onehot = to_categorical(domain_source, num_classes=8)
    
    return X_source, y_source_onehot, domain_source_onehot, X_target, y_target_onehot


class DANNTrainingCallback(Callback):
    """
    DANN 训练回调
    动态更新梯度反转层的 lambda 参数
    """
    def __init__(self, grl_layer, lambda_schedule, log_file=None):
        super().__init__()
        self.grl_layer = grl_layer
        self.lambda_schedule = lambda_schedule
        self.log_file = log_file
        self.epoch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        # 更新 lambda
        new_lambda = self.lambda_schedule.get_lambda(epoch)
        self.grl_layer.lambda_factor = new_lambda
        
        if self.log_file:
            msg = f"Epoch {epoch+1}: lambda = {new_lambda:.4f}"
            print(msg)
            self.log_file.write(msg + '\n')
    
    def on_epoch_end(self, epoch, logs=None):
        if logs and self.log_file:
            msg = (f"Epoch {epoch+1}: "
                  f"label_loss={logs.get('label_output_loss', 0):.4f}, "
                  f"domain_loss={logs.get('domain_output_loss', 0):.4f}, "
                  f"label_acc={logs.get('label_output_accuracy', 0):.4f}, "
                  f"domain_acc={logs.get('domain_output_accuracy', 0):.4f}")
            print(msg)
            self.log_file.write(msg + '\n')


def train_dann(dataset_conf, train_conf, results_path):
    """
    使用 DANN 训练 EEGNet 模型
    
    Parameters
    ----------
    dataset_conf : dict
        数据集配置
    train_conf : dict
        训练配置
    results_path : str
        结果保存路径
    """
    # 创建结果目录
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    
    in_exp = time.time()
    
    # 打开日志文件
    best_models = open(results_path + "/best_models.txt", "w")
    log_write = open(results_path + "/log.txt", "w")
    
    # 获取参数
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    lr = train_conf.get('lr')
    n_train = train_conf.get('n_train', 1)
    
    # 初始化结果存储
    test_acc = np.zeros((n_sub, n_train))
    test_kappa = np.zeros((n_sub, n_train))
    
    # LOSO 循环
    for target_sub in range(n_sub):
        print(f'\n{"="*60}')
        print(f'DANN Training - Target Subject: {target_sub+1}')
        print(f'Source Subjects: {[i+1 for i in range(n_sub) if i != target_sub]}')
        print(f'{"="*60}')
        
        log_write.write(f'\n{"="*60}\n')
        log_write.write(f'Target Subject: {target_sub+1}\n')
        log_write.write(f'{"="*60}\n')
        
        best_acc = 0
        
        # 多次训练取最佳
        for run in range(n_train):
            print(f'\nTraining Run {run+1}/{n_train}')
            
            # 设置随机种子
            tf.random.set_seed(run+1)
            np.random.seed(run+1)
            
            in_run = time.time()
            
            # 准备数据
            print("Loading and preparing DANN data...")
            X_source, y_source, domain_source, X_target, y_target = prepare_dann_data(
                data_path, target_sub, dataset='BCI2b', isStandard=isStandard)
            
            print(f"Source domain: {X_source.shape[0]} samples from 8 subjects")
            print(f"Target domain: {X_target.shape[0]} samples")
            
            # 划分验证集（从源域中划分）
            X_train, X_val, y_train, y_val, domain_train, domain_val = train_test_split(
                X_source, y_source, domain_source, test_size=0.2, random_state=42)
            
            # 创建模型
            dann_model, feature_extractor, label_classifier = get_dann_model(
                n_classes=dataset_conf.get('n_classes'),
                n_channels=dataset_conf.get('n_channels'),
                in_samples=dataset_conf.get('in_samples'),
                n_domains=8  # 8 个源域被试
            )
            
            # 编译模型
            dann_model.compile(
                optimizer=Adam(learning_rate=lr),
                loss={
                    'label_output': CategoricalCrossentropy(),
                    'domain_output': CategoricalCrossentropy()
                },
                loss_weights={
                    'label_output': 1.0,
                    'domain_output': 1.0
                },
                metrics={
                    'label_output': ['accuracy'],
                    'domain_output': ['accuracy']
                }
            )
            
            # 创建模型保存路径
            run_dir = os.path.join(results_path, 'saved_models', f'run-{run+1}')
            os.makedirs(run_dir, exist_ok=True)
            filepath = os.path.join(run_dir, f'subject-{target_sub+1}.weights.h5')
            
            # Lambda 调度器
            lambda_schedule = DomainAdaptationSchedule(total_epochs=epochs, gamma=10.0)
            
            # 获取 GRL 层
            grl_layer = None
            for layer in dann_model.layers:
                if isinstance(layer, GradientReversalLayer):
                    grl_layer = layer
                    break
            
            # 回调
            dann_callback = DANNTrainingCallback(grl_layer, lambda_schedule, log_write)
            
            # 训练模型
            print("\nTraining DANN model...")
            history = dann_model.fit(
                X_train,
                {'label_output': y_train, 'domain_output': domain_train},
                validation_data=(X_val, {'label_output': y_val, 'domain_output': domain_val}),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[dann_callback],
                verbose=0
            )
            
            # 保存最佳模型权重
            dann_model.save_weights(filepath)
            
            # 在目标域上评估（使用标签分类器）
            print("\nEvaluating on target domain...")
            y_pred = label_classifier.predict(X_target, verbose=0)
            y_pred_labels = y_pred.argmax(axis=-1)
            y_true_labels = y_target.argmax(axis=-1)
            
            test_acc[target_sub, run] = accuracy_score(y_true_labels, y_pred_labels)
            test_kappa[target_sub, run] = cohen_kappa_score(y_true_labels, y_pred_labels)
            
            out_run = time.time()
            
            # 记录结果
            info = (f'Target Sub: {target_sub+1}, Run: {run+1}, '
                   f'Time: {(out_run-in_run)/60:.1f}m, '
                   f'Test Acc: {test_acc[target_sub, run]:.4f}, '
                   f'Test Kappa: {test_kappa[target_sub, run]:.4f}')
            print(info)
            log_write.write(info + '\n')
            
            # 更新最佳模型
            if test_acc[target_sub, run] > best_acc:
                best_acc = test_acc[target_sub, run]
                best_model_path = filepath
        
        # 记录最佳模型
        best_run = np.argmax(test_acc[target_sub, :])
        best_model_path = os.path.join(results_path, 'saved_models', 
                                       f'run-{best_run+1}', 
                                       f'subject-{target_sub+1}.weights.h5')
        best_models.write(f'{best_model_path}\ttest_acc={test_acc[target_sub, best_run]:.4f}\n')
    
    out_exp = time.time()
    
    # 汇总结果
    info = '\n' + '='*60 + '\n'
    info += 'DANN-EEGNet Test Results (LOSO Cross-Validation)\n'
    info += '='*60 + '\n'
    
    for run in range(n_train):
        info += f'\nRun {run+1}:\n'
        info += 'Subject: '
        for sub in range(n_sub):
            info += f'{test_acc[sub, run]*100:5.2f} '
        info += f' | Avg: {np.mean(test_acc[:, run])*100:.2f}%\n'
    
    info += f'\nOverall Average Accuracy: {np.mean(test_acc)*100:.2f}%\n'
    info += f'Overall Average Kappa: {np.mean(test_kappa):.4f}\n'
    info += f'Total Training Time: {(out_exp-in_exp)/60:.1f} min\n'
    info += '='*60 + '\n'
    
    print(info)
    log_write.write(info)
    
    best_models.close()
    log_write.close()
    
    return test_acc, test_kappa


def run():
    """主函数"""
    # 获取数据集配置
    dataset_conf = get_BCI2b_dataset_info()
    
    # 数据路径
    data_path = "C:/Users/35696/Desktop/BCI_2b"
    if not data_path.endswith('/'):
        data_path += '/'
    dataset_conf['data_path'] = data_path
    
    # 结果路径
    results_path = os.getcwd() + "/results_DANN_EEGNet_BCI2b_LOSO"
    
    # 训练配置
    train_conf = {
        'batch_size': 64,
        'epochs': 100,  # DANN 通常不需要太多 epoch
        'lr': 0.001,
        'n_train': 1,
    }
    
    print("="*60)
    print("DANN-EEGNet Training on BCI Competition IV-2b")
    print("Domain Adversarial Neural Network for Cross-Subject Transfer")
    print("="*60)
    print(f"Dataset: {dataset_conf['name']}")
    print(f"Subjects: {dataset_conf['n_sub']} (LOSO)")
    print(f"Classes: {dataset_conf['n_classes']}")
    print(f"Channels: {dataset_conf['n_channels']}")
    print(f"Samples: {dataset_conf['in_samples']}")
    print(f"Training Epochs: {train_conf['epochs']}")
    print(f"Batch Size: {train_conf['batch_size']}")
    print("="*60)
    
    # 训练
    test_acc, test_kappa = train_dann(dataset_conf, train_conf, results_path)
    
    print("\nTraining completed!")
    print(f"Results saved in: {results_path}")


if __name__ == "__main__":
    run()
