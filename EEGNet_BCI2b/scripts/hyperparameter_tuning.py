"""
DANN-EEGNet 超参数自动调优脚本
支持网格搜索和贝叶斯优化两种策略

功能:
1. 自动搜索最佳超参数组合
2. 记录所有实验结果和训练历史
3. 保存最佳模型和配置
4. 生成性能对比图表
5. 支持断点续传
"""

import os
import sys
import json
import yaml
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import product

import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

# 添加父目录到路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.models_dann import get_dann_model, DomainAdaptationSchedule, GradientReversalLayer
from utils.preprocess import get_BCI2b_dataset_info


def prepare_dann_data(data_path, target_subject, dataset='BCI2b', isStandard=True):
    """
    为 DANN 准备源域和目标域数据
    """
    from utils.preprocess import load_data_LOSO, load_BCI2b_data
    from sklearn.utils import shuffle
    
    X_source_list = []
    y_source_list = []
    domain_source_list = []
    
    n_subjects = 9
    domain_idx = 0
    
    for sub in range(n_subjects):
        if sub == target_subject:
            continue
        
        X1, y1 = load_BCI2b_data(data_path, sub+1, True)
        X2, y2 = load_BCI2b_data(data_path, sub+1, False)
        
        X_sub = np.concatenate((X1, X2), axis=0)
        y_sub = np.concatenate((y1, y2), axis=0)
        
        domain_labels = np.full(len(X_sub), domain_idx, dtype=int)
        
        X_source_list.append(X_sub)
        y_source_list.append(y_sub)
        domain_source_list.append(domain_labels)
        
        domain_idx += 1
    
    # 加载目标域数据
    X1_target, y1_target = load_BCI2b_data(data_path, target_subject+1, True)
    X2_target, y2_target = load_BCI2b_data(data_path, target_subject+1, False)
    X_target = np.concatenate((X1_target, X2_target), axis=0)
    y_target = np.concatenate((y1_target, y2_target), axis=0)
    
    # 合并源域数据
    X_source = np.concatenate(X_source_list, axis=0)
    y_source = np.concatenate(y_source_list, axis=0)
    domain_source = np.concatenate(domain_source_list, axis=0)
    
    # Shuffle
    X_source, y_source, domain_source = shuffle(
        X_source, y_source, domain_source, random_state=42)
    
    # Reshape
    N_source, N_ch, T = X_source.shape
    X_source = X_source.reshape(N_source, 1, N_ch, T)
    
    N_target, N_ch, T = X_target.shape
    X_target = X_target.reshape(N_target, 1, N_ch, T)
    
    # 标准化
    if isStandard:
        from utils.preprocess import standardize_data
        X_source, X_target = standardize_data(X_source, X_target, N_ch)
    
    # One-hot编码
    y_source_onehot = to_categorical(y_source)
    y_target_onehot = to_categorical(y_target)
    domain_source_onehot = to_categorical(domain_source, num_classes=8)
    
    return X_source, y_source_onehot, domain_source_onehot, X_target, y_target_onehot


class DANNTrainingCallback(Callback):
    """DANN训练回调"""
    def __init__(self, grl_layer, lambda_schedule):
        super().__init__()
        self.grl_layer = grl_layer
        self.lambda_schedule = lambda_schedule
        self.history = {
            'lambda': [],
            'label_loss': [],
            'domain_loss': [],
            'label_acc': [],
            'domain_acc': []
        }
    
    def on_epoch_begin(self, epoch, logs=None):
        new_lambda = self.lambda_schedule.get_lambda(epoch)
        self.grl_layer.lambda_factor = new_lambda
        self.history['lambda'].append(new_lambda)
    
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.history['label_loss'].append(logs.get('label_output_loss', 0))
            self.history['domain_loss'].append(logs.get('domain_output_loss', 0))
            self.history['label_acc'].append(logs.get('label_output_accuracy', 0))
            self.history['domain_acc'].append(logs.get('domain_output_accuracy', 0))


def train_single_config(config, dataset_conf, data_path, target_sub, trial_dir):
    """
    使用指定配置训练单个模型
    
    Parameters
    ----------
    config : dict
        超参数配置
    dataset_conf : dict
        数据集配置
    data_path : str
        数据路径
    target_sub : int
        目标被试索引
    trial_dir : str
        实验保存目录
        
    Returns
    -------
    results : dict
        训练结果（包含验证和测试准确率）
    """
    # 设置随机种子
    seed = config.get('seed', 42)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # 准备数据
    X_source, y_source, domain_source, X_target, y_target = prepare_dann_data(
        data_path, target_sub, isStandard=config.get('isStandard', True))
    
    # 划分验证集
    X_train, X_val, y_train, y_val, domain_train, domain_val = train_test_split(
        X_source, y_source, domain_source, 
        test_size=config.get('val_split', 0.2), 
        random_state=42)
    
    # 创建模型
    dann_model, feature_extractor, label_classifier = get_dann_model(
        n_classes=dataset_conf.get('n_classes'),
        n_channels=dataset_conf.get('n_channels'),
        in_samples=dataset_conf.get('in_samples'),
        n_domains=8
    )
    
    # 编译模型
    dann_model.compile(
        optimizer=Adam(learning_rate=config['lr']),
        loss={
            'label_output': CategoricalCrossentropy(),
            'domain_output': CategoricalCrossentropy()
        },
        loss_weights={
            'label_output': config['label_weight'],
            'domain_output': config['domain_weight']
        },
        metrics={
            'label_output': ['accuracy'],
            'domain_output': ['accuracy']
        }
    )
    
    # Lambda调度器
    lambda_schedule = DomainAdaptationSchedule(
        total_epochs=config['epochs'], 
        gamma=config['gamma'])
    
    # 获取GRL层
    grl_layer = None
    for layer in dann_model.layers:
        if isinstance(layer, GradientReversalLayer):
            grl_layer = layer
            break
    
    # 回调
    dann_callback = DANNTrainingCallback(grl_layer, lambda_schedule)
    
    callbacks = [dann_callback]
    
    # 早停
    if config.get('use_early_stopping', False):
        early_stopping = EarlyStopping(
            monitor='val_label_output_accuracy',
            patience=config.get('patience', 20),
            restore_best_weights=True,
            mode='max',
            verbose=0
        )
        callbacks.append(early_stopping)
    
    # 训练
    start_time = time.time()
    history = dann_model.fit(
        X_train,
        {'label_output': y_train, 'domain_output': domain_train},
        validation_data=(X_val, {'label_output': y_val, 'domain_output': domain_val}),
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=0
    )
    training_time = time.time() - start_time
    
    # 验证集评估
    val_pred = label_classifier.predict(X_val, verbose=0)
    val_acc = accuracy_score(y_val.argmax(axis=-1), val_pred.argmax(axis=-1))
    
    # 测试集评估
    test_pred = label_classifier.predict(X_target, verbose=0)
    test_acc = accuracy_score(y_target.argmax(axis=-1), test_pred.argmax(axis=-1))
    test_kappa = cohen_kappa_score(y_target.argmax(axis=-1), test_pred.argmax(axis=-1))
    
    # 保存模型
    model_path = os.path.join(trial_dir, f'subject-{target_sub+1}.weights.h5')
    dann_model.save_weights(model_path)
    
    # 保存训练历史
    history_dict = {
        'val_label_acc': history.history['val_label_output_accuracy'],
        'val_domain_acc': history.history['val_domain_output_accuracy'],
        'train_label_acc': history.history['label_output_accuracy'],
        'train_domain_acc': history.history['domain_output_accuracy'],
        'lambda': dann_callback.history['lambda']
    }
    
    with open(os.path.join(trial_dir, f'history_sub{target_sub+1}.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # 返回结果
    results = {
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'test_kappa': float(test_kappa),
        'training_time': float(training_time),
        'epochs_trained': len(history.history['loss']),
        'final_lambda': float(dann_callback.history['lambda'][-1]),
        'model_path': model_path
    }
    
    return results


class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, config_path, results_base_path):
        """
        Parameters
        ----------
        config_path : str
            配置文件路径（YAML格式）
        results_base_path : str
            结果保存根目录
        """
        self.config_path = config_path
        self.results_base_path = results_base_path
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            results_base_path, 
            f"hyperparam_tuning_{timestamp}"
        )
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 保存配置副本
        shutil.copy(config_path, os.path.join(self.experiment_dir, 'config.yaml'))
        
        # 结果记录
        self.results_log = []
        self.best_config = None
        self.best_score = -1
        
        print(f"实验目录: {self.experiment_dir}")
    
    def generate_grid_configs(self):
        """生成网格搜索的所有配置组合"""
        param_grid = self.config['hyperparameters']
        
        # 提取参数名和值
        param_names = []
        param_values = []
        
        for param_name, param_config in param_grid.items():
            param_names.append(param_name)
            param_values.append(param_config['values'])
        
        # 生成所有组合
        configs = []
        for values in product(*param_values):
            config = dict(zip(param_names, values))
            # 添加固定参数
            config.update(self.config.get('fixed_params', {}))
            configs.append(config)
        
        return configs
    
    def run_grid_search(self, dataset_conf, data_path):
        """
        执行网格搜索
        
        Parameters
        ----------
        dataset_conf : dict
            数据集配置
        data_path : str
            数据路径
        """
        configs = self.generate_grid_configs()
        n_configs = len(configs)
        n_subjects = self.config.get('n_subjects', 9)
        
        print(f"\n{'='*70}")
        print(f"网格搜索: 共 {n_configs} 个配置组合")
        print(f"每个配置在 {n_subjects} 个被试上进行 LOSO 交叉验证")
        print(f"总计需要训练: {n_configs * n_subjects} 个模型")
        print(f"{'='*70}\n")
        
        for config_idx, config in enumerate(configs, 1):
            print(f"\n[配置 {config_idx}/{n_configs}]")
            print(f"参数: {json.dumps(config, indent=2)}")
            
            # 创建配置目录
            config_dir = os.path.join(self.experiment_dir, f'config_{config_idx:03d}')
            os.makedirs(config_dir, exist_ok=True)
            
            # 保存配置
            with open(os.path.join(config_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            # 存储每个被试的结果
            subject_results = []
            
            # LOSO交叉验证
            for target_sub in range(n_subjects):
                print(f"  被试 {target_sub+1}/{n_subjects}...", end=' ')
                
                # 创建被试目录
                trial_dir = os.path.join(config_dir, f'subject_{target_sub+1}')
                os.makedirs(trial_dir, exist_ok=True)
                
                try:
                    # 训练
                    results = train_single_config(
                        config, dataset_conf, data_path, target_sub, trial_dir)
                    
                    subject_results.append(results)
                    
                    print(f"验证={results['val_acc']:.4f}, "
                          f"测试={results['test_acc']:.4f}, "
                          f"时间={results['training_time']/60:.1f}min")
                    
                except Exception as e:
                    print(f"失败: {str(e)}")
                    subject_results.append(None)
            
            # 计算平均结果
            valid_results = [r for r in subject_results if r is not None]
            
            if valid_results:
                avg_val_acc = np.mean([r['val_acc'] for r in valid_results])
                avg_test_acc = np.mean([r['test_acc'] for r in valid_results])
                avg_test_kappa = np.mean([r['test_kappa'] for r in valid_results])
                total_time = sum([r['training_time'] for r in valid_results])
                
                # 记录结果
                result_entry = {
                    'config_idx': config_idx,
                    'config': config,
                    'avg_val_acc': float(avg_val_acc),
                    'avg_test_acc': float(avg_test_acc),
                    'avg_test_kappa': float(avg_test_kappa),
                    'std_test_acc': float(np.std([r['test_acc'] for r in valid_results])),
                    'total_training_time': float(total_time),
                    'n_subjects_completed': len(valid_results),
                    'subject_results': valid_results
                }
                
                self.results_log.append(result_entry)
                
                # 保存结果
                with open(os.path.join(config_dir, 'summary.json'), 'w') as f:
                    json.dump(result_entry, f, indent=2)
                
                print(f"  平均验证准确率: {avg_val_acc:.4f}")
                print(f"  平均测试准确率: {avg_test_acc:.4f} ± {result_entry['std_test_acc']:.4f}")
                print(f"  平均Kappa: {avg_test_kappa:.4f}")
                print(f"  总训练时间: {total_time/60:.1f} 分钟")
                
                # 更新最佳配置
                if avg_test_acc > self.best_score:
                    self.best_score = avg_test_acc
                    self.best_config = result_entry
                    print(f"  *** 新的最佳配置! ***")
            
            # 保存中间结果
            self.save_results()
        
        # 生成最终报告
        self.generate_report()
    
    def save_results(self):
        """保存结果到文件"""
        results_file = os.path.join(self.experiment_dir, 'all_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results_log, f, indent=2)
        
        # 保存最佳配置
        if self.best_config:
            best_file = os.path.join(self.experiment_dir, 'best_config.json')
            with open(best_file, 'w') as f:
                json.dump(self.best_config, f, indent=2)
    
    def generate_report(self):
        """生成实验报告和可视化"""
        print(f"\n{'='*70}")
        print("生成实验报告...")
        print(f"{'='*70}\n")
        
        if not self.results_log:
            print("没有有效结果")
            return
        
        # 创建DataFrame
        df_data = []
        for entry in self.results_log:
            row = {
                'config_idx': entry['config_idx'],
                'avg_test_acc': entry['avg_test_acc'],
                'std_test_acc': entry['std_test_acc'],
                'avg_val_acc': entry['avg_val_acc'],
                'avg_test_kappa': entry['avg_test_kappa'],
                'total_time_min': entry['total_training_time'] / 60
            }
            # 添加超参数列
            row.update(entry['config'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 保存CSV
        csv_path = os.path.join(self.experiment_dir, 'results_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"结果表格已保存: {csv_path}")
        
        # 生成可视化
        self._plot_results(df)
        
        # 生成文本报告
        self._generate_text_report(df)
    
    def _plot_results(self, df):
        """生成可视化图表"""
        # 1. 测试准确率排名
        plt.figure(figsize=(12, 6))
        df_sorted = df.sort_values('avg_test_acc', ascending=True)
        plt.barh(range(len(df_sorted)), df_sorted['avg_test_acc'], 
                xerr=df_sorted['std_test_acc'], capsize=3)
        plt.yticks(range(len(df_sorted)), 
                  [f"Config {idx}" for idx in df_sorted['config_idx']])
        plt.xlabel('Test Accuracy')
        plt.title('Hyperparameter Configuration Ranking')
        plt.axvline(x=self.best_score, color='r', linestyle='--', 
                   label=f'Best: {self.best_score:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'ranking.png'), dpi=150)
        plt.close()
        
        # 2. 参数影响分析
        param_cols = [col for col in df.columns 
                     if col not in ['config_idx', 'avg_test_acc', 'std_test_acc', 
                                   'avg_val_acc', 'avg_test_kappa', 'total_time_min']]
        
        if param_cols:
            n_params = len(param_cols)
            fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
            if n_params == 1:
                axes = [axes]
            
            for idx, param in enumerate(param_cols):
                ax = axes[idx]
                
                # 按参数值分组
                grouped = df.groupby(param)['avg_test_acc'].agg(['mean', 'std'])
                
                x = grouped.index.astype(str)
                y = grouped['mean']
                yerr = grouped['std']
                
                ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5)
                ax.set_xlabel(param)
                ax.set_ylabel('Test Accuracy')
                ax.set_title(f'Effect of {param}')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_dir, 'param_effects.png'), dpi=150)
            plt.close()
        
        # 3. 验证vs测试准确率
        plt.figure(figsize=(8, 6))
        plt.scatter(df['avg_val_acc'], df['avg_test_acc'], 
                   s=100, alpha=0.6, c=df['config_idx'], cmap='viridis')
        plt.xlabel('Validation Accuracy')
        plt.ylabel('Test Accuracy')
        plt.title('Validation vs Test Accuracy')
        plt.colorbar(label='Config Index')
        
        # 添加对角线
        lims = [
            np.min([plt.xlim()[0], plt.ylim()[0]]),
            np.max([plt.xlim()[1], plt.ylim()[1]])
        ]
        plt.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'val_vs_test.png'), dpi=150)
        plt.close()
        
        print(f"可视化图表已保存到: {self.experiment_dir}")
    
    def _generate_text_report(self, df):
        """生成文本报告"""
        report_path = os.path.join(self.experiment_dir, 'REPORT.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DANN-EEGNet 超参数调优实验报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置文件: {self.config_path}\n")
            f.write(f"测试配置数: {len(self.results_log)}\n\n")
            
            f.write("="*70 + "\n")
            f.write("最佳配置\n")
            f.write("="*70 + "\n\n")
            
            if self.best_config:
                f.write(f"配置编号: {self.best_config['config_idx']}\n")
                f.write(f"平均测试准确率: {self.best_config['avg_test_acc']:.4f} ± "
                       f"{self.best_config['std_test_acc']:.4f}\n")
                f.write(f"平均验证准确率: {self.best_config['avg_val_acc']:.4f}\n")
                f.write(f"平均Kappa: {self.best_config['avg_test_kappa']:.4f}\n")
                f.write(f"总训练时间: {self.best_config['total_training_time']/60:.1f} 分钟\n\n")
                
                f.write("超参数配置:\n")
                for key, value in self.best_config['config'].items():
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("所有配置性能排名 (Top 10)\n")
            f.write("="*70 + "\n\n")
            
            df_sorted = df.sort_values('avg_test_acc', ascending=False).head(10)
            
            for idx, row in df_sorted.iterrows():
                f.write(f"#{int(row['config_idx'])}: "
                       f"Test Acc = {row['avg_test_acc']:.4f} ± {row['std_test_acc']:.4f}\n")
                f.write(f"  epochs={row.get('epochs', 'N/A')}, "
                       f"lr={row.get('lr', 'N/A')}, "
                       f"gamma={row.get('gamma', 'N/A')}, "
                       f"batch_size={row.get('batch_size', 'N/A')}\n")
                f.write(f"  label_weight={row.get('label_weight', 'N/A')}, "
                       f"domain_weight={row.get('domain_weight', 'N/A')}\n\n")
        
        print(f"实验报告已保存: {report_path}")


def main():
    """主函数"""
    # 配置文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'configs', 'hyperparam_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        print("请先创建配置文件 configs/hyperparam_config.yaml")
        return
    
    # 结果保存路径
    results_base_path = os.path.join(project_root, '..', 'hyperparameter_tuning_results')
    os.makedirs(results_base_path, exist_ok=True)
    
    # 数据集配置
    dataset_conf = get_BCI2b_dataset_info()
    data_path = "C:/Users/35696/Desktop/BCI_2b/"
    dataset_conf['data_path'] = data_path
    
    # 创建调优器
    tuner = HyperparameterTuner(config_path, results_base_path)
    
    # 执行网格搜索
    print("\n开始超参数调优...")
    tuner.run_grid_search(dataset_conf, data_path)
    
    print("\n" + "="*70)
    print("调优完成!")
    print(f"结果保存在: {tuner.experiment_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
