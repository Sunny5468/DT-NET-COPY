"""
查看超参数调优结果
快速浏览最佳配置和性能对比
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


def find_latest_experiment():
    """查找最新的实验结果"""
    results_base = os.path.join(os.getcwd(), 'hyperparameter_tuning_results')
    
    if not os.path.exists(results_base):
        print("错误: 没有找到调优结果目录")
        return None
    
    experiments = glob(os.path.join(results_base, 'hyperparam_tuning_*'))
    
    if not experiments:
        print("错误: 没有找到任何实验结果")
        return None
    
    # 按时间排序，返回最新的
    experiments.sort(reverse=True)
    return experiments[0]


def load_results(experiment_dir):
    """加载实验结果"""
    # 加载最佳配置
    best_config_path = os.path.join(experiment_dir, 'best_config.json')
    if os.path.exists(best_config_path):
        with open(best_config_path, 'r') as f:
            best_config = json.load(f)
    else:
        best_config = None
    
    # 加载所有结果
    all_results_path = os.path.join(experiment_dir, 'all_results.json')
    if os.path.exists(all_results_path):
        with open(all_results_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = None
    
    # 加载CSV
    csv_path = os.path.join(experiment_dir, 'results_summary.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = None
    
    return best_config, all_results, df


def print_best_config(best_config):
    """打印最佳配置"""
    if not best_config:
        print("未找到最佳配置")
        return
    
    print("\n" + "="*70)
    print("最佳配置")
    print("="*70)
    print(f"\n配置编号: {best_config['config_idx']}")
    print(f"平均测试准确率: {best_config['avg_test_acc']:.4f} ± {best_config['std_test_acc']:.4f}")
    print(f"平均验证准确率: {best_config['avg_val_acc']:.4f}")
    print(f"平均Kappa: {best_config['avg_test_kappa']:.4f}")
    print(f"总训练时间: {best_config['total_training_time']/60:.1f} 分钟")
    
    print("\n超参数:")
    for key, value in best_config['config'].items():
        print(f"  {key:20s}: {value}")
    
    print("\n每个被试的测试结果:")
    for i, result in enumerate(best_config['subject_results'], 1):
        if result:
            print(f"  被试 {i}: 测试准确率 = {result['test_acc']:.4f}, "
                  f"Kappa = {result['test_kappa']:.4f}")


def print_top_configs(df, n=5):
    """打印Top N配置"""
    if df is None:
        print("未找到结果数据")
        return
    
    print("\n" + "="*70)
    print(f"Top {n} 配置")
    print("="*70)
    
    df_sorted = df.sort_values('avg_test_acc', ascending=False).head(n)
    
    for rank, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"\n第 {rank} 名:")
        print(f"  配置编号: {int(row['config_idx'])}")
        print(f"  测试准确率: {row['avg_test_acc']:.4f} ± {row['std_test_acc']:.4f}")
        print(f"  验证准确率: {row['avg_val_acc']:.4f}")
        
        # 打印关键参数
        if 'epochs' in row:
            print(f"  epochs={int(row['epochs'])}, lr={row['lr']:.4f}, "
                  f"gamma={row['gamma']}, batch_size={int(row['batch_size'])}")


def plot_comparison(df, experiment_dir):
    """生成对比图"""
    if df is None:
        return
    
    print("\n生成对比图...")
    
    # 参数vs准确率对比
    param_cols = ['epochs', 'lr', 'gamma', 'batch_size', 'label_weight', 'domain_weight']
    param_cols = [col for col in param_cols if col in df.columns]
    
    if not param_cols:
        print("没有可用的参数列")
        return
    
    n_params = len(param_cols)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, param in enumerate(param_cols):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # 按参数分组
        grouped = df.groupby(param)['avg_test_acc'].agg(['mean', 'std', 'count'])
        
        x = grouped.index.astype(str)
        y = grouped['mean']
        yerr = grouped['std']
        
        ax.errorbar(range(len(x)), y, yerr=yerr, marker='o', capsize=5, linewidth=2)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=45)
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.set_title(f'Effect of {param}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([y.min() - 0.02, y.max() + 0.02])
    
    # 隐藏多余的子图
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(experiment_dir, 'comparison_detailed.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {save_path}")
    plt.close()


def show_statistics(df):
    """显示统计信息"""
    if df is None:
        return
    
    print("\n" + "="*70)
    print("统计信息")
    print("="*70)
    
    print(f"\n总配置数: {len(df)}")
    print(f"平均测试准确率: {df['avg_test_acc'].mean():.4f} ± {df['avg_test_acc'].std():.4f}")
    print(f"最高测试准确率: {df['avg_test_acc'].max():.4f}")
    print(f"最低测试准确率: {df['avg_test_acc'].min():.4f}")
    print(f"准确率范围: {df['avg_test_acc'].max() - df['avg_test_acc'].min():.4f}")
    
    if 'total_time_min' in df.columns:
        print(f"\n总训练时间: {df['total_time_min'].sum():.1f} 分钟 ({df['total_time_min'].sum()/60:.1f} 小时)")
        print(f"平均每配置: {df['total_time_min'].mean():.1f} 分钟")


def main():
    """主函数"""
    print("="*70)
    print("超参数调优结果查看器")
    print("="*70)
    
    # 查找实验
    print("\n正在查找实验结果...")
    experiment_dir = find_latest_experiment()
    
    if not experiment_dir:
        return
    
    print(f"找到实验: {os.path.basename(experiment_dir)}")
    print(f"路径: {experiment_dir}")
    
    # 加载结果
    print("\n正在加载结果...")
    best_config, all_results, df = load_results(experiment_dir)
    
    # 显示结果
    print_best_config(best_config)
    print_top_configs(df, n=5)
    show_statistics(df)
    
    # 生成图表
    if df is not None and len(df) > 1:
        plot_comparison(df, experiment_dir)
    
    # 显示报告位置
    report_path = os.path.join(experiment_dir, 'REPORT.txt')
    if os.path.exists(report_path):
        print("\n" + "="*70)
        print(f"完整报告: {report_path}")
        print("="*70)
        
        response = input("\n是否打印完整报告? (y/n): ").strip().lower()
        if response == 'y':
            with open(report_path, 'r', encoding='utf-8') as f:
                print("\n" + f.read())
    
    print("\n完成!")


if __name__ == "__main__":
    main()
