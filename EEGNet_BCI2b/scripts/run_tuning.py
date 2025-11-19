"""
DANN-EEGNet 超参数调优 - 交互式启动脚本
"""

import os
import sys


def print_banner():
    """打印欢迎信息"""
    print("\n" + "="*70)
    print(" " * 15 + "DANN-EEGNet 超参数自动调优系统")
    print("="*70)
    print()


def select_config():
    """选择配置文件"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    configs_dir = os.path.join(project_root, 'configs')
    
    configs = {
        '1': {
            'file': os.path.join(configs_dir, 'hyperparam_config_quick.yaml'),
            'name': '快速测试配置',
            'desc': '4个配置 × 2个被试 = 8个模型',
            'time': '约10-20分钟'
        },
        '2': {
            'file': os.path.join(configs_dir, 'hyperparam_config.yaml'),
            'name': '完整搜索配置',
            'desc': '972个配置 × 9个被试 = 8748个模型',
            'time': '约30-50天（不推荐）'
        },
        '3': {
            'file': 'custom',
            'name': '自定义配置文件',
            'desc': '输入自己的配置文件路径',
            'time': '取决于配置'
        }
    }
    
    print("请选择配置方案:")
    print()
    for key, config in configs.items():
        print(f"[{key}] {config['name']}")
        print(f"    - {config['desc']}")
        print(f"    - 预计时间: {config['time']}")
        print()
    
    while True:
        choice = input("请输入选项 (1/2/3) 或 q 退出: ").strip()
        
        if choice.lower() == 'q':
            print("已退出")
            sys.exit(0)
        
        if choice in configs:
            if choice == '3':
                custom_path = input("请输入配置文件路径: ").strip()
                if os.path.exists(custom_path):
                    return custom_path
                else:
                    print(f"错误: 文件不存在: {custom_path}")
                    continue
            
            config_path = configs[choice]['file']
            
            if not os.path.exists(config_path):
                print(f"错误: 配置文件不存在: {config_path}")
                continue
            
            return config_path
        else:
            print("无效选项，请重新输入")


def confirm_start(config_path):
    """确认开始"""
    print("\n" + "-"*70)
    print(f"已选择配置文件: {os.path.basename(config_path)}")
    print()
    
    # 读取配置显示信息
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        n_subjects = config.get('n_subjects', 9)
        
        # 计算配置数量
        from itertools import product
        param_grid = config['hyperparameters']
        param_values = [param['values'] for param in param_grid.values()]
        n_configs = len(list(product(*param_values)))
        
        print(f"配置数量: {n_configs}")
        print(f"被试数量: {n_subjects}")
        print(f"总训练次数: {n_configs * n_subjects}")
        print()
        
    except Exception as e:
        print(f"警告: 无法解析配置文件: {e}")
    
    print("-"*70)
    
    while True:
        response = input("\n确认开始调优? (y/n): ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("请输入 y 或 n")


def main():
    """主函数"""
    print_banner()
    
    # 添加父目录到路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)
    
    # 检查必要文件
    required_files = [
        os.path.join(script_dir, 'hyperparameter_tuning.py'),
        os.path.join(project_root, 'models', 'models_dann.py'),
        os.path.join(project_root, 'utils', 'preprocess.py')
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("错误: 缺少必要文件:")
        for file in missing:
            print(f"  - {file}")
        print("\n请确保在正确的目录中运行此脚本")
        sys.exit(1)
    
    # 选择配置
    config_path = select_config()
    
    # 确认开始
    if not confirm_start(config_path):
        print("已取消")
        sys.exit(0)
    
    print("\n" + "="*70)
    print("开始超参数调优...")
    print("="*70 + "\n")
    
    # 修改主脚本的配置路径并运行
    # 修改主脚本的配置路径并运行
    # 创建临时脚本
    temp_script = """
import os
import sys

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from scripts.hyperparameter_tuning import HyperparameterTuner
from utils.preprocess import get_BCI2b_dataset_info

# 配置文件路径
config_path = r'{config_path}'

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
tuner.run_grid_search(dataset_conf, data_path)

print("\\n" + "="*70)
print("调优完成!")
print(f"结果保存在: {{tuner.experiment_dir}}")
print("="*70)
""".format(config_path=config_path)
    
    # 执行
    try:
        exec(temp_script)
    except KeyboardInterrupt:
        print("\n\n用户中断训练")
        print("中间结果已保存，可以稍后继续")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
