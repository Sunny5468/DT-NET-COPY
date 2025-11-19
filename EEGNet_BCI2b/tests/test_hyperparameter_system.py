"""
快速测试脚本 - 验证超参数调优系统是否正常工作
仅使用极少的数据和配置，快速验证功能
"""

import os
import sys
import yaml
import tempfile

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 创建最小测试配置
test_config = {
    'n_subjects': 1,  # 仅测试1个被试
    'hyperparameters': {
        'epochs': {'values': [10], 'description': '测试用极少epoch'},
        'lr': {'values': [0.001], 'description': '学习率'},
        'gamma': {'values': [10.0], 'description': 'gamma'},
        'batch_size': {'values': [64], 'description': '批次大小'},
        'label_weight': {'values': [1.0], 'description': '标签权重'},
        'domain_weight': {'values': [0.5], 'description': '域权重'}
    },
    'fixed_params': {
        'seed': 42,
        'isStandard': True,
        'val_split': 0.2,
        'use_early_stopping': False,  # 禁用早停加快测试
        'patience': 5
    }
}

def test_config_loading():
    """测试配置加载"""
    print("测试1: 配置文件加载...")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        with open(temp_config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config['n_subjects'] == 1
        assert 'hyperparameters' in loaded_config
        print("✓ 配置加载成功")
        
        os.unlink(temp_config_path)
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def test_imports():
    """测试必要的模块导入"""
    print("\n测试2: 模块导入...")
    try:
        from scripts.hyperparameter_tuning import HyperparameterTuner, train_single_config
        from models.models_dann import get_dann_model
        from utils.preprocess import get_BCI2b_dataset_info
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n测试3: DANN模型创建...")
    try:
        from models.models_dann import get_dann_model
        
        model, feature_extractor, label_classifier = get_dann_model(
            n_classes=2,
            n_channels=3,
            in_samples=1125,
            n_domains=8
        )
        
        assert model is not None
        assert feature_extractor is not None
        assert label_classifier is not None
        
        print(f"✓ 模型创建成功")
        print(f"  - DANN模型参数: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False


def test_grid_generation():
    """测试配置组合生成"""
    print("\n测试4: 网格配置生成...")
    try:
        from itertools import product
        
        param_grid = test_config['hyperparameters']
        param_names = []
        param_values = []
        
        for param_name, param_config in param_grid.items():
            param_names.append(param_name)
            param_values.append(param_config['values'])
        
        configs = []
        for values in product(*param_values):
            config = dict(zip(param_names, values))
            config.update(test_config.get('fixed_params', {}))
            configs.append(config)
        
        print(f"✓ 成功生成 {len(configs)} 个配置组合")
        print(f"  示例配置: {configs[0]}")
        return True
    except Exception as e:
        print(f"✗ 配置生成失败: {e}")
        return False


def test_data_preparation():
    """测试数据准备（如果数据可用）"""
    print("\n测试5: 数据准备...")
    data_path = "C:/Users/35696/Desktop/BCI_2b/"
    
    if not os.path.exists(data_path):
        print("⚠ 数据路径不存在，跳过数据准备测试")
        return True
    
    try:
        from scripts.hyperparameter_tuning import prepare_dann_data
        
        print("  正在加载数据（可能需要几秒钟）...")
        X_source, y_source, domain_source, X_target, y_target = prepare_dann_data(
            data_path, target_subject=0, isStandard=True)
        
        print(f"✓ 数据加载成功")
        print(f"  - 源域数据: {X_source.shape}")
        print(f"  - 目标域数据: {X_target.shape}")
        print(f"  - 域标签: {domain_source.shape}")
        return True
    except Exception as e:
        print(f"✗ 数据准备失败: {e}")
        return False


def test_single_training():
    """测试单个配置训练（如果数据可用）"""
    print("\n测试6: 单配置训练（可选）...")
    data_path = "C:/Users/35696/Desktop/BCI_2b/"
    
    if not os.path.exists(data_path):
        print("⚠ 数据路径不存在，跳过训练测试")
        return True
    
    response = input("  是否进行完整训练测试？这将需要2-3分钟。(y/N): ")
    if response.lower() != 'y':
        print("  跳过训练测试")
        return True
    
    try:
        from scripts.hyperparameter_tuning import train_single_config
        from utils.preprocess import get_BCI2b_dataset_info
        import tempfile
        
        dataset_conf = get_BCI2b_dataset_info()
        
        config = {
            'epochs': 5,  # 极少epoch
            'lr': 0.001,
            'gamma': 10.0,
            'batch_size': 64,
            'label_weight': 1.0,
            'domain_weight': 0.5,
            'seed': 42,
            'isStandard': True,
            'val_split': 0.2,
            'use_early_stopping': False,
            'patience': 3
        }
        
        with tempfile.TemporaryDirectory() as trial_dir:
            print("  开始训练（5个epoch）...")
            results = train_single_config(
                config, dataset_conf, data_path, 
                target_sub=0, trial_dir=trial_dir)
            
            print(f"✓ 训练完成")
            print(f"  - 验证准确率: {results['val_acc']:.4f}")
            print(f"  - 测试准确率: {results['test_acc']:.4f}")
            print(f"  - 训练时间: {results['training_time']:.1f}秒")
            return True
    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*70)
    print("DANN-EEGNet 超参数调优系统 - 快速测试")
    print("="*70)
    
    tests = [
        test_config_loading,
        test_imports,
        test_model_creation,
        test_grid_generation,
        test_data_preparation,
        test_single_training,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ 测试异常: {e}")
            results.append(False)
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n✓ 所有测试通过！系统可以正常使用。")
        print("\n下一步:")
        print("1. 使用快速配置测试: python hyperparameter_tuning.py")
        print("2. 查看 README_HYPERPARAMETER_TUNING.md 了解详细用法")
    else:
        print("\n⚠ 部分测试失败，请检查错误信息。")
    
    print("="*70)


if __name__ == "__main__":
    main()
