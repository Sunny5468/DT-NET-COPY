"""
简单的结构测试脚本 - 无特殊字符
"""

import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*70)
print("文件结构测试")
print("="*70)

# 测试1: 导入models模块
print("\n[测试1] 导入models模块...")
try:
    from models.models import get_EEGNet_model
    print("成功: models.models")
except Exception as e:
    print(f"失败: {e}")

try:
    from models.models_dann import get_dann_model
    print("成功: models.models_dann")
except Exception as e:
    print(f"失败: {e}")

# 测试2: 导入utils模块
print("\n[测试2] 导入utils模块...")
try:
    from utils.preprocess import get_BCI2b_dataset_info
    print("成功: utils.preprocess")
except Exception as e:
    print(f"失败: {e}")

# 测试3: 检查配置文件
print("\n[测试3] 检查配置文件...")
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
configs_dir = os.path.join(project_root, 'configs')

config_files = [
    'hyperparam_config.yaml',
    'hyperparam_config_quick.yaml',
    'requirements.txt'
]

for cf in config_files:
    path = os.path.join(configs_dir, cf)
    if os.path.exists(path):
        print(f"存在: {cf}")
    else:
        print(f"缺失: {cf}")

# 测试4: 检查文档
print("\n[测试4] 检查文档...")
docs_dir = os.path.join(project_root, 'docs')
doc_files = os.listdir(docs_dir) if os.path.exists(docs_dir) else []
print(f"文档数量: {len(doc_files)}")
for df in sorted(doc_files)[:5]:
    print(f"  - {df}")

# 测试5: 检查脚本
print("\n[测试5] 检查脚本...")
scripts_dir = os.path.join(project_root, 'scripts')
script_files = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
print(f"脚本数量: {len(script_files)}")
for sf in sorted(script_files):
    print(f"  - {sf}")

print("\n" + "="*70)
print("测试完成!")
print("="*70)
