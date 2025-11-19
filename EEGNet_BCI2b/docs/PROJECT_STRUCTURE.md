# 📁 项目文件结构整理说明

## 整理前后对比

### ❌ 整理前（混乱）
所有文件堆在一个目录下，难以管理和查找。

### ✅ 整理后（清晰）
按功能分类到不同文件夹，结构清晰，易于维护。

## 📊 新文件结构

```
EEGNet_BCI2b/
│
├── 📋 README.md                      # 项目主文档
├── 📦 __init__.py                    # Python包初始化
│
├── 📁 configs/                       # 配置文件夹
│   ├── __init__.py
│   ├── hyperparam_config.yaml        # 完整超参数配置
│   ├── hyperparam_config_quick.yaml  # 快速测试配置
│   └── requirements.txt              # 依赖包列表
│
├── 📁 models/                        # 模型定义文件夹
│   ├── __init__.py
│   ├── models.py                     # 标准EEGNet模型
│   └── models_dann.py                # DANN-EEGNet模型
│
├── 📁 utils/                         # 工具函数文件夹
│   ├── __init__.py
│   └── preprocess.py                 # 数据预处理工具
│
├── 📁 scripts/                       # 执行脚本文件夹
│   ├── __init__.py
│   ├── main_EEGNet_BCI2b_LOSO.py           # 标准EEGNet训练
│   ├── main_DANN_EEGNet_BCI2b_LOSO.py      # DANN训练
│   ├── hyperparameter_tuning.py             # 超参数调优引擎
│   ├── run_tuning.py                        # 交互式调优启动
│   ├── view_tuning_results.py               # 结果查看器
│   └── USAGE_GUIDE.py                       # 使用指南程序
│
├── 📁 tests/                         # 测试脚本文件夹
│   ├── __init__.py
│   └── test_hyperparameter_system.py # 系统功能测试
│
├── 📁 docs/                          # 文档文件夹
│   ├── README.md                     # 原始项目说明
│   ├── README_DANN.md                # DANN实现说明
│   ├── README_HYPERPARAMETER_TUNING.md  # 超参数调优详细文档
│   ├── README_TUNING.md              # 调优导航文档
│   ├── QUICKSTART.md                 # 快速入门指南
│   └── PROJECT_SUMMARY.md            # 项目总结
│
├── 📁 results_DANN_EEGNet_BCI2b_LOSO/ # DANN训练结果
│   ├── best_models.txt
│   └── log.txt
│
└── 📁 __pycache__/                   # Python缓存（自动生成）

```

## 🎯 文件夹功能说明

### 1. `configs/` - 配置文件
**用途**: 存放所有配置文件
- **hyperparam_config.yaml**: 完整的超参数搜索空间配置（972个组合）
- **hyperparam_config_quick.yaml**: 快速测试配置（4个组合）
- **requirements.txt**: Python包依赖列表

**为什么**: 配置文件集中管理，便于版本控制和修改

### 2. `models/` - 模型定义
**用途**: 存放神经网络模型定义
- **models.py**: 标准EEGNet模型实现
- **models_dann.py**: DANN-EEGNet模型（包含梯度反转层）

**为什么**: 模型代码独立，便于复用和测试

### 3. `utils/` - 工具函数
**用途**: 存放通用工具函数
- **preprocess.py**: 数据加载、预处理、标准化

**为什么**: 工具函数统一管理，避免代码重复

### 4. `scripts/` - 执行脚本
**用途**: 存放所有可执行的Python脚本
- **main_*.py**: 训练脚本
- **hyperparameter_tuning.py**: 自动调优引擎
- **run_tuning.py**: 交互式界面
- **view_tuning_results.py**: 结果分析

**为什么**: 执行脚本集中管理，清晰知道哪些是入口程序

### 5. `tests/` - 测试脚本
**用途**: 存放单元测试和系统测试
- **test_hyperparameter_system.py**: 系统功能测试

**为什么**: 测试代码分离，符合软件工程规范

### 6. `docs/` - 文档
**用途**: 存放所有Markdown文档
- 各种README和使用指南

**为什么**: 文档集中存放，便于查阅和维护

## 🔧 导入路径更新

整理后，所有脚本的导入路径已自动更新：

### 之前（旧）
```python
from models import get_EEGNet_model
from preprocess import get_data
from models_dann import get_dann_model
```

### 现在（新）
```python
# 在脚本开头添加项目根目录到路径
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 使用新的模块化导入
from models.models import get_EEGNet_model
from utils.preprocess import get_data
from models.models_dann import get_dann_model
```

## 🚀 使用方法更新

### 1. 标准EEGNet训练
```powershell
cd EEGNet_BCI2b/scripts
python main_EEGNet_BCI2b_LOSO.py
```

### 2. DANN训练
```powershell
cd EEGNet_BCI2b/scripts
python main_DANN_EEGNet_BCI2b_LOSO.py
```

### 3. 超参数调优
```powershell
cd EEGNet_BCI2b/scripts
python run_tuning.py
```

### 4. 查看结果
```powershell
cd EEGNet_BCI2b/scripts
python view_tuning_results.py
```

### 5. 系统测试
```powershell
cd EEGNet_BCI2b/tests
python test_hyperparameter_system.py
```

## ✅ 整理带来的好处

1. **清晰的结构** - 一眼就能找到需要的文件
2. **模块化设计** - 代码更易维护和扩展
3. **符合规范** - 遵循Python项目最佳实践
4. **便于协作** - 团队成员容易理解项目结构
5. **易于测试** - 测试代码独立，便于持续集成

## 📝 注意事项

1. **配置文件位置**: 现在在 `configs/` 文件夹中
2. **脚本执行**: 需要在 `scripts/` 目录下运行
3. **导入路径**: 已自动更新，无需手动修改
4. **文档查阅**: 所有文档在 `docs/` 文件夹中

## 🔄 快速导航

- 📖 查看文档 → `cd docs`
- ⚙️ 修改配置 → `cd configs`
- 🔬 查看模型 → `cd models`
- 🚀 运行脚本 → `cd scripts`
- 🧪 运行测试 → `cd tests`

---

**整理完成日期**: 2025年11月7日  
**整理工具**: 自动化脚本 + 手动调整
