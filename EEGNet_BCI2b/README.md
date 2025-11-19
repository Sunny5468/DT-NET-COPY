# EEGNet BCI-2b 项目

基于EEGNet的BCI Competition IV-2b数据集分类，包含DANN域对抗迁移学习和超参数自动调优系统。

## 📁 项目结构

```
EEGNet_BCI2b/
├── configs/                          # 配置文件
│   ├── hyperparam_config.yaml        # 完整超参数搜索配置
│   ├── hyperparam_config_quick.yaml  # 快速测试配置
│   └── requirements.txt              # Python依赖
│
├── models/                           # 模型定义
│   ├── __init__.py
│   ├── models.py                     # 标准EEGNet模型
│   └── models_dann.py                # DANN-EEGNet模型
│
├── utils/                            # 工具函数
│   ├── __init__.py
│   └── preprocess.py                 # 数据预处理
│
├── scripts/                          # 执行脚本
│   ├── __init__.py
│   ├── main_EEGNet_BCI2b_LOSO.py           # 标准EEGNet训练
│   ├── main_DANN_EEGNet_BCI2b_LOSO.py      # DANN-EEGNet训练
│   ├── hyperparameter_tuning.py             # 超参数调优引擎
│   ├── run_tuning.py                        # 交互式调优启动
│   ├── view_tuning_results.py               # 结果查看器
│   └── USAGE_GUIDE.py                       # 使用指南
│
├── tests/                            # 测试脚本
│   ├── __init__.py
│   └── test_hyperparameter_system.py # 系统测试
│
├── docs/                             # 文档
│   ├── README.md                     # 项目说明
│   ├── README_DANN.md                # DANN实现说明
│   ├── README_HYPERPARAMETER_TUNING.md  # 超参数调优详细文档
│   ├── README_TUNING.md              # 调优导航
│   ├── QUICKSTART.md                 # 快速入门
│   └── PROJECT_SUMMARY.md            # 项目总结
│
├── results_DANN_EEGNet_BCI2b_LOSO/   # DANN训练结果
└── __pycache__/                      # Python缓存

```

## 🚀 快速开始

### 1. 标准EEGNet训练
```powershell
cd scripts
python main_EEGNet_BCI2b_LOSO.py
```

### 2. DANN-EEGNet训练
```powershell
cd scripts
python main_DANN_EEGNet_BCI2b_LOSO.py
```

### 3. 超参数自动调优
```powershell
cd scripts
python run_tuning.py
```

## 📚 文档导航

- **快速入门**: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **DANN说明**: [docs/README_DANN.md](docs/README_DANN.md)
- **超参数调优**: [docs/README_HYPERPARAMETER_TUNING.md](docs/README_HYPERPARAMETER_TUNING.md)
- **项目总结**: [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)

## 🎯 性能基准

| 方法 | 测试准确率 | 标准差 |
|------|-----------|--------|
| 标准EEGNet | 76.11% | ±8.2% |
| DANN-EEGNet | 75.34% | ±7.9% |

## 📦 主要模块

### Models (`models/`)
- `models.py`: 标准EEGNet实现（F1=8, D=2, kernLength=64）
- `models_dann.py`: DANN-EEGNet，包含梯度反转层和域分类器

### Utils (`utils/`)
- `preprocess.py`: BCI-2b数据加载、LOSO划分、标准化

### Scripts (`scripts/`)
- `main_EEGNet_BCI2b_LOSO.py`: 标准训练流程
- `main_DANN_EEGNet_BCI2b_LOSO.py`: DANN训练流程
- `hyperparameter_tuning.py`: 自动调优引擎（网格搜索）
- `run_tuning.py`: 交互式调优界面
- `view_tuning_results.py`: 结果分析和可视化

### Tests (`tests/`)
- `test_hyperparameter_system.py`: 6项系统功能测试

### Configs (`configs/`)
- `hyperparam_config.yaml`: 完整搜索空间配置
- `hyperparam_config_quick.yaml`: 快速测试配置（推荐）
- `requirements.txt`: Python依赖列表

## 🔧 环境配置

```powershell
conda activate eeg_env
pip install -r configs/requirements.txt
```

## 📊 数据集

BCI Competition IV-2b:
- 9个被试
- 3个EEG通道（C3, Cz, C4）
- 2类运动想象（左手/右手）
- 250Hz采样率
- 1125个采样点（4.5秒）

## 🎓 引用

```bibtex
@article{lawhern2018eegnet,
  title={EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
  author={Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal={Journal of neural engineering},
  year={2018}
}

@inproceedings{ganin2015unsupervised,
  title={Unsupervised domain adaptation by backpropagation},
  author={Ganin, Yaroslav and Lempitsky, Victor},
  booktitle={International conference on machine learning},
  year={2015}
}
```

## 📝 许可

本项目用于学术研究。

## 👥 维护者

BCI_DTNet_copy 项目组

---

**最后更新**: 2025年11月7日
