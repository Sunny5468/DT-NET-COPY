# 超参数自动调优系统

## 🎯 快速导航

- **新手入门**: 阅读 [`QUICKSTART.md`](QUICKSTART.md)
- **详细文档**: 阅读 [`README_HYPERPARAMETER_TUNING.md`](README_HYPERPARAMETER_TUNING.md)
- **项目总结**: 阅读 [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)

## ⚡ 三步快速开始

### 1️⃣ 测试系统
```powershell
python test_hyperparameter_system.py
```
✅ 确保所有测试通过（6/6）

### 2️⃣ 快速试运行
```powershell
python run_tuning.py
```
选择 `[1] 快速测试配置`（15-20分钟）

### 3️⃣ 查看结果
```powershell
python view_tuning_results.py
```

## 📁 文件说明

### 核心脚本
| 文件 | 说明 | 用途 |
|------|------|------|
| `hyperparameter_tuning.py` | 主调优脚本 | 核心功能实现 |
| `models_dann.py` | DANN模型 | 模型定义 |
| `preprocess.py` | 数据处理 | 数据加载 |

### 配置文件
| 文件 | 配置数 | 时间 | 用途 |
|------|--------|------|------|
| `hyperparam_config_quick.yaml` | 4 × 2 = 8 | 15-20分钟 | 快速测试 ⭐ |
| `hyperparam_config.yaml` | 972 × 9 = 8748 | 30-50天 | 完整搜索 |

### 辅助工具
| 文件 | 说明 |
|------|------|
| `test_hyperparameter_system.py` | 系统测试 |
| `run_tuning.py` | 交互式启动 ⭐ |
| `view_tuning_results.py` | 结果查看 ⭐ |

### 文档
| 文件 | 内容 |
|------|------|
| `QUICKSTART.md` | 快速入门 ⭐ |
| `README_HYPERPARAMETER_TUNING.md` | 详细文档 |
| `PROJECT_SUMMARY.md` | 项目总结 |

## 🎓 使用场景

### 场景1: 快速验证（新手推荐）
```powershell
python test_hyperparameter_system.py  # 测试
python run_tuning.py                   # 选择[1]快速配置
python view_tuning_results.py          # 查看结果
```
⏱️ 总时间：20-30分钟

### 场景2: 初步调优
编辑 `hyperparam_config_quick.yaml`，增加被试数到5-7个
```powershell
python run_tuning.py  # 选择[1]快速配置
```
⏱️ 时间：2-4小时

### 场景3: 完整调优
创建自定义配置，平衡搜索空间和时间
```powershell
python run_tuning.py  # 选择[3]自定义配置
```
⏱️ 时间：根据配置而定

## 📊 输出结果

每次实验生成：
```
hyperparameter_tuning_results/
└── hyperparam_tuning_YYYYMMDD_HHMMSS/
    ├── best_config.json      ⭐ 最佳配置
    ├── results_summary.csv   ⭐ Excel表格
    ├── REPORT.txt           ⭐ 完整报告
    ├── ranking.png          📊 性能排名
    ├── param_effects.png    📊 参数影响
    └── val_vs_test.png      📊 对比图
```

## 🎯 优化目标

| 指标 | 基准 | DANN初始 | 目标 |
|------|------|----------|------|
| 测试准确率 | 76.11% | 75.34% | ≥77.0% |
| 标准差 | 8.2% | 7.9% | <8.0% |

## 📚 更多信息

- **系统架构**: 见 `PROJECT_SUMMARY.md`
- **参数说明**: 见 `README_HYPERPARAMETER_TUNING.md`
- **故障排除**: 见 `QUICKSTART.md` FAQ部分

## ✅ 系统状态

- ✅ 所有测试通过（6/6）
- ✅ 文档完整
- ✅ 示例配置就绪
- ✅ 可立即使用

## 🚀 立即开始

```powershell
cd EEGNet_BCI2b
python run_tuning.py
```

选择快速配置，开始你的第一次调优！
