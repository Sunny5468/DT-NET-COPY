# DANN-EEGNet 超参数自动调优系统

## 概述

本系统提供了完整的超参数自动搜索和优化功能，帮助找到DANN-EEGNet模型的最佳配置。

## 文件说明

- `hyperparameter_tuning.py` - 主调优脚本
- `hyperparam_config.yaml` - 完整搜索配置（972个配置组合）
- `hyperparam_config_quick.yaml` - 快速测试配置（4个配置组合）

## 功能特性

### 1. 自动超参数搜索
- ✅ 网格搜索策略
- ✅ 支持多维参数空间
- ✅ LOSO交叉验证
- ✅ 断点续传（自动保存中间结果）

### 2. 可调参数
- `epochs`: 训练轮数
- `lr`: 学习率
- `gamma`: Lambda调度gamma值
- `batch_size`: 批次大小
- `label_weight`: 标签分类器损失权重
- `domain_weight`: 域分类器损失权重

### 3. 结果记录
每次实验自动生成：
- ✅ JSON格式的详细结果
- ✅ CSV格式的汇总表格
- ✅ 最佳配置记录
- ✅ 训练历史保存
- ✅ 模型权重文件

### 4. 可视化分析
自动生成：
- ✅ 配置性能排名图
- ✅ 参数影响分析图
- ✅ 验证vs测试准确率散点图
- ✅ 完整文本报告

## 使用方法

### 快速开始（推荐）

使用快速配置进行初步测试：

```powershell
cd EEGNet_BCI2b
python hyperparameter_tuning.py
```

默认使用 `hyperparam_config.yaml`。要使用快速配置：

```python
# 在 hyperparameter_tuning.py 的 main() 函数中修改：
config_path = os.path.join(os.getcwd(), 'EEGNet_BCI2b', 'hyperparam_config_quick.yaml')
```

### 完整搜索

1. **编辑配置文件** `hyperparam_config.yaml`：
   ```yaml
   hyperparameters:
     epochs:
       values: [100, 150, 200]
     lr:
       values: [0.0001, 0.0005, 0.001, 0.002]
     # ... 其他参数
   ```

2. **运行调优脚本**：
   ```powershell
   python hyperparameter_tuning.py
   ```

3. **监控进度**：
   脚本会实时显示：
   - 当前测试的配置编号
   - 每个被试的训练进度
   - 中间结果统计

### 自定义搜索

#### 方法1：修改YAML配置

```yaml
hyperparameters:
  epochs:
    values: [100, 200]  # 只测试2个值
  lr:
    values: [0.001]     # 固定学习率
  # ...
```

#### 方法2：两阶段搜索

**阶段1：粗粒度快速搜索**
```yaml
# hyperparam_config_coarse.yaml
hyperparameters:
  epochs: {values: [100, 200]}
  lr: {values: [0.0005, 0.001]}
  gamma: {values: [8.0, 10.0]}
  batch_size: {values: [64]}
  label_weight: {values: [1.0, 1.5]}
  domain_weight: {values: [0.5, 1.0]}
# 总计: 32 个配置
```

**阶段2：精细搜索（在最优区域）**
```yaml
# 假设阶段1发现最优区域为: lr=0.001, gamma=10.0
hyperparameters:
  epochs: {values: [150, 200, 250]}
  lr: {values: [0.0008, 0.001, 0.0012]}
  gamma: {values: [9.0, 10.0, 11.0]}
  batch_size: {values: [64, 128]}
  label_weight: {values: [1.5, 2.0]}
  domain_weight: {values: [0.5, 0.75]}
```

## 结果解读

### 目录结构

```
hyperparameter_tuning_results/
└── hyperparam_tuning_20241107_143052/
    ├── config.yaml                  # 配置副本
    ├── all_results.json            # 所有实验结果
    ├── best_config.json            # 最佳配置
    ├── results_summary.csv         # 结果汇总表
    ├── REPORT.txt                  # 文本报告
    ├── ranking.png                 # 性能排名图
    ├── param_effects.png           # 参数影响图
    ├── val_vs_test.png            # 验证vs测试图
    ├── config_001/                # 配置1的详细结果
    │   ├── config.json
    │   ├── summary.json
    │   └── subject_1/
    │       ├── subject-1.weights.h5
    │       └── history_sub1.json
    ├── config_002/
    └── ...
```

### 关键指标

1. **avg_test_acc**: 平均测试准确率（主要优化目标）
2. **std_test_acc**: 测试准确率标准差（稳定性指标）
3. **avg_val_acc**: 平均验证准确率
4. **avg_test_kappa**: 平均Kappa系数
5. **total_training_time**: 总训练时间

### 最佳配置示例

```json
{
  "config_idx": 42,
  "avg_test_acc": 0.7689,
  "std_test_acc": 0.0423,
  "avg_val_acc": 0.8205,
  "avg_test_kappa": 0.5378,
  "config": {
    "epochs": 200,
    "lr": 0.001,
    "gamma": 10.0,
    "batch_size": 64,
    "label_weight": 1.5,
    "domain_weight": 0.5
  }
}
```

## 时间估算

### 单个模型训练时间
- 约 3-5 分钟/被试（50 epochs）
- 约 6-10 分钟/被试（100 epochs）
- 约 12-20 分钟/被试（200 epochs）

### 完整搜索时间估算

**快速测试配置**（4个配置 × 2个被试）：
- 预计时间：10-20 分钟

**粗粒度搜索**（32个配置 × 9个被试，100 epochs）：
- 预计时间：24-48 小时

**完整搜索**（972个配置 × 9个被试，平均150 epochs）：
- 预计时间：30-50 天（不推荐一次性运行）

### 建议策略

1. **快速验证**（1小时内）
   - 使用 `hyperparam_config_quick.yaml`
   - 2个被试 × 4个配置

2. **初步筛选**（1-2天）
   - 粗粒度参数空间
   - 全部9个被试
   - 较少epoch（50-100）

3. **精细优化**（1-2天）
   - 在最优区域细化搜索
   - 全部9个被试
   - 充足epoch（150-200）

## 高级功能

### 1. 并行化（未来版本）
```python
# 未来可以添加多GPU并行训练
from multiprocessing import Pool

# 在不同GPU上并行测试不同配置
```

### 2. 贝叶斯优化（未来版本）
```python
# 使用 Optuna 或 scikit-optimize
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    # ...
```

### 3. 早停优化
```yaml
fixed_params:
  use_early_stopping: true
  patience: 20  # 根据资源调整
```

## 故障排除

### 问题1：内存不足
**解决方案**：
- 减小 batch_size
- 减少并行测试的配置数
- 清理之前的结果文件

### 问题2：训练时间过长
**解决方案**：
- 使用快速配置先测试
- 减少 epochs 和参数组合数
- 使用早停机制

### 问题3：结果不收敛
**解决方案**：
- 检查学习率是否合适
- 增加训练轮数
- 调整 patience 值

## 最佳实践

1. **渐进式搜索**：从粗到细，逐步缩小范围
2. **记录实验**：保存每次搜索的配置和结果
3. **验证重现性**：用最佳配置多次训练验证稳定性
4. **资源规划**：根据可用时间合理设置搜索空间

## 性能基准

根据现有实验结果：

- **标准EEGNet（无DANN）**: 76.11% ± 8.2%
- **DANN-EEGNet（初始配置）**: 75.34% ± 7.9%
- **目标性能**: ≥ 77.0%

通过超参数调优，期望找到更优配置使DANN性能超越基准。

## 引用

如果使用本调优系统，请引用：

```bibtex
@article{ganin2015unsupervised,
  title={Unsupervised domain adaptation by backpropagation},
  author={Ganin, Yaroslav and Lempitsky, Victor},
  journal={ICML},
  year={2015}
}
```

## 联系方式

如有问题或建议，请提交Issue或联系维护者。
