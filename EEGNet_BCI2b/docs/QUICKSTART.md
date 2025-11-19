# DANN-EEGNet 超参数调优 - 快速上手指南

## 📋 文件清单

### 核心文件
- `hyperparameter_tuning.py` - 主调优脚本
- `models_dann.py` - DANN模型定义
- `preprocess.py` - 数据预处理

### 配置文件
- `hyperparam_config.yaml` - 完整搜索配置（972个组合）
- `hyperparam_config_quick.yaml` - 快速测试配置（4个组合）

### 辅助工具
- `run_tuning.py` - 交互式启动脚本（推荐）
- `test_hyperparameter_system.py` - 系统测试脚本
- `view_tuning_results.py` - 结果查看器

### 文档
- `README_HYPERPARAMETER_TUNING.md` - 详细文档
- `QUICKSTART.md` - 本文档

## 🚀 三步快速开始

### 步骤1: 测试系统

首先验证系统是否正常工作：

```powershell
cd EEGNet_BCI2b
python test_hyperparameter_system.py
```

应该看到所有测试通过（6/6）。

### 步骤2: 快速试运行

使用快速配置进行试运行（约15-20分钟）：

```powershell
python run_tuning.py
```

选择选项 `[1] 快速测试配置`，然后确认开始。

这将测试4个配置组合 × 2个被试 = 8个模型。

### 步骤3: 查看结果

试运行完成后，查看结果：

```powershell
python view_tuning_results.py
```

## 📊 预期输出

### 训练过程输出
```
======================================================================
网格搜索: 共 4 个配置组合
每个配置在 2 个被试上进行 LOSO 交叉验证
总计需要训练: 8 个模型
======================================================================

[配置 1/4]
参数: {
  "epochs": 30,
  "lr": 0.001,
  ...
}
  被试 1/2... 验证=0.7854, 测试=0.7456, 时间=2.3min
  被试 2/2... 验证=0.8123, 测试=0.7689, 时间=2.1min
  平均验证准确率: 0.7989
  平均测试准确率: 0.7573 ± 0.0165
  ...
```

### 结果文件
```
hyperparameter_tuning_results/
└── hyperparam_tuning_20241107_162345/
    ├── best_config.json           ← 最佳配置
    ├── results_summary.csv        ← Excel可打开
    ├── REPORT.txt                 ← 完整报告
    ├── ranking.png                ← 性能排名图
    ├── param_effects.png          ← 参数影响图
    └── config_001/, config_002/   ← 详细结果
```

## 🎯 下一步行动

### 如果快速测试成功：

**选项A: 进行更全面的搜索**
1. 编辑 `hyperparam_config_quick.yaml`
2. 增加被试数量到4-5个
3. 增加参数值（例如测试更多学习率）
4. 重新运行 `python run_tuning.py`

**选项B: 使用完整配置**
1. 确保有充足时间（1-2天）
2. 修改 `hyperparam_config.yaml` 减少参数组合
3. 运行完整搜索

### 推荐的渐进策略

#### 第一轮：粗搜索（4-6小时）
```yaml
# hyperparam_config_coarse.yaml
n_subjects: 9
hyperparameters:
  epochs: {values: [100, 200]}
  lr: {values: [0.0005, 0.001]}
  gamma: {values: [8.0, 10.0]}
  batch_size: {values: [64]}
  label_weight: {values: [1.0, 1.5]}
  domain_weight: {values: [0.5, 1.0]}
# 2×2×2×1×2×2 = 32 个配置
```

#### 第二轮：细搜索（在最优区域）
假设第一轮发现最优为: lr=0.001, gamma=10.0

```yaml
hyperparameters:
  epochs: {values: [150, 200, 250]}
  lr: {values: [0.0008, 0.001, 0.0012]}
  gamma: {values: [9.0, 10.0, 11.0]}
  batch_size: {values: [64, 128]}
  label_weight: {values: [1.5, 2.0]}
  domain_weight: {values: [0.5, 0.75]}
```

## 💡 常见问题

### Q1: 如何中断训练？
**A:** 按 `Ctrl+C`。中间结果会自动保存，可以查看已完成的配置结果。

### Q2: 如何查看正在运行的实验？
**A:** 结果实时保存在 `hyperparameter_tuning_results/` 目录。可以打开 `all_results.json` 查看。

### Q3: 训练太慢怎么办？
**A:** 
1. 减少 epochs（例如从200改到100）
2. 减少参数组合数量
3. 使用早停（已默认启用）
4. 先测试少数被试（修改 n_subjects）

### Q4: 如何使用找到的最佳配置？
**A:** 查看 `best_config.json`，然后修改 `main_DANN_EEGNet_BCI2b_LOSO.py` 中的参数：

```python
train_conf = {
    'batch_size': 64,          # 从best_config复制
    'epochs': 200,             # 从best_config复制
    'lr': 0.001,               # 从best_config复制
    'n_train': 1,
}
```

### Q5: 如何对比不同实验结果？
**A:** 运行 `view_tuning_results.py`，它会自动找到最新实验。要对比多个实验，手动打开不同目录的 `REPORT.txt`。

## 📈 性能目标

- **基准（标准EEGNet）**: 76.11% ± 8.2%
- **初始DANN**: 75.34% ± 7.9%
- **优化目标**: ≥ 77.0%

如果调优后准确率仍低于基准，可能需要：
1. 尝试更多epoch（250-300）
2. 调整损失权重（favor label learning）
3. 使用不同的gamma调度策略
4. 考虑数据增强或其他技巧

## 🔧 高级用法

### 修改优化目标
默认优化测试准确率。要优化其他指标，修改 `hyperparameter_tuning.py`:

```python
# 在 run_grid_search() 中，修改这一行:
if avg_test_acc > self.best_score:  # 改为 avg_test_kappa

# 同时修改排序:
df_sorted = df.sort_values('avg_test_acc', ...)  # 改为对应列
```

### 添加新的超参数
在 YAML 配置中添加：

```yaml
hyperparameters:
  dropout:  # 新参数
    values: [0.25, 0.3, 0.4]
    description: "Dropout率"
```

然后在 `models_dann.py` 的 `build_dann_eegnet()` 中使用该参数。

### 使用不同的调度策略
修改 `DomainAdaptationSchedule` 类实现新的lambda调度。

## 📞 获取帮助

遇到问题时：
1. 检查 `test_hyperparameter_system.py` 是否全部通过
2. 查看详细文档 `README_HYPERPARAMETER_TUNING.md`
3. 检查错误日志和traceback
4. 确认数据路径正确

## ✅ 检查清单

开始调优前确认：
- [ ] 测试脚本全部通过（6/6）
- [ ] 数据路径正确（C:/Users/35696/Desktop/BCI_2b/）
- [ ] 有足够磁盘空间（建议≥10GB）
- [ ] 配置文件符合预期
- [ ] 了解预计时间

祝调优顺利！ 🎉
