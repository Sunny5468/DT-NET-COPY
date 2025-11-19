# DANN-EEGNet for BCI Competition IV-2b

基于域对抗神经网络（Domain Adversarial Neural Network, DANN）的 EEGNet 实现，用于脑机接口的跨被试迁移学习。

## 📋 目录结构

```
EEGNet_BCI2b/
├── models.py                          # 原始 EEGNet 模型
├── models_dann.py                     # DANN-EEGNet 模型（新增）
├── main_EEGNet_BCI2b_LOSO.py         # 标准 EEGNet 训练脚本
├── main_DANN_EEGNet_BCI2b_LOSO.py    # DANN-EEGNet 训练脚本（新增）
├── preprocess.py                      # 数据预处理
└── README_DANN.md                     # 本文档
```

## 🎯 DANN 原理

### 什么是 DANN？

Domain Adversarial Neural Network（域对抗神经网络）是一种用于域适应的深度学习方法，目标是学习**域不变特征表示**，使模型能够从源域（有标签数据）迁移到目标域（无标签或少量标签数据）。

### 核心思想

1. **特征提取器（Feature Extractor）**：提取共享特征
2. **标签分类器（Label Classifier）**：预测任务标签（如左手/右手）
3. **域分类器（Domain Classifier）**：预测数据来自哪个域（哪个被试）
4. **梯度反转层（Gradient Reversal Layer, GRL）**：
   - 前向传播：正常传递特征
   - 反向传播：反转梯度（乘以 -λ）
   - 效果：特征提取器学习**混淆域分类器**的特征

### 训练目标

```
min_θf,θy max_θd  L_label - λ * L_domain

其中:
- θf: 特征提取器参数
- θy: 标签分类器参数  
- θd: 域分类器参数
- λ: 梯度反转强度（从 0 逐渐增加到 1）
```

### 在 BCI 中的应用

- **源域**：8 个被试的 EEG 数据（有标签）
- **目标域**：第 9 个被试的 EEG 数据（测试时用）
- **目标**：学习与被试无关的运动想象特征

## 🏗️ DANN-EEGNet 架构

```
输入 EEG 信号 (1, 3, 1125)
         ↓
    [EEGNet 特征提取器]
    - 时域卷积 (F1=8)
    - 深度可分离卷积 (D=2)
    - 可分离卷积 (F2=16)
         ↓
      Flatten
    ┌─────┴─────┐
    ↓           ↓
[标签分类器]  [GRL] → [域分类器]
    ↓                    ↓
左手/右手           被试1-8
(2类)              (8类)
```

### 关键组件

1. **GradientReversalLayer**
```python
@tf.custom_gradient
def gradient_reversal(x, lambda_factor):
    def grad_fn(dy):
        return -lambda_factor * dy, None  # 反转梯度
    return x, grad_fn
```

2. **Lambda 调度**
```python
λ(p) = 2 / (1 + exp(-10*p)) - 1
其中 p = current_epoch / total_epochs
```
- Epoch 0: λ ≈ 0（域分类器不工作）
- Epoch 50: λ ≈ 0.9（逐渐对抗）
- Epoch 100: λ ≈ 1.0（完全对抗）

## 🚀 使用方法

### 1. 测试模型构建

```bash
cd EEGNet_BCI2b
python models_dann.py
```

输出：
- DANN 完整模型结构
- 特征提取器结构
- 标签分类器结构

### 2. 训练 DANN-EEGNet

```bash
python main_DANN_EEGNet_BCI2b_LOSO.py
```

训练过程：
- 对每个目标被试，使用其他 8 个被试作为源域
- 训练标签分类器和域分类器
- 动态调整 λ 参数
- 在目标被试上测试泛化性能

### 3. 对比标准 EEGNet

```bash
# 标准 EEGNet (LOSO)
python main_EEGNet_BCI2b_LOSO.py

# DANN-EEGNet (LOSO)
python main_DANN_EEGNet_BCI2b_LOSO.py
```

## 📊 预期效果

### DANN vs 标准 EEGNet

| 方法 | 优势 | 劣势 |
|------|------|------|
| **标准 EEGNet** | - 简单直接<br>- 训练快速 | - 被试间差异大<br>- 泛化能力弱 |
| **DANN-EEGNet** | - 跨被试泛化更好<br>- 学习域不变特征 | - 训练更复杂<br>- 需要调整 λ |

### 性能提升

根据相关文献，DANN 在跨被试 BCI 任务中通常能带来：
- 平均准确率提升：**2-5%**
- Kappa 系数提升：**0.05-0.10**
- 对"困难被试"改善更明显

## 🔧 超参数调优

### 关键参数

```python
train_conf = {
    'batch_size': 64,        # 批大小
    'epochs': 100,           # 训练轮数（DANN 通常不需要太多）
    'lr': 0.001,             # 学习率
    'n_train': 1,            # 每个被试训练次数
}

# Lambda 调度
lambda_schedule = DomainAdaptationSchedule(
    total_epochs=100,
    gamma=10.0               # 控制 λ 增长速度
)
```

### 调优建议

1. **Epochs**：
   - 太少（<50）：域对抗不充分
   - 太多（>200）：可能过拟合
   - 推荐：80-120

2. **Gamma**：
   - 小（5）：λ 增长慢，对抗温和
   - 大（15）：λ 增长快，对抗激进
   - 推荐：8-12

3. **域分类器复杂度**：
   - 当前：2 层 Dense(256) + Dropout(0.5)
   - 可调整为：1-3 层，128-512 神经元

4. **Loss 权重**：
   ```python
   loss_weights={
       'label_output': 1.0,    # 标签损失权重
       'domain_output': 1.0    # 域损失权重（可调整为 0.5-2.0）
   }
   ```

## 📈 结果分析

### 查看训练日志

```bash
# 标准 EEGNet 结果
cat results_EEGNet_BCI2b_LOSO/log.txt

# DANN-EEGNet 结果
cat results_DANN_EEGNet_BCI2b_LOSO/log.txt
```

### 日志内容

```
Epoch 1: lambda = 0.0909
[Sub 1 Seed 1] Epoch 001 label_loss=0.6932 domain_loss=2.0794 ...

Target Sub: 1, Run: 1, Time: 5.2m, Test Acc: 0.7250, Test Kappa: 0.4500
...
Overall Average Accuracy: 72.50%
Overall Average Kappa: 0.4500
```

### 可视化（可选扩展）

```python
# 绘制 lambda 曲线
import matplotlib.pyplot as plt
epochs = range(100)
lambdas = [lambda_schedule.get_lambda(e) for e in epochs]
plt.plot(epochs, lambdas)
plt.xlabel('Epoch')
plt.ylabel('Lambda')
plt.title('Lambda Schedule')
plt.show()
```

## 🔬 进阶功能

### 1. 半监督 DANN

如果目标域有少量标签数据：

```python
# 在 train_dann() 中添加
X_target_labeled, y_target_labeled = ...  # 少量标签数据

# 混合训练
X_mixed = np.concatenate([X_source, X_target_labeled])
y_mixed = np.concatenate([y_source, y_target_labeled])
```

### 2. 多源域集成

```python
# 训练多个模型，集成预测
predictions = []
for model in trained_models:
    pred = model.predict(X_test)
    predictions.append(pred)

# 投票或平均
final_pred = np.mean(predictions, axis=0)
```

### 3. 域混淆损失（可选）

除了对抗训练，还可以添加域混淆损失：

```python
from keras.losses import categorical_crossentropy

def domain_confusion_loss(y_true, y_pred):
    """域混淆：最大化域预测的熵"""
    uniform = tf.ones_like(y_pred) / n_domains
    return -categorical_crossentropy(uniform, y_pred)
```

## 📚 参考文献

1. **DANN 原始论文**:
   Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. In ICML.

2. **EEGNet 原始论文**:
   Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. Journal of neural engineering, 15(5), 056013.

3. **BCI 域适应综述**:
   He, H., & Wu, D. (2020). Transfer learning for brain–computer interfaces: A Euclidean space data alignment approach. IEEE Transactions on Biomedical Engineering, 67(2), 399-410.

## 🐛 故障排查

### 问题 1：域分类器准确率过高（>95%）

**原因**：域特征过于明显，对抗不充分

**解决**：
- 增加 λ 的 gamma 参数
- 减少域分类器复杂度
- 增加 domain_loss 权重

### 问题 2：标签分类器性能下降

**原因**：域对抗过强，破坏了判别特征

**解决**：
- 减小 lambda 或 gamma
- 减小 domain_loss 权重（如 0.5）
- 增加 label_loss 权重（如 1.5）

### 问题 3：训练不稳定

**原因**：梯度冲突

**解决**：
- 降低学习率
- 使用梯度裁剪
- 先预训练标签分类器，再加入域对抗

## 💡 最佳实践

1. **先训练标准 EEGNet 作为基线**
2. **逐步引入域对抗**（先小 λ，观察效果）
3. **监控两个损失的平衡**
4. **对每个被试分析改善程度**
5. **可视化特征分布**（使用 t-SNE）

## 📞 联系与贡献

如有问题或改进建议，欢迎：
- 提交 Issue
- 发起 Pull Request
- 参考 EEG-DCNet 原始实现

---

**注意**：DANN 训练时间约为标准 EEGNet 的 1.5-2 倍，但能显著提升跨被试泛化能力。
