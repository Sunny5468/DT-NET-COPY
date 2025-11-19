"""
DANN (Domain Adversarial Neural Network) implementation for EEGNet
用于 BCI Competition IV-2b 数据集的跨被试迁移学习

参考文献:
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation.
  In International conference on machine learning (pp. 1180-1189). PMLR.
"""

import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda


@tf.custom_gradient
def gradient_reversal(x, lambda_factor):
    """
    梯度反转函数
    前向传播时正常传递，反向传播时翻转梯度
    
    Parameters
    ----------
    x : tensor
        输入张量
    lambda_factor : float
        梯度反转强度（通常随训练增加）
        
    Returns
    -------
    x : tensor
        前向传播输出（不变）
    grad_fn : function
        自定义梯度函数
    """
    def grad_fn(dy):
        return -lambda_factor * dy, None
    
    return x, grad_fn


class GradientReversalLayer(Layer):
    """
    梯度反转层 (Gradient Reversal Layer)
    
    在前向传播时作为恒等映射
    在反向传播时反转梯度并乘以 lambda 系数
    
    Parameters
    ----------
    lambda_factor : float
        梯度反转强度，通常在训练过程中逐渐增加
        论文建议: lambda_p = 2 / (1 + exp(-10 * p)) - 1, 其中 p = epoch / total_epochs
    """
    def __init__(self, lambda_factor=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_factor = lambda_factor
    
    def call(self, inputs):
        return gradient_reversal(inputs, self.lambda_factor)
    
    def get_config(self):
        config = super().get_config()
        config.update({'lambda_factor': self.lambda_factor})
        return config


class DomainAdaptationSchedule:
    """
    DANN 训练过程中的 lambda 调度器
    lambda 从 0 逐渐增加到 1，控制域对抗的强度
    """
    def __init__(self, total_epochs, gamma=10.0):
        """
        Parameters
        ----------
        total_epochs : int
            总训练轮数
        gamma : float
            调节增长速度的参数（默认10，论文推荐值）
        """
        self.total_epochs = total_epochs
        self.gamma = gamma
    
    def get_lambda(self, current_epoch):
        """
        计算当前 epoch 的 lambda 值
        
        Formula: lambda_p = 2 / (1 + exp(-gamma * p)) - 1
        其中 p = current_epoch / total_epochs
        """
        p = current_epoch / self.total_epochs
        lambda_p = 2.0 / (1.0 + tf.exp(-self.gamma * p)) - 1.0
        return float(lambda_p)


def build_dann_eegnet(n_classes=2, n_channels=3, in_samples=1125, 
                      n_domains=8, lambda_factor=1.0,
                      F1=8, D=2, kernLength=64, dropout=0.25):
    """
    构建带有域对抗训练的 EEGNet 模型 (DANN-EEGNet)
    
    架构：
    1. 特征提取器 (Feature Extractor): EEGNet 的卷积层部分
    2. 标签分类器 (Label Classifier): 用于运动想象分类
    3. 域分类器 (Domain Classifier): 用于区分数据来自哪个被试（域）
    
    训练目标：
    - 标签分类器：最小化标签预测误差
    - 域分类器：最小化域预测误差（通过梯度反转，实际是最大化）
    - 整体：学习域不变的特征表示
    
    Parameters
    ----------
    n_classes : int
        运动想象类别数（2: 左手/右手）
    n_channels : int
        EEG 通道数
    in_samples : int
        时间采样点数
    n_domains : int
        域的数量（对于 LOSO，这是训练集中的被试数量，通常是 8）
    lambda_factor : float
        梯度反转强度（训练时动态调整）
    F1, D, kernLength, dropout : 
        EEGNet 参数
        
    Returns
    -------
    model : keras.Model
        DANN-EEGNet 模型，有两个输出：[label_output, domain_output]
    feature_extractor : keras.Model
        特征提取器（用于测试时只提取特征）
    """
    from models import EEGNet
    
    # 输入层
    input_data = Input(shape=(1, n_channels, in_samples), name='input_data')
    
    # 数据预处理（与原始 EEGNet 一致）
    from keras.layers import Permute
    x = Permute((3, 2, 1))(input_data)
    
    # ===== 特征提取器 (Shared Feature Extractor) =====
    features = EEGNet(input_layer=x, F1=F1, kernLength=kernLength, 
                     D=D, Chans=n_channels, dropout=dropout)
    
    from keras.layers import Flatten
    features_flat = Flatten(name='features')(features)
    
    # ===== 标签分类器 (Label Classifier) =====
    # 用于运动想象任务的分类（左手 vs 右手）
    from keras.layers import Dense, Activation
    from keras.constraints import max_norm
    
    label_classifier = Dense(n_classes, name='label_dense', 
                            kernel_constraint=max_norm(0.25))(features_flat)
    label_output = Activation('softmax', name='label_output')(label_classifier)
    
    # ===== 域分类器 (Domain Classifier) =====
    # 通过梯度反转层，学习域不变特征
    grl = GradientReversalLayer(lambda_factor=lambda_factor, name='grl')(features_flat)
    
    # 域分类器网络（通常比标签分类器深一些）
    domain_hidden = Dense(256, activation='relu', name='domain_hidden1')(grl)
    domain_hidden = Dropout(0.5)(domain_hidden)
    domain_hidden = Dense(256, activation='relu', name='domain_hidden2')(domain_hidden)
    domain_hidden = Dropout(0.5)(domain_hidden)
    domain_classifier = Dense(n_domains, name='domain_dense')(domain_hidden)
    domain_output = Activation('softmax', name='domain_output')(domain_classifier)
    
    # ===== 构建模型 =====
    # 完整 DANN 模型（用于训练）
    dann_model = Model(inputs=input_data, 
                      outputs=[label_output, domain_output],
                      name='DANN_EEGNet')
    
    # 特征提取器（用于测试/推理）
    feature_extractor = Model(inputs=input_data, 
                             outputs=features_flat,
                             name='Feature_Extractor')
    
    # 仅标签分类器（用于测试）
    label_classifier_model = Model(inputs=input_data,
                                   outputs=label_output,
                                   name='Label_Classifier')
    
    return dann_model, feature_extractor, label_classifier_model


def get_dann_model(n_classes=2, n_channels=3, in_samples=1125, n_domains=8):
    """
    便捷函数：创建 DANN-EEGNet 模型
    
    Parameters
    ----------
    n_classes : int
        类别数
    n_channels : int
        通道数
    in_samples : int
        采样点数
    n_domains : int
        域的数量（训练集被试数）
        
    Returns
    -------
    dann_model : keras.Model
        完整 DANN 模型
    feature_extractor : keras.Model
        特征提取器
    label_classifier : keras.Model
        标签分类器
    """
    return build_dann_eegnet(
        n_classes=n_classes,
        n_channels=n_channels,
        in_samples=in_samples,
        n_domains=n_domains,
        lambda_factor=0.0,  # 初始为 0，训练时动态调整
        F1=8,
        D=2,
        kernLength=64,
        dropout=0.25
    )


if __name__ == "__main__":
    # 测试模型构建
    print("="*60)
    print("DANN-EEGNet 模型测试")
    print("="*60)
    
    # 为 LOSO 构建模型（8 个源域被试）
    dann_model, feature_extractor, label_classifier = get_dann_model(
        n_classes=2,
        n_channels=3,
        in_samples=1125,
        n_domains=8  # LOSO: 9个被试中的8个作为源域
    )
    
    print("\n1. DANN 完整模型:")
    dann_model.summary()
    
    print("\n2. 特征提取器:")
    feature_extractor.summary()
    
    print("\n3. 标签分类器:")
    label_classifier.summary()
    
    print("\n模型构建成功!")
