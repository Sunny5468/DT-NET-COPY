"""
EEGNet 模型模块
包含标准EEGNet和DANN-EEGNet模型定义
"""

from .models import EEGNet
from .models_dann import (
    get_dann_model,
    build_dann_eegnet,
    GradientReversalLayer,
    DomainAdaptationSchedule
)

__all__ = [
    'EEGNet',
    'get_dann_model',
    'build_dann_eegnet',
    'GradientReversalLayer',
    'DomainAdaptationSchedule'
]
