"""
更新导入路径脚本
将脚本中的导入路径更新为新的模块结构
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 现在可以使用新的导入方式
# from models import models, models_dann
# from utils import preprocess
