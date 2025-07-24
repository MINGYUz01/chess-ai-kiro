"""
模型训练模块

提供YOLO11模型训练功能，包括：
- 模型训练和验证
- 训练参数配置和验证
- 训练进度监控和指标记录
- 模型权重自动保存
- 超参数优化
- 模型评估和报告生成
- 模型导出和格式转换
"""

from .trainer import YOLO11Trainer
from .config_validator import TrainingConfigValidator, DataConfigGenerator
from .monitor import TrainingMonitor
from .hyperparameter_optimizer import HyperparameterOptimizer
from .evaluator import ModelEvaluator
from .model_exporter import ModelExporter

__all__ = [
    "YOLO11Trainer",
    "TrainingConfigValidator",
    "DataConfigGenerator",
    "TrainingMonitor",
    "HyperparameterOptimizer",
    "ModelEvaluator",
    "ModelExporter",
]