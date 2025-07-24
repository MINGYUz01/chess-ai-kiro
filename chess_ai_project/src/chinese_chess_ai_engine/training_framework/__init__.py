"""
训练框架模块

包含自对弈数据生成、模型训练和评估功能。
"""

from .self_play_generator import SelfPlayGenerator
from .trainer import Trainer
from .model_evaluator import ModelEvaluator
from .training_example import TrainingExample

__all__ = ['SelfPlayGenerator', 'Trainer', 'ModelEvaluator', 'TrainingExample']