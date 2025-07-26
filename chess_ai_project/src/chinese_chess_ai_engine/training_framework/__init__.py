"""
训练框架模块

包含自对弈数据生成、模型训练和评估功能。
"""

from .training_example import TrainingExample, TrainingDataset
from .board_encoder import BoardEncoder
from .self_play_generator import SelfPlayGenerator, SelfPlayConfig, GameResult
from .training_config import TrainingConfig, EvaluationConfig
from .trainer import Trainer, ChessTrainingDataset, EarlyStopping, ModelEMA
from .model_evaluator import ModelEvaluator, EvaluationResult, BenchmarkPosition, ELOCalculator

__all__ = [
    'TrainingExample', 
    'TrainingDataset',
    'BoardEncoder',
    'SelfPlayGenerator', 
    'SelfPlayConfig',
    'GameResult',
    'TrainingConfig',
    'EvaluationConfig',
    'Trainer',
    'ChessTrainingDataset',
    'EarlyStopping',
    'ModelEMA',
    'ModelEvaluator',
    'EvaluationResult',
    'BenchmarkPosition',
    'ELOCalculator'
]