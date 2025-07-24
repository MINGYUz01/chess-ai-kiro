"""
中国象棋AI引擎

基于深度强化学习的高性能象棋对弈系统，采用AlphaZero架构结合蒙特卡洛树搜索(MCTS)算法。
包括规则引擎、神经网络模型、搜索算法、训练框架和推理接口。
"""

__version__ = "1.0.0"
__author__ = "Chess AI Team"

# 导入核心组件
from .rules_engine import ChessBoard, Move
from .config import ConfigManager, MCTSConfig, TrainingConfig, ModelConfig, AIConfig
from .utils import setup_logger, get_logger, ChessAIError

# 其他组件将在后续实现时导入
# from .rules_engine import RuleEngine
# from .neural_network import ChessNet, ModelManager, InferenceEngine
# from .search_algorithm import MCTSNode, MCTSSearcher, ParallelSearcher
# from .training_framework import SelfPlayGenerator, Trainer, ModelEvaluator
# from .inference_interface import ChessAI, GameInterface, APIServer

__all__ = [
    "__version__", "__author__",
    "ChessBoard", "Move",
    "ConfigManager", "MCTSConfig", "TrainingConfig", "ModelConfig", "AIConfig",
    "setup_logger", "get_logger", "ChessAIError"
]