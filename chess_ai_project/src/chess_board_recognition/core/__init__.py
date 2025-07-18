"""
棋盘识别系统核心模块

包含系统的核心接口、数据结构和基础功能。
"""

from .interfaces import (
    ChessboardDetector,
    Detection,
    BoardState,
    ChessboardRecognitionError,
    ModelLoadError,
    InferenceError,
    DataValidationError,
)

from .config import ConfigManager
from .logger import setup_logger

__all__ = [
    "ChessboardDetector",
    "Detection",
    "BoardState", 
    "ChessboardRecognitionError",
    "ModelLoadError",
    "InferenceError", 
    "DataValidationError",
    "ConfigManager",
    "setup_logger",
]