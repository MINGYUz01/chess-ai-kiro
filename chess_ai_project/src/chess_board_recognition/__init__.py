"""
棋盘识别系统 - 基于YOLO11的中国象棋棋局识别

该模块提供完整的棋盘识别解决方案，包括：
- 数据收集和标注支持
- YOLO11模型训练
- 实时棋局识别和状态输出
- 系统配置和监控
"""

__version__ = "0.1.0"
__author__ = "Chess AI Kiro Team"

# 导入核心组件
from .core.interfaces import (
    ChessboardDetector,
    Detection,
    BoardState,
    ChessboardRecognitionError,
    ModelLoadError,
    InferenceError,
    DataValidationError,
)

from .core.config import ConfigManager
from .core.logger import setup_logger

__all__ = [
    # 核心接口
    "ChessboardDetector",
    "Detection", 
    "BoardState",
    # 异常类
    "ChessboardRecognitionError",
    "ModelLoadError", 
    "InferenceError",
    "DataValidationError",
    # 工具类
    "ConfigManager",
    "setup_logger",
]