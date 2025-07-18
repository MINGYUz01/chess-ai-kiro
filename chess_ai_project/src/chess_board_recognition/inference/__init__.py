"""
推理引擎模块

提供棋局检测和识别功能，包括：
- 模型加载和推理
- 图像预处理和后处理
- 结果转换和验证
- 棋局状态输出
"""

from .chessboard_detector import ChessboardDetectorImpl
from .result_processor import ResultProcessor

__all__ = [
    "ChessboardDetectorImpl",
    "ResultProcessor",
]