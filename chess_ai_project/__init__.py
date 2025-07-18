"""
中国象棋AI系统 (Chess AI Kiro)

一个基于深度学习的中国象棋AI系统，包含棋盘识别、AI引擎和实时分析功能。
"""

__version__ = "0.1.0"
__author__ = "Chess AI Kiro Team"
__email__ = "team@chess-ai-kiro.com"
__description__ = "中国象棋AI系统 - 基于深度学习的棋盘识别、AI引擎和实时分析系统"

# 导入主要模块
from chess_ai_project.src import (
    chess_board_recognition,
    chinese_chess_ai_engine,
    real_time_analysis_system,
)

__all__ = [
    "chess_board_recognition",
    "chinese_chess_ai_engine", 
    "real_time_analysis_system",
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]