"""
Chess AI Kiro 源代码模块

包含三个主要子系统：
- chess_board_recognition: 棋盘识别系统
- chinese_chess_ai_engine: 象棋AI引擎
- real_time_analysis_system: 实时分析系统
"""

from . import chess_board_recognition
from . import chinese_chess_ai_engine
from . import real_time_analysis_system

__all__ = [
    "chess_board_recognition",
    "chinese_chess_ai_engine",
    "real_time_analysis_system",
]