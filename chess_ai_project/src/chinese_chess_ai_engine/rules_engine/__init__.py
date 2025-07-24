"""
象棋规则引擎模块

包含棋局表示、走法生成、规则验证等核心功能。
"""

from .chess_board import ChessBoard
from .move import Move
# RuleEngine将在后续任务中实现
# from .rule_engine import RuleEngine

__all__ = ['ChessBoard', 'Move']