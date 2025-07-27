"""
推理接口模块

包含AI分析、游戏接口和API服务器。
"""

from .chess_ai import ChessAI, AnalysisResult, AIConfig, DifficultyManager
from .game_interface import (
    GameInterface, GameSession, GameConfig, MoveRecord,
    GameState, GameResult, PlayerType,
    GameInterfaceError, SessionNotFoundError, InvalidMoveError, GameStateError
)

from .api_server import APIServer, create_api_server

__all__ = [
    # AI核心
    'ChessAI',
    'AnalysisResult', 
    'AIConfig',
    'DifficultyManager',
    
    # 游戏接口
    'GameInterface',
    'GameSession',
    'GameConfig',
    'MoveRecord',
    
    # API服务器
    'APIServer',
    'create_api_server',
    
    # 枚举类型
    'GameState',
    'GameResult',
    'PlayerType',
    
    # 异常类
    'GameInterfaceError',
    'SessionNotFoundError',
    'InvalidMoveError',
    'GameStateError'
]