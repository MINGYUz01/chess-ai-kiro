"""
游戏接口和会话管理

提供完整的对弈会话管理功能，包括游戏状态跟踪、历史记录、
走法验证和异常处理机制。
"""

import time
import uuid
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..rules_engine import ChessBoard, Move
from .chess_ai import ChessAI, AnalysisResult, AIConfig


class GameState(Enum):
    """游戏状态枚举"""
    WAITING = "waiting"          # 等待开始
    PLAYING = "playing"          # 对局进行中
    PAUSED = "paused"           # 暂停
    FINISHED = "finished"       # 已结束
    ERROR = "error"             # 错误状态


class GameResult(Enum):
    """游戏结果枚举"""
    ONGOING = "ongoing"         # 进行中
    RED_WIN = "red_win"         # 红方胜
    BLACK_WIN = "black_win"     # 黑方胜
    DRAW = "draw"               # 和棋
    TIMEOUT = "timeout"         # 超时
    RESIGN = "resign"           # 认输
    ERROR = "error"             # 错误结束


class PlayerType(Enum):
    """玩家类型枚举"""
    HUMAN = "human"             # 人类玩家
    AI = "ai"                   # AI玩家


@dataclass
class GameConfig:
    """游戏配置"""
    # 基本设置
    red_player_type: PlayerType = PlayerType.HUMAN
    black_player_type: PlayerType = PlayerType.AI
    ai_difficulty: int = 5
    
    # 时间控制
    time_limit_per_move: Optional[float] = None  # 每步时间限制（秒）
    total_time_limit: Optional[float] = None     # 总时间限制（秒）
    
    # AI设置
    ai_config: Optional[AIConfig] = None
    
    # 游戏规则
    allow_undo: bool = True
    max_moves: int = 300        # 最大回合数
    draw_by_repetition: bool = True  # 重复局面和棋
    
    # 记录设置
    save_game_record: bool = True
    record_analysis: bool = False  # 是否记录每步分析
    
    def __post_init__(self):
        """初始化后处理"""
        if self.ai_config is None:
            self.ai_config = AIConfig(
                model_path="models/chess_model.pth",
                difficulty_level=self.ai_difficulty
            )


@dataclass
class MoveRecord:
    """走法记录"""
    move: Move
    player: int                 # 1=红方, -1=黑方
    timestamp: datetime
    time_used: float           # 用时（秒）
    analysis: Optional[AnalysisResult] = None
    comment: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'move': self.move.to_dict(),
            'player': self.player,
            'timestamp': self.timestamp.isoformat(),
            'time_used': self.time_used,
            'analysis': self.analysis.to_dict() if self.analysis else None,
            'comment': self.comment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MoveRecord':
        """从字典创建"""
        return cls(
            move=Move.from_dict(data['move']),
            player=data['player'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            time_used=data['time_used'],
            analysis=AnalysisResult.from_dict(data['analysis']) if data['analysis'] else None,
            comment=data.get('comment', '')
        )


@dataclass
class GameSession:
    """游戏会话"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: GameConfig = field(default_factory=GameConfig)
    board: ChessBoard = field(default_factory=ChessBoard)
    
    # 游戏状态
    state: GameState = GameState.WAITING
    result: GameResult = GameResult.ONGOING
    current_player: int = 1     # 1=红方, -1=黑方
    
    # 时间管理
    start_time: Optional[datetime] = None
    red_time_used: float = 0.0
    black_time_used: float = 0.0
    last_move_time: Optional[datetime] = None
    
    # 历史记录
    move_history: List[MoveRecord] = field(default_factory=list)
    board_history: List[str] = field(default_factory=list)  # FEN格式的棋局历史
    
    # 统计信息
    total_moves: int = 0
    repetition_count: Dict[str, int] = field(default_factory=dict)
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        # 记录初始棋局
        self.board_history.append(self.board.to_fen())
        self.repetition_count[self.board.to_fen()] = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'config': {
                'red_player_type': self.config.red_player_type.value,
                'black_player_type': self.config.black_player_type.value,
                'ai_difficulty': self.config.ai_difficulty,
                'time_limit_per_move': self.config.time_limit_per_move,
                'total_time_limit': self.config.total_time_limit,
                'allow_undo': self.config.allow_undo,
                'max_moves': self.config.max_moves,
                'draw_by_repetition': self.config.draw_by_repetition,
                'save_game_record': self.config.save_game_record,
                'record_analysis': self.config.record_analysis
            },
            'board': self.board.to_fen(),
            'state': self.state.value,
            'result': self.result.value,
            'current_player': self.current_player,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'red_time_used': self.red_time_used,
            'black_time_used': self.black_time_used,
            'move_history': [record.to_dict() for record in self.move_history],
            'board_history': self.board_history,
            'total_moves': self.total_moves,
            'repetition_count': self.repetition_count,
            'created_at': self.created_at.isoformat(),
            'finished_at': self.finished_at.isoformat() if self.finished_at else None,
            'metadata': self.metadata
        }


class GameInterface:
    """游戏接口和会话管理器"""
    
    def __init__(self, ai_model_path: Optional[str] = None, 
                 save_directory: Optional[str] = None):
        """
        初始化游戏接口
        
        Args:
            ai_model_path: AI模型路径
            save_directory: 游戏记录保存目录
        """
        self.logger = logging.getLogger(__name__)
        
        # AI引擎
        self.ai_model_path = ai_model_path or "models/chess_model.pth"
        self.ai_engine: Optional[ChessAI] = None
        
        # 会话管理
        self.current_session: Optional[GameSession] = None
        self.sessions: Dict[str, GameSession] = {}
        
        # 保存设置
        self.save_directory = Path(save_directory) if save_directory else Path("data/sessions")
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"游戏接口初始化完成，保存目录: {self.save_directory}")
    
    def _initialize_ai(self, config: AIConfig) -> ChessAI:
        """初始化AI引擎"""
        try:
            if self.ai_engine is None or self.ai_engine.config != config:
                self.ai_engine = ChessAI(self.ai_model_path, config)
                self.logger.info(f"AI引擎初始化完成，难度级别: {config.difficulty_level}")
            return self.ai_engine
        except Exception as e:
            self.logger.error(f"AI引擎初始化失败: {e}")
            raise
    
    def create_session(self, config: Optional[GameConfig] = None) -> str:
        """
        创建新的游戏会话
        
        Args:
            config: 游戏配置
            
        Returns:
            会话ID
        """
        try:
            if config is None:
                config = GameConfig()
            
            session = GameSession(config=config)
            
            # 初始化AI引擎（如果需要）
            if (config.red_player_type == PlayerType.AI or 
                config.black_player_type == PlayerType.AI):
                self._initialize_ai(config.ai_config)
            
            self.sessions[session.session_id] = session
            self.current_session = session
            
            self.logger.info(f"创建新会话: {session.session_id}")
            return session.session_id
            
        except Exception as e:
            self.logger.error(f"创建会话失败: {e}")
            raise
    
    def start_game(self, session_id: Optional[str] = None) -> bool:
        """
        开始游戏
        
        Args:
            session_id: 会话ID，如果为None则使用当前会话
            
        Returns:
            是否成功开始
        """
        try:
            session = self._get_session(session_id)
            
            if session.state != GameState.WAITING:
                raise ValueError(f"游戏状态错误: {session.state}")
            
            session.state = GameState.PLAYING
            session.start_time = datetime.now()
            session.last_move_time = session.start_time
            
            self.logger.info(f"游戏开始: {session.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"开始游戏失败: {e}")
            return False
    
    def make_move(self, move: Union[Move, str, Dict], 
                  session_id: Optional[str] = None,
                  analysis: bool = False) -> Tuple[bool, str]:
        """
        执行走法
        
        Args:
            move: 走法（Move对象、坐标记法字符串或字典）
            session_id: 会话ID
            analysis: 是否进行分析
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            session = self._get_session(session_id)
            
            if session.state != GameState.PLAYING:
                return False, f"游戏状态错误: {session.state}"
            
            # 转换走法格式
            if isinstance(move, str):
                move = Move.from_coordinate_notation(move)
            elif isinstance(move, dict):
                move = Move.from_dict(move)
            
            # 验证轮次
            if session.current_player != session.board.current_player:
                return False, "不是当前玩家的回合"
            
            # 验证走法合法性
            legal_moves = session.board.get_legal_moves()
            if move not in legal_moves:
                return False, "非法走法"
            
            # 检查时间限制
            current_time = datetime.now()
            move_time = (current_time - session.last_move_time).total_seconds()
            
            if (session.config.time_limit_per_move and 
                move_time > session.config.time_limit_per_move):
                return False, "超时"
            
            # 执行走法
            session.board.make_move(move)
            
            # 更新时间统计
            if session.current_player == 1:
                session.red_time_used += move_time
            else:
                session.black_time_used += move_time
            
            # 记录走法
            move_analysis = None
            if analysis and session.config.record_analysis:
                try:
                    # 在执行走法前分析位置
                    temp_board = session.board.copy()
                    temp_board.undo_move()
                    move_analysis = self.ai_engine.analyze_position(temp_board)
                except Exception as e:
                    self.logger.warning(f"走法分析失败: {e}")
            
            move_record = MoveRecord(
                move=move,
                player=session.current_player,
                timestamp=current_time,
                time_used=move_time,
                analysis=move_analysis
            )
            
            session.move_history.append(move_record)
            session.total_moves += 1
            
            # 更新棋局历史
            fen = session.board.to_fen()
            session.board_history.append(fen)
            session.repetition_count[fen] = session.repetition_count.get(fen, 0) + 1
            
            # 切换玩家
            session.current_player = -session.current_player
            session.last_move_time = current_time
            
            # 检查游戏结束条件
            self._check_game_end(session)
            
            self.logger.debug(f"走法执行成功: {move.to_coordinate_notation()}")
            return True, ""
            
        except Exception as e:
            self.logger.error(f"执行走法失败: {e}")
            return False, str(e)
    
    def get_ai_move(self, session_id: Optional[str] = None,
                    time_limit: Optional[float] = None) -> Optional[Move]:
        """
        获取AI走法
        
        Args:
            session_id: 会话ID
            time_limit: 时间限制
            
        Returns:
            AI走法，如果失败返回None
        """
        try:
            session = self._get_session(session_id)
            
            if session.state != GameState.PLAYING:
                self.logger.warning(f"游戏状态错误: {session.state}")
                return None
            
            # 检查当前玩家是否为AI
            current_player_type = (session.config.red_player_type 
                                 if session.current_player == 1 
                                 else session.config.black_player_type)
            
            if current_player_type != PlayerType.AI:
                self.logger.warning("当前玩家不是AI")
                return None
            
            # 确保AI引擎已初始化
            if self.ai_engine is None:
                self._initialize_ai(session.config.ai_config)
            
            # 获取AI走法
            ai_move = self.ai_engine.get_best_move(
                session.board, 
                time_limit=time_limit or session.config.time_limit_per_move
            )
            
            if ai_move:
                self.logger.debug(f"AI走法: {ai_move.to_coordinate_notation()}")
            else:
                self.logger.warning("AI未能生成走法")
            
            return ai_move
            
        except Exception as e:
            self.logger.error(f"获取AI走法失败: {e}")
            return None
    
    def undo_move(self, session_id: Optional[str] = None) -> bool:
        """
        撤销走法
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功撤销
        """
        try:
            session = self._get_session(session_id)
            
            if not session.config.allow_undo:
                return False
            
            if not session.move_history:
                return False
            
            # 撤销棋盘走法
            session.board.undo_move()
            
            # 移除记录
            last_record = session.move_history.pop()
            session.board_history.pop()
            session.total_moves -= 1
            
            # 更新重复计数
            fen = session.board_history[-1]
            session.repetition_count[fen] -= 1
            if session.repetition_count[fen] == 0:
                del session.repetition_count[fen]
            
            # 恢复玩家和时间
            session.current_player = last_record.player
            if session.current_player == 1:
                session.red_time_used -= last_record.time_used
            else:
                session.black_time_used -= last_record.time_used
            
            # 重置游戏状态（如果已结束）
            if session.state == GameState.FINISHED:
                session.state = GameState.PLAYING
                session.result = GameResult.ONGOING
                session.finished_at = None
            
            self.logger.debug("撤销走法成功")
            return True
            
        except Exception as e:
            self.logger.error(f"撤销走法失败: {e}")
            return False 
   
    def pause_game(self, session_id: Optional[str] = None) -> bool:
        """
        暂停游戏
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功暂停
        """
        try:
            session = self._get_session(session_id)
            
            if session.state != GameState.PLAYING:
                return False
            
            session.state = GameState.PAUSED
            self.logger.info(f"游戏暂停: {session.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"暂停游戏失败: {e}")
            return False
    
    def resume_game(self, session_id: Optional[str] = None) -> bool:
        """
        恢复游戏
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功恢复
        """
        try:
            session = self._get_session(session_id)
            
            if session.state != GameState.PAUSED:
                return False
            
            session.state = GameState.PLAYING
            session.last_move_time = datetime.now()
            
            self.logger.info(f"游戏恢复: {session.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复游戏失败: {e}")
            return False
    
    def resign_game(self, player: int, session_id: Optional[str] = None) -> bool:
        """
        认输
        
        Args:
            player: 认输的玩家 (1=红方, -1=黑方)
            session_id: 会话ID
            
        Returns:
            是否成功认输
        """
        try:
            session = self._get_session(session_id)
            
            if session.state != GameState.PLAYING:
                return False
            
            session.state = GameState.FINISHED
            session.result = GameResult.BLACK_WIN if player == 1 else GameResult.RED_WIN
            session.finished_at = datetime.now()
            
            # 保存游戏记录
            if session.config.save_game_record:
                self._save_session(session)
            
            self.logger.info(f"玩家认输: {player}, 会话: {session.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"认输失败: {e}")
            return False
    
    def get_game_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取游戏状态
        
        Args:
            session_id: 会话ID
            
        Returns:
            游戏状态信息
        """
        try:
            session = self._get_session(session_id)
            
            # 获取合法走法
            legal_moves = []
            if session.state == GameState.PLAYING:
                legal_moves = [move.to_coordinate_notation() 
                             for move in session.board.get_legal_moves()]
            
            # 计算剩余时间
            remaining_time = None
            if session.config.total_time_limit:
                elapsed = (datetime.now() - session.start_time).total_seconds() if session.start_time else 0
                remaining_time = max(0, session.config.total_time_limit - elapsed)
            
            return {
                'session_id': session.session_id,
                'state': session.state.value,
                'result': session.result.value,
                'current_player': session.current_player,
                'board_fen': session.board.to_fen(),
                'board_matrix': session.board.to_matrix().tolist(),
                'legal_moves': legal_moves,
                'total_moves': session.total_moves,
                'red_time_used': session.red_time_used,
                'black_time_used': session.black_time_used,
                'remaining_time': remaining_time,
                'can_undo': len(session.move_history) > 0 and session.config.allow_undo,
                'last_move': (session.move_history[-1].move.to_coordinate_notation() 
                            if session.move_history else None),
                'repetition_count': max(session.repetition_count.values()) if session.repetition_count else 0
            }
            
        except Exception as e:
            self.logger.error(f"获取游戏状态失败: {e}")
            raise
    
    def get_move_history(self, session_id: Optional[str] = None,
                        include_analysis: bool = False) -> List[Dict[str, Any]]:
        """
        获取走法历史
        
        Args:
            session_id: 会话ID
            include_analysis: 是否包含分析信息
            
        Returns:
            走法历史列表
        """
        try:
            session = self._get_session(session_id)
            
            history = []
            for i, record in enumerate(session.move_history):
                move_info = {
                    'move_number': i + 1,
                    'move': record.move.to_coordinate_notation(),
                    'move_chinese': record.move.to_chinese_notation(),
                    'player': record.player,
                    'player_name': '红方' if record.player == 1 else '黑方',
                    'timestamp': record.timestamp.isoformat(),
                    'time_used': record.time_used,
                    'comment': record.comment
                }
                
                if include_analysis and record.analysis:
                    move_info['analysis'] = {
                        'evaluation': record.analysis.evaluation,
                        'win_probability': record.analysis.win_probability,
                        'search_depth': record.analysis.search_depth,
                        'nodes_searched': record.analysis.nodes_searched
                    }
                
                history.append(move_info)
            
            return history
            
        except Exception as e:
            self.logger.error(f"获取走法历史失败: {e}")
            return []
    
    def analyze_position(self, session_id: Optional[str] = None,
                        depth: Optional[int] = None) -> Optional[AnalysisResult]:
        """
        分析当前位置
        
        Args:
            session_id: 会话ID
            depth: 分析深度
            
        Returns:
            分析结果
        """
        try:
            session = self._get_session(session_id)
            
            # 确保AI引擎已初始化
            if self.ai_engine is None:
                self._initialize_ai(session.config.ai_config)
            
            return self.ai_engine.analyze_position(session.board, depth=depth)
            
        except Exception as e:
            self.logger.error(f"位置分析失败: {e}")
            return None
    
    def save_session(self, session_id: Optional[str] = None,
                    filepath: Optional[str] = None) -> bool:
        """
        保存会话
        
        Args:
            session_id: 会话ID
            filepath: 保存路径
            
        Returns:
            是否成功保存
        """
        try:
            session = self._get_session(session_id)
            return self._save_session(session, filepath)
            
        except Exception as e:
            self.logger.error(f"保存会话失败: {e}")
            return False
    
    def load_session(self, filepath: str) -> Optional[str]:
        """
        加载会话
        
        Args:
            filepath: 文件路径
            
        Returns:
            会话ID，如果失败返回None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建会话对象
            session = self._rebuild_session_from_dict(data)
            
            self.sessions[session.session_id] = session
            self.current_session = session
            
            self.logger.info(f"会话加载成功: {session.session_id}")
            return session.session_id
            
        except Exception as e:
            self.logger.error(f"加载会话失败: {e}")
            return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话
        
        Returns:
            会话信息列表
        """
        sessions_info = []
        for session_id, session in self.sessions.items():
            sessions_info.append({
                'session_id': session_id,
                'state': session.state.value,
                'result': session.result.value,
                'total_moves': session.total_moves,
                'created_at': session.created_at.isoformat(),
                'finished_at': session.finished_at.isoformat() if session.finished_at else None,
                'red_player': session.config.red_player_type.value,
                'black_player': session.config.black_player_type.value,
                'ai_difficulty': session.config.ai_difficulty
            })
        
        return sessions_info
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功删除
        """
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                
                # 如果是当前会话，清空当前会话
                if self.current_session and self.current_session.session_id == session_id:
                    self.current_session = None
                
                self.logger.info(f"会话删除成功: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"删除会话失败: {e}")
            return False
    
    def get_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            统计信息
        """
        try:
            if session_id:
                session = self._get_session(session_id)
                return self._get_session_statistics(session)
            else:
                # 返回所有会话的统计
                total_sessions = len(self.sessions)
                finished_sessions = sum(1 for s in self.sessions.values() 
                                      if s.state == GameState.FINISHED)
                total_moves = sum(s.total_moves for s in self.sessions.values())
                
                return {
                    'total_sessions': total_sessions,
                    'finished_sessions': finished_sessions,
                    'active_sessions': total_sessions - finished_sessions,
                    'total_moves': total_moves,
                    'average_moves_per_game': total_moves / max(finished_sessions, 1)
                }
                
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def _get_session(self, session_id: Optional[str] = None) -> GameSession:
        """获取会话对象"""
        if session_id is None:
            if self.current_session is None:
                raise ValueError("没有当前会话")
            return self.current_session
        
        if session_id not in self.sessions:
            raise ValueError(f"会话不存在: {session_id}")
        
        return self.sessions[session_id]
    
    def _check_game_end(self, session: GameSession):
        """检查游戏结束条件"""
        try:
            # 检查将死/困毙
            if session.board.is_checkmate():
                session.state = GameState.FINISHED
                # 被将死的是当前玩家，所以对手获胜
                session.result = (GameResult.BLACK_WIN if session.current_player == 1 
                                else GameResult.RED_WIN)
                session.finished_at = datetime.now()
                return
            
            # 检查和棋条件
            if session.board.is_stalemate():
                session.state = GameState.FINISHED
                session.result = GameResult.DRAW
                session.finished_at = datetime.now()
                return
            
            # 检查重复局面
            if (session.config.draw_by_repetition and 
                max(session.repetition_count.values()) >= 3):
                session.state = GameState.FINISHED
                session.result = GameResult.DRAW
                session.finished_at = datetime.now()
                return
            
            # 检查最大回合数
            if session.total_moves >= session.config.max_moves:
                session.state = GameState.FINISHED
                session.result = GameResult.DRAW
                session.finished_at = datetime.now()
                return
            
            # 检查总时间限制
            if session.config.total_time_limit and session.start_time:
                elapsed = (datetime.now() - session.start_time).total_seconds()
                if elapsed > session.config.total_time_limit:
                    session.state = GameState.FINISHED
                    session.result = GameResult.TIMEOUT
                    session.finished_at = datetime.now()
                    return
            
            # 保存游戏记录（如果游戏结束）
            if session.state == GameState.FINISHED and session.config.save_game_record:
                self._save_session(session)
                
        except Exception as e:
            self.logger.error(f"检查游戏结束条件失败: {e}")
            session.state = GameState.ERROR
    
    def _save_session(self, session: GameSession, filepath: Optional[str] = None) -> bool:
        """保存会话到文件"""
        try:
            if filepath is None:
                filename = f"game_{session.session_id}_{session.created_at.strftime('%Y%m%d_%H%M%S')}.json"
                filepath = self.save_directory / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"会话保存成功: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存会话失败: {e}")
            return False
    
    def _rebuild_session_from_dict(self, data: Dict[str, Any]) -> GameSession:
        """从字典重建会话对象"""
        # 重建配置
        config_data = data['config']
        config = GameConfig(
            red_player_type=PlayerType(config_data['red_player_type']),
            black_player_type=PlayerType(config_data['black_player_type']),
            ai_difficulty=config_data['ai_difficulty'],
            time_limit_per_move=config_data.get('time_limit_per_move'),
            total_time_limit=config_data.get('total_time_limit'),
            allow_undo=config_data.get('allow_undo', True),
            max_moves=config_data.get('max_moves', 300),
            draw_by_repetition=config_data.get('draw_by_repetition', True),
            save_game_record=config_data.get('save_game_record', True),
            record_analysis=config_data.get('record_analysis', False)
        )
        
        # 重建棋盘
        board = ChessBoard()
        board.from_fen(data['board'])
        
        # 重建走法历史
        move_history = [MoveRecord.from_dict(record_data) 
                       for record_data in data['move_history']]
        
        # 创建会话对象
        session = GameSession(
            session_id=data['session_id'],
            config=config,
            board=board,
            state=GameState(data['state']),
            result=GameResult(data['result']),
            current_player=data['current_player'],
            red_time_used=data['red_time_used'],
            black_time_used=data['black_time_used'],
            move_history=move_history,
            board_history=data['board_history'],
            total_moves=data['total_moves'],
            repetition_count=data['repetition_count'],
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {})
        )
        
        if data.get('start_time'):
            session.start_time = datetime.fromisoformat(data['start_time'])
        
        if data.get('finished_at'):
            session.finished_at = datetime.fromisoformat(data['finished_at'])
        
        return session
    
    def _get_session_statistics(self, session: GameSession) -> Dict[str, Any]:
        """获取单个会话的统计信息"""
        stats = {
            'session_id': session.session_id,
            'state': session.state.value,
            'result': session.result.value,
            'total_moves': session.total_moves,
            'red_time_used': session.red_time_used,
            'black_time_used': session.black_time_used,
            'game_duration': 0.0,
            'average_move_time': 0.0,
            'max_repetitions': max(session.repetition_count.values()) if session.repetition_count else 0
        }
        
        # 计算游戏时长
        if session.start_time:
            end_time = session.finished_at or datetime.now()
            stats['game_duration'] = (end_time - session.start_time).total_seconds()
        
        # 计算平均每步用时
        if session.move_history:
            total_time = sum(record.time_used for record in session.move_history)
            stats['average_move_time'] = total_time / len(session.move_history)
        
        return stats


# 异常处理类
class GameInterfaceError(Exception):
    """游戏接口异常"""
    pass


class SessionNotFoundError(GameInterfaceError):
    """会话未找到异常"""
    pass


class InvalidMoveError(GameInterfaceError):
    """非法走法异常"""
    pass


class GameStateError(GameInterfaceError):
    """游戏状态异常"""
    pass