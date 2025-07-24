"""
象棋棋盘数据结构

定义象棋棋盘的表示、操作和格式转换功能。
"""

import numpy as np
import json
import copy
from typing import List, Optional, Tuple, Dict, Any
from .move import Move


class ChessBoard:
    """
    象棋棋盘类
    
    维护棋局状态，支持多种表示格式的转换和基本操作。
    """
    
    # 棋子常量定义
    EMPTY = 0
    # 红方棋子 (正数)
    RED_KING = 1      # 帅
    RED_ADVISOR = 2   # 仕
    RED_BISHOP = 3    # 相
    RED_KNIGHT = 4    # 马
    RED_ROOK = 5      # 车
    RED_CANNON = 6    # 炮
    RED_PAWN = 7      # 兵
    
    # 黑方棋子 (负数)
    BLACK_KING = -1   # 将
    BLACK_ADVISOR = -2 # 士
    BLACK_BISHOP = -3  # 象
    BLACK_KNIGHT = -4  # 马
    BLACK_ROOK = -5    # 车
    BLACK_CANNON = -6  # 炮
    BLACK_PAWN = -7    # 卒
    
    # 棋子名称映射
    PIECE_NAMES = {
        0: "  ", 1: "帅", 2: "仕", 3: "相", 4: "马", 5: "车", 6: "炮", 7: "兵",
        -1: "将", -2: "士", -3: "象", -4: "马", -5: "车", -6: "炮", -7: "卒"
    }
    
    # FEN记法中的棋子符号
    FEN_PIECES = {
        1: 'K', 2: 'A', 3: 'B', 4: 'N', 5: 'R', 6: 'C', 7: 'P',
        -1: 'k', -2: 'a', -3: 'b', -4: 'n', -5: 'r', -6: 'c', -7: 'p'
    }
    
    def __init__(self, fen: Optional[str] = None):
        """
        初始化棋盘
        
        Args:
            fen: FEN格式的棋局字符串，如果为None则创建初始局面
        """
        # 10x9的棋盘矩阵 (行x列)
        self.board = np.zeros((10, 9), dtype=int)
        
        # 当前轮到的玩家 (1: 红方, -1: 黑方)
        self.current_player = 1
        
        # 走法历史
        self.move_history: List[Move] = []
        
        # 棋局状态历史 (用于检测重复局面)
        self.board_history: List[np.ndarray] = []
        
        if fen:
            self.from_fen(fen)
        else:
            self._setup_initial_position()
    
    def _setup_initial_position(self):
        """设置象棋初始局面"""
        # 黑方 (上方)
        self.board[0] = [-5, -4, -3, -2, -1, -2, -3, -4, -5]  # 车马象士将士象马车
        self.board[2] = [0, -6, 0, 0, 0, 0, 0, -6, 0]         # 炮
        self.board[3] = [-7, 0, -7, 0, -7, 0, -7, 0, -7]      # 卒
        
        # 空行
        for i in range(4, 6):
            self.board[i] = [0] * 9
        
        # 红方 (下方)
        self.board[6] = [7, 0, 7, 0, 7, 0, 7, 0, 7]           # 兵
        self.board[7] = [0, 6, 0, 0, 0, 0, 0, 6, 0]           # 炮
        self.board[9] = [5, 4, 3, 2, 1, 2, 3, 4, 5]           # 车马相仕帅仕相马车
        
        # 保存初始状态
        self.board_history.append(self.board.copy())
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, current_player: int = 1) -> 'ChessBoard':
        """
        从矩阵创建棋盘对象
        
        Args:
            matrix: 10x9的棋盘矩阵
            current_player: 当前玩家
            
        Returns:
            ChessBoard: 棋盘对象
        """
        board = cls()
        board.board = matrix.copy()
        board.current_player = current_player
        board.board_history = [matrix.copy()]
        return board
    
    def to_matrix(self) -> np.ndarray:
        """
        转换为矩阵格式
        
        Returns:
            np.ndarray: 10x9的棋盘矩阵
        """
        return self.board.copy()
    
    def to_fen(self) -> str:
        """
        转换为FEN格式
        
        Returns:
            str: FEN格式字符串
        """
        fen_parts = []
        
        # 棋盘部分
        for row in self.board:
            fen_row = ""
            empty_count = 0
            
            for piece in row:
                if piece == 0:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += self.FEN_PIECES[piece]
            
            if empty_count > 0:
                fen_row += str(empty_count)
            
            fen_parts.append(fen_row)
        
        board_fen = "/".join(fen_parts)
        
        # 当前玩家
        player_char = "w" if self.current_player == 1 else "b"
        
        # 其他信息 (暂时简化)
        return f"{board_fen} {player_char} - - 0 1"
    
    def from_fen(self, fen: str):
        """
        从FEN格式加载棋局
        
        Args:
            fen: FEN格式字符串
        """
        parts = fen.split()
        if len(parts) < 2:
            raise ValueError("无效的FEN格式")
        
        board_fen = parts[0]
        player_char = parts[1]
        
        # 解析棋盘
        rows = board_fen.split("/")
        if len(rows) != 10:
            raise ValueError("FEN格式应包含10行")
        
        # 反向映射FEN棋子符号
        fen_to_piece = {v: k for k, v in self.FEN_PIECES.items()}
        
        self.board = np.zeros((10, 9), dtype=int)
        
        for i, row in enumerate(rows):
            col = 0
            for char in row:
                if char.isdigit():
                    col += int(char)
                else:
                    if col >= 9:
                        raise ValueError(f"第{i+1}行列数超出范围")
                    self.board[i, col] = fen_to_piece.get(char, 0)
                    col += 1
        
        # 设置当前玩家
        self.current_player = 1 if player_char == "w" else -1
        
        # 保存状态
        self.board_history = [self.board.copy()]
    
    def make_move(self, move: Move) -> 'ChessBoard':
        """
        执行走法，返回新的棋盘状态
        
        Args:
            move: 要执行的走法
            
        Returns:
            ChessBoard: 新的棋盘状态
        """
        new_board = copy.deepcopy(self)
        
        # 执行移动
        from_row, from_col = move.from_pos
        to_row, to_col = move.to_pos
        
        # 记录被吃掉的棋子
        captured_piece = new_board.board[to_row, to_col]
        if captured_piece != 0:
            move.captured_piece = captured_piece
        
        # 移动棋子
        new_board.board[to_row, to_col] = new_board.board[from_row, from_col]
        new_board.board[from_row, from_col] = 0
        
        # 切换玩家
        new_board.current_player = -new_board.current_player
        
        # 记录历史
        new_board.move_history.append(move)
        new_board.board_history.append(new_board.board.copy())
        
        return new_board
    
    def undo_move(self) -> 'ChessBoard':
        """
        撤销上一步走法
        
        Returns:
            ChessBoard: 撤销后的棋盘状态
        """
        if not self.move_history:
            return self
        
        new_board = copy.deepcopy(self)
        
        # 移除最后一步
        last_move = new_board.move_history.pop()
        new_board.board_history.pop()
        
        # 恢复棋盘状态
        if new_board.board_history:
            new_board.board = new_board.board_history[-1].copy()
        
        # 切换玩家
        new_board.current_player = -new_board.current_player
        
        return new_board
    
    def get_piece_at(self, pos: Tuple[int, int]) -> int:
        """
        获取指定位置的棋子
        
        Args:
            pos: 位置坐标 (行, 列)
            
        Returns:
            int: 棋子类型
        """
        row, col = pos
        if 0 <= row <= 9 and 0 <= col <= 8:
            return self.board[row, col]
        return 0
    
    def is_empty(self, pos: Tuple[int, int]) -> bool:
        """
        检查指定位置是否为空
        
        Args:
            pos: 位置坐标
            
        Returns:
            bool: 是否为空
        """
        return self.get_piece_at(pos) == 0
    
    def is_enemy_piece(self, pos: Tuple[int, int], player: int) -> bool:
        """
        检查指定位置是否为敌方棋子
        
        Args:
            pos: 位置坐标
            player: 玩家 (1: 红方, -1: 黑方)
            
        Returns:
            bool: 是否为敌方棋子
        """
        piece = self.get_piece_at(pos)
        return piece != 0 and (piece > 0) != (player > 0)
    
    def is_own_piece(self, pos: Tuple[int, int], player: int) -> bool:
        """
        检查指定位置是否为己方棋子
        
        Args:
            pos: 位置坐标
            player: 玩家
            
        Returns:
            bool: 是否为己方棋子
        """
        piece = self.get_piece_at(pos)
        return piece != 0 and (piece > 0) == (player > 0)
    
    def find_king(self, player: int) -> Optional[Tuple[int, int]]:
        """
        找到指定玩家的王(帅/将)的位置
        
        Args:
            player: 玩家
            
        Returns:
            Optional[Tuple[int, int]]: 王的位置，如果找不到返回None
        """
        king_piece = self.RED_KING if player > 0 else self.BLACK_KING
        
        for row in range(10):
            for col in range(9):
                if self.board[row, col] == king_piece:
                    return (row, col)
        
        return None
    
    def to_visual_string(self) -> str:
        """
        转换为可视化字符串
        
        Returns:
            str: 可视化的棋盘字符串
        """
        lines = []
        lines.append("  a b c d e f g h i")
        lines.append("  +-+-+-+-+-+-+-+-+")
        
        for row in range(10):
            line = f"{row}|"
            for col in range(9):
                piece = self.board[row, col]
                line += self.PIECE_NAMES[piece] + "|"
            lines.append(line)
            lines.append("  +-+-+-+-+-+-+-+-+")
        
        lines.append(f"当前玩家: {'红方' if self.current_player == 1 else '黑方'}")
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """
        转换为JSON格式
        
        Returns:
            str: JSON格式字符串
        """
        data = {
            'board': self.board.tolist(),
            'current_player': self.current_player,
            'move_history': [
                {
                    'from_pos': move.from_pos,
                    'to_pos': move.to_pos,
                    'piece': move.piece,
                    'captured_piece': move.captured_piece
                }
                for move in self.move_history
            ]
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ChessBoard':
        """
        从JSON格式创建棋盘对象
        
        Args:
            json_str: JSON格式字符串
            
        Returns:
            ChessBoard: 棋盘对象
        """
        data = json.loads(json_str)
        
        board = cls()
        board.board = np.array(data['board'], dtype=int)
        board.current_player = data['current_player']
        
        # 重建走法历史
        board.move_history = []
        for move_data in data['move_history']:
            move = Move(
                from_pos=tuple(move_data['from_pos']),
                to_pos=tuple(move_data['to_pos']),
                piece=move_data['piece'],
                captured_piece=move_data.get('captured_piece')
            )
            board.move_history.append(move)
        
        return board
    
    def __str__(self) -> str:
        """字符串表示"""
        return self.to_visual_string()
    
    def __eq__(self, other) -> bool:
        """相等性比较"""
        if not isinstance(other, ChessBoard):
            return False
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player)
    
    def __hash__(self) -> int:
        """哈希值计算"""
        return hash((self.board.tobytes(), self.current_player))