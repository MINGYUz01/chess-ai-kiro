"""
象棋棋盘数据结构

定义象棋棋盘的表示、操作和格式转换功能。
"""

import numpy as np
import json
import copy
import pickle
import hashlib
from typing import List, Optional, Tuple, Dict, Any, Union
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
        
        # 棋局元数据
        self.metadata: Dict[str, Any] = {
            'created_at': None,
            'game_id': None,
            'round_count': 0,
            'last_capture_round': 0,  # 最后一次吃子的回合数
            'repetition_count': {}    # 局面重复计数
        }
        
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
            # 更新最后吃子回合
            new_board.metadata['last_capture_round'] = len(new_board.move_history)
        
        # 移动棋子
        new_board.board[to_row, to_col] = new_board.board[from_row, from_col]
        new_board.board[from_row, from_col] = 0
        
        # 更新回合计数
        new_board.metadata['round_count'] = len(new_board.move_history) + 1
        
        # 切换玩家
        new_board.current_player = -new_board.current_player
        
        # 记录历史
        new_board.move_history.append(move)
        new_board.board_history.append(new_board.board.copy())
        
        # 更新重复局面计数
        board_hash = new_board.get_board_hash()
        if board_hash not in new_board.metadata['repetition_count']:
            new_board.metadata['repetition_count'][board_hash] = 0
        new_board.metadata['repetition_count'][board_hash] += 1
        
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
        else:
            # 如果没有历史状态，重置为初始状态
            new_board._setup_initial_position()
        
        # 更新回合计数
        new_board.metadata['round_count'] = len(new_board.move_history)
        
        # 重新计算最后吃子回合
        new_board.metadata['last_capture_round'] = 0
        for i, move in enumerate(new_board.move_history):
            if move.captured_piece is not None:
                new_board.metadata['last_capture_round'] = i
        
        # 切换玩家
        new_board.current_player = -new_board.current_player
        
        # 重新计算重复局面计数
        new_board.metadata['repetition_count'] = {}
        temp_board = ChessBoard()
        for move in new_board.move_history:
            temp_board = temp_board.make_move(move)
            board_hash = temp_board.get_board_hash()
            if board_hash not in new_board.metadata['repetition_count']:
                new_board.metadata['repetition_count'][board_hash] = 0
            new_board.metadata['repetition_count'][board_hash] += 1
        
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
    
    # ==================== 序列化和反序列化功能 ====================
    
    def to_binary(self) -> bytes:
        """
        转换为二进制格式
        
        Returns:
            bytes: 二进制数据
        """
        data = {
            'board': self.board,
            'current_player': self.current_player,
            'move_history': self.move_history,
            'metadata': self.metadata
        }
        return pickle.dumps(data)
    
    @classmethod
    def from_binary(cls, binary_data: bytes) -> 'ChessBoard':
        """
        从二进制格式创建棋盘对象
        
        Args:
            binary_data: 二进制数据
            
        Returns:
            ChessBoard: 棋盘对象
        """
        data = pickle.loads(binary_data)
        
        board = cls()
        board.board = data['board']
        board.current_player = data['current_player']
        board.move_history = data['move_history']
        board.metadata = data.get('metadata', {})
        
        # 重建棋局历史
        board.board_history = [board.board.copy()]
        temp_board = board.board.copy()
        
        for i, move in enumerate(board.move_history):
            # 重建每一步的棋局状态
            if i > 0:
                prev_move = board.move_history[i-1]
                # 执行前一步走法来重建状态
                from_row, from_col = prev_move.from_pos
                to_row, to_col = prev_move.to_pos
                temp_board[to_row, to_col] = temp_board[from_row, from_col]
                temp_board[from_row, from_col] = 0
                board.board_history.append(temp_board.copy())
        
        return board
    
    def save_to_file(self, filepath: str, format: str = 'json') -> None:
        """
        保存棋局到文件
        
        Args:
            filepath: 文件路径
            format: 保存格式 ('json', 'binary', 'fen')
        """
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
        elif format == 'binary':
            with open(filepath, 'wb') as f:
                f.write(self.to_binary())
        elif format == 'fen':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_fen())
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @classmethod
    def load_from_file(cls, filepath: str, format: str = 'auto') -> 'ChessBoard':
        """
        从文件加载棋局
        
        Args:
            filepath: 文件路径
            format: 文件格式 ('json', 'binary', 'fen', 'auto')
            
        Returns:
            ChessBoard: 棋盘对象
        """
        if format == 'auto':
            # 根据文件扩展名自动判断格式
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.bin') or filepath.endswith('.pkl'):
                format = 'binary'
            elif filepath.endswith('.fen'):
                format = 'fen'
            else:
                # 尝试读取文件内容判断格式
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content.startswith('{'):
                            format = 'json'
                        else:
                            format = 'fen'
                except:
                    format = 'binary'
        
        if format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                return cls.from_json(f.read())
        elif format == 'binary':
            with open(filepath, 'rb') as f:
                return cls.from_binary(f.read())
        elif format == 'fen':
            with open(filepath, 'r', encoding='utf-8') as f:
                return cls(fen=f.read().strip())
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    # ==================== 棋局验证功能 ====================
    
    def validate_board_state(self) -> Tuple[bool, List[str]]:
        """
        验证棋局状态的合法性
        
        Returns:
            Tuple[bool, List[str]]: (是否合法, 错误信息列表)
        """
        # 使用专门的验证器进行验证
        from .board_validator import BoardValidator
        validator = BoardValidator()
        return validator.full_validation(self)
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        获取详细的验证报告
        
        Returns:
            Dict[str, Any]: 验证报告
        """
        from .board_validator import BoardValidator
        validator = BoardValidator()
        return validator.get_validation_report(self)
    
    def get_board_hash(self) -> str:
        """
        获取棋局状态的哈希值
        
        Returns:
            str: 棋局状态的MD5哈希值
        """
        # 将棋盘状态和当前玩家组合成字符串
        state_str = f"{self.board.tobytes()}{self.current_player}"
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def is_repetition(self, max_repetitions: int = 3) -> bool:
        """
        检查是否出现重复局面
        
        Args:
            max_repetitions: 最大重复次数
            
        Returns:
            bool: 是否达到重复次数限制
        """
        current_hash = self.get_board_hash()
        
        # 更新重复计数
        if current_hash not in self.metadata['repetition_count']:
            self.metadata['repetition_count'][current_hash] = 0
        
        self.metadata['repetition_count'][current_hash] += 1
        
        return self.metadata['repetition_count'][current_hash] >= max_repetitions
    
    # ==================== 走法历史管理 ====================
    
    def get_move_count(self) -> int:
        """
        获取走法总数
        
        Returns:
            int: 走法总数
        """
        return len(self.move_history)
    
    def get_last_move(self) -> Optional[Move]:
        """
        获取最后一步走法
        
        Returns:
            Optional[Move]: 最后一步走法，如果没有则返回None
        """
        return self.move_history[-1] if self.move_history else None
    
    def get_move_at(self, index: int) -> Optional[Move]:
        """
        获取指定索引的走法
        
        Args:
            index: 走法索引
            
        Returns:
            Optional[Move]: 指定的走法，如果索引无效则返回None
        """
        if 0 <= index < len(self.move_history):
            return self.move_history[index]
        return None
    
    def get_moves_since(self, round_number: int) -> List[Move]:
        """
        获取从指定回合开始的所有走法
        
        Args:
            round_number: 起始回合数
            
        Returns:
            List[Move]: 走法列表
        """
        if round_number < 0 or round_number >= len(self.move_history):
            return []
        
        return self.move_history[round_number:]
    
    def clear_history_after(self, round_number: int) -> None:
        """
        清除指定回合之后的历史记录
        
        Args:
            round_number: 保留到的回合数
        """
        if 0 <= round_number < len(self.move_history):
            self.move_history = self.move_history[:round_number + 1]
            self.board_history = self.board_history[:round_number + 2]  # 包含初始状态
    
    def replay_moves(self, moves: List[Move]) -> 'ChessBoard':
        """
        重放一系列走法
        
        Args:
            moves: 要重放的走法列表
            
        Returns:
            ChessBoard: 重放后的棋盘状态
        """
        # 从当前状态开始重放
        result_board = copy.deepcopy(self)
        
        for move in moves:
            result_board = result_board.make_move(move)
        
        return result_board
    
    # ==================== 实用工具方法 ====================
    
    def copy(self) -> 'ChessBoard':
        """
        创建棋盘的深拷贝
        
        Returns:
            ChessBoard: 棋盘副本
        """
        return copy.deepcopy(self)
    
    def reset_to_initial(self) -> None:
        """重置到初始局面"""
        self.__init__()
    
    def get_all_pieces(self, player: Optional[int] = None) -> List[Tuple[Tuple[int, int], int]]:
        """
        获取所有棋子的位置和类型
        
        Args:
            player: 指定玩家，None表示获取所有棋子
            
        Returns:
            List[Tuple[Tuple[int, int], int]]: [(位置, 棋子类型), ...]
        """
        pieces = []
        
        for row in range(10):
            for col in range(9):
                piece = self.board[row, col]
                if piece != 0:
                    if player is None or (piece > 0) == (player > 0):
                        pieces.append(((row, col), piece))
        
        return pieces
    
    def count_pieces(self, player: Optional[int] = None) -> Dict[int, int]:
        """
        统计棋子数量
        
        Args:
            player: 指定玩家，None表示统计所有棋子
            
        Returns:
            Dict[int, int]: {棋子类型: 数量}
        """
        counts = {}
        
        for row in range(10):
            for col in range(9):
                piece = self.board[row, col]
                if piece != 0:
                    if player is None or (piece > 0) == (player > 0):
                        counts[piece] = counts.get(piece, 0) + 1
        
        return counts
    
    def get_material_value(self, player: int) -> int:
        """
        计算指定玩家的棋子总价值
        
        Args:
            player: 玩家
            
        Returns:
            int: 棋子总价值
        """
        # 棋子价值表
        piece_values = {
            1: 10000, -1: 10000,  # 帅/将
            2: 200, -2: 200,      # 仕/士
            3: 200, -3: 200,      # 相/象
            4: 400, -4: 400,      # 马
            5: 900, -5: 900,      # 车
            6: 450, -6: 450,      # 炮
            7: 100, -7: 100       # 兵/卒
        }
        
        total_value = 0
        pieces = self.get_all_pieces(player)
        
        for pos, piece in pieces:
            total_value += piece_values.get(piece, 0)
        
        return total_value   
 
    # ==================== 游戏状态检测方法 ====================
    
    def get_legal_moves(self, player: Optional[int] = None) -> List[Move]:
        """
        获取合法走法列表
        
        Args:
            player: 指定玩家，None表示当前玩家
            
        Returns:
            List[Move]: 合法走法列表
        """
        from .rule_engine import RuleEngine
        
        if not hasattr(self, '_rule_engine'):
            self._rule_engine = RuleEngine()
        
        return self._rule_engine.generate_legal_moves(self, player)
    
    def is_game_over(self) -> bool:
        """
        检查游戏是否结束
        
        Returns:
            bool: 游戏是否结束
        """
        # 检查是否有合法走法
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            return True
        
        # 检查是否达到最大重复次数
        if self.is_repetition(max_repetitions=3):
            return True
        
        # 检查是否达到50回合规则（无吃子）
        current_round = self.metadata.get('round_count', 0)
        last_capture_round = self.metadata.get('last_capture_round', 0)
        if current_round - last_capture_round >= 100:  # 50回合 = 100步
            return True
        
        return False
    
    def get_winner(self) -> int:
        """
        获取游戏获胜者
        
        Returns:
            int: 获胜者 (1: 红方, -1: 黑方, 0: 平局)
        """
        if not self.is_game_over():
            return 0  # 游戏未结束
        
        # 检查是否有合法走法
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            # 检查是否被将军
            from .rule_engine import RuleEngine
            if not hasattr(self, '_rule_engine'):
                self._rule_engine = RuleEngine()
            
            if self._rule_engine.is_in_check(self, self.current_player):
                # 被将死，对手获胜
                return -self.current_player
            else:
                # 困毙，平局
                return 0
        
        # 其他情况都是平局
        return 0
    
    def is_legal_move(self, move: Move) -> bool:
        """
        检查走法是否合法
        
        Args:
            move: 要检查的走法
            
        Returns:
            bool: 走法是否合法
        """
        legal_moves = self.get_legal_moves()
        return move in legal_moves