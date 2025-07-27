"""
象棋走法数据结构

定义象棋走法的表示和转换功能。
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import re


@dataclass
class Move:
    """
    象棋走法类
    
    表示一个象棋走法，包含起始位置、目标位置、棋子信息等。
    """
    from_pos: Tuple[int, int]  # 起始位置 (行, 列)
    to_pos: Tuple[int, int]    # 目标位置 (行, 列)
    piece: int                 # 移动的棋子类型
    captured_piece: Optional[int] = None  # 被吃掉的棋子类型
    is_check: bool = False     # 是否将军
    is_checkmate: bool = False # 是否将死
    
    def __post_init__(self):
        """初始化后验证数据有效性"""
        self._validate_positions()
    
    def _validate_positions(self):
        """验证位置坐标的有效性"""
        for pos in [self.from_pos, self.to_pos]:
            row, col = pos
            if not (0 <= row <= 9 and 0 <= col <= 8):
                raise ValueError(f"无效的位置坐标: {pos}")
    
    def to_coordinate_notation(self) -> str:
        """
        转换为坐标记法
        
        Returns:
            str: 坐标记法字符串，如 "a0b1"
        """
        from_col = chr(ord('a') + self.from_pos[1])
        from_row = str(self.from_pos[0])
        to_col = chr(ord('a') + self.to_pos[1])
        to_row = str(self.to_pos[0])
        
        return f"{from_col}{from_row}{to_col}{to_row}"
    
    def to_chinese_notation(self) -> str:
        """
        转换为中文纵线记法
        
        Returns:
            str: 中文记法字符串，如 "炮二平五"
        """
        # 棋子名称映射
        piece_names = {
            1: "帅", 2: "仕", 3: "相", 4: "马", 5: "车", 6: "炮", 7: "兵",
            -1: "将", -2: "士", -3: "象", -4: "马", -5: "车", -6: "炮", -7: "卒"
        }
        
        # 数字映射
        numbers = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
        
        piece_name = piece_names.get(self.piece, "")
        from_col = numbers[9 - self.from_pos[1]] if self.piece > 0 else numbers[self.from_pos[1] + 1]
        to_col = numbers[9 - self.to_pos[1]] if self.piece > 0 else numbers[self.to_pos[1] + 1]
        
        # 判断移动方向
        if self.from_pos[1] == self.to_pos[1]:  # 直进直退
            if self.piece > 0:  # 红方
                direction = "进" if self.to_pos[0] < self.from_pos[0] else "退"
            else:  # 黑方
                direction = "进" if self.to_pos[0] > self.from_pos[0] else "退"
            steps = abs(self.to_pos[0] - self.from_pos[0])
            return f"{piece_name}{from_col}{direction}{numbers[steps]}"
        else:  # 平移
            return f"{piece_name}{from_col}平{to_col}"
    
    @classmethod
    def from_coordinate_notation(cls, notation: str, piece: int) -> 'Move':
        """
        从坐标记法创建Move对象
        
        Args:
            notation: 坐标记法字符串，如 "a0b1"
            piece: 移动的棋子类型
            
        Returns:
            Move: Move对象
        """
        if len(notation) != 4:
            raise ValueError(f"无效的坐标记法: {notation}")
        
        from_col = ord(notation[0]) - ord('a')
        from_row = int(notation[1])
        to_col = ord(notation[2]) - ord('a')
        to_row = int(notation[3])
        
        return cls(
            from_pos=(from_row, from_col),
            to_pos=(to_row, to_col),
            piece=piece
        )
    
    @classmethod
    def from_chinese_notation(cls, notation: str, board_state) -> 'Move':
        """
        从中文纵线记法创建Move对象
        
        Args:
            notation: 中文记法字符串，如 "炮二平五"
            board_state: 当前棋局状态，用于确定具体位置
            
        Returns:
            Move: Move对象
        """
        # 这里需要根据棋局状态来解析中文记法
        # 由于中文记法的解析比较复杂，这里先提供基本框架
        # 具体实现需要结合棋局状态来确定唯一的走法
        raise NotImplementedError("中文记法解析功能待实现")
    
    def __str__(self) -> str:
        """字符串表示"""
        return self.to_coordinate_notation()
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"Move(from_pos={self.from_pos}, to_pos={self.to_pos}, "
                f"piece={self.piece}, captured_piece={self.captured_piece})")
    
    def __eq__(self, other) -> bool:
        """相等性比较"""
        if not isinstance(other, Move):
            return False
        return (self.from_pos == other.from_pos and 
                self.to_pos == other.to_pos and
                self.piece == other.piece)
    
    def __hash__(self) -> int:
        """哈希值计算"""
        return hash((self.from_pos, self.to_pos, self.piece))
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'from_pos': self.from_pos,
            'to_pos': self.to_pos,
            'piece': self.piece,
            'captured_piece': self.captured_piece,
            'is_check': self.is_check,
            'is_checkmate': self.is_checkmate
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Move':
        """从字典创建Move对象"""
        return cls(
            from_pos=tuple(data['from_pos']),
            to_pos=tuple(data['to_pos']),
            piece=data['piece'],
            captured_piece=data.get('captured_piece'),
            is_check=data.get('is_check', False),
            is_checkmate=data.get('is_checkmate', False)
        )