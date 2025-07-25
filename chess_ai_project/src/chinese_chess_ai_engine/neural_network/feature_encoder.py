"""
棋局特征编码器

将棋局状态转换为神经网络可以处理的张量格式。
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ..rules_engine import ChessBoard, Move


class FeatureEncoder:
    """
    棋局特征编码器
    
    将ChessBoard对象转换为多通道的特征张量。
    """
    
    def __init__(self, history_length: int = 8):
        """
        初始化编码器
        
        Args:
            history_length: 历史走法长度
        """
        self.history_length = history_length
        
        # 棋子类型映射
        self.piece_types = {
            0: 0,   # 空位
            1: 1, -1: 1,    # 帅/将
            2: 2, -2: 2,    # 仕/士
            3: 3, -3: 3,    # 相/象
            4: 4, -4: 4,    # 马
            5: 5, -5: 5,    # 车
            6: 6, -6: 6,    # 炮
            7: 7, -7: 7     # 兵/卒
        }
        
        # 特征通道定义
        self.feature_channels = {
            'red_pieces': 7,      # 红方各类棋子 (7个通道)
            'black_pieces': 7,    # 黑方各类棋子 (7个通道)
            'current_player': 1,  # 当前玩家 (1个通道)
            'history': 1,         # 历史信息 (1个通道)
            'legal_moves': 1,     # 合法走法 (1个通道)
            'attack_defend': 2,   # 攻击和防守信息 (2个通道)
            'special_rules': 1    # 特殊规则信息 (1个通道)
        }
        
        self.total_channels = sum(self.feature_channels.values())  # 总共20个通道
    
    def encode_board(
        self, 
        board: ChessBoard, 
        legal_moves: Optional[List[Move]] = None,
        move_history: Optional[List[Move]] = None
    ) -> torch.Tensor:
        """
        编码棋盘状态
        
        Args:
            board: 棋盘对象
            legal_moves: 合法走法列表
            move_history: 走法历史
            
        Returns:
            torch.Tensor: 特征张量 [channels, 10, 9]
        """
        features = torch.zeros(self.total_channels, 10, 9, dtype=torch.float32)
        channel_idx = 0
        
        # 1. 编码红方棋子 (7个通道)
        for piece_type in range(1, 8):  # 1-7对应7种棋子
            piece_channel = torch.zeros(10, 9)
            for row in range(10):
                for col in range(9):
                    piece = board.board[row, col]
                    if piece > 0 and self.piece_types[piece] == piece_type:
                        piece_channel[row, col] = 1.0
            features[channel_idx] = piece_channel
            channel_idx += 1
        
        # 2. 编码黑方棋子 (7个通道)
        for piece_type in range(1, 8):
            piece_channel = torch.zeros(10, 9)
            for row in range(10):
                for col in range(9):
                    piece = board.board[row, col]
                    if piece < 0 and self.piece_types[piece] == piece_type:
                        piece_channel[row, col] = 1.0
            features[channel_idx] = piece_channel
            channel_idx += 1
        
        # 3. 编码当前玩家 (1个通道)
        current_player_channel = torch.full((10, 9), 1.0 if board.current_player == 1 else 0.0)
        features[channel_idx] = current_player_channel
        channel_idx += 1
        
        # 4. 编码历史信息 (1个通道)
        history_channel = self._encode_history(board, move_history)
        features[channel_idx] = history_channel
        channel_idx += 1
        
        # 5. 编码合法走法 (1个通道)
        legal_moves_channel = self._encode_legal_moves(board, legal_moves)
        features[channel_idx] = legal_moves_channel
        channel_idx += 1
        
        # 6. 编码攻击和防守信息 (2个通道)
        attack_channel, defend_channel = self._encode_attack_defend(board)
        features[channel_idx] = attack_channel
        features[channel_idx + 1] = defend_channel
        channel_idx += 2
        
        # 7. 编码特殊规则信息 (1个通道)
        special_rules_channel = self._encode_special_rules(board)
        features[channel_idx] = special_rules_channel
        
        return features
    
    def _encode_history(self, board: ChessBoard, move_history: Optional[List[Move]]) -> torch.Tensor:
        """
        编码历史信息
        
        Args:
            board: 棋盘对象
            move_history: 走法历史
            
        Returns:
            torch.Tensor: 历史特征 [10, 9]
        """
        history_channel = torch.zeros(10, 9)
        
        if move_history and len(move_history) > 0:
            # 编码最近的走法
            recent_moves = move_history[-min(self.history_length, len(move_history)):]
            
            for i, move in enumerate(recent_moves):
                # 根据走法的新旧程度给予不同的权重
                weight = (i + 1) / len(recent_moves)
                
                # 标记起始和结束位置
                from_row, from_col = move.from_pos
                to_row, to_col = move.to_pos
                
                history_channel[from_row, from_col] = max(history_channel[from_row, from_col], weight * 0.5)
                history_channel[to_row, to_col] = max(history_channel[to_row, to_col], weight)
        
        return history_channel
    
    def _encode_legal_moves(self, board: ChessBoard, legal_moves: Optional[List[Move]]) -> torch.Tensor:
        """
        编码合法走法
        
        Args:
            board: 棋盘对象
            legal_moves: 合法走法列表
            
        Returns:
            torch.Tensor: 合法走法特征 [10, 9]
        """
        legal_moves_channel = torch.zeros(10, 9)
        
        if legal_moves:
            # 统计每个位置作为目标的频次
            target_counts = {}
            for move in legal_moves:
                to_pos = move.to_pos
                target_counts[to_pos] = target_counts.get(to_pos, 0) + 1
            
            # 归一化并填充通道
            max_count = max(target_counts.values()) if target_counts else 1
            for (row, col), count in target_counts.items():
                legal_moves_channel[row, col] = count / max_count
        
        return legal_moves_channel
    
    def _encode_attack_defend(self, board: ChessBoard) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码攻击和防守信息
        
        Args:
            board: 棋盘对象
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (攻击特征, 防守特征)
        """
        attack_channel = torch.zeros(10, 9)
        defend_channel = torch.zeros(10, 9)
        
        current_player = board.current_player
        
        # 分析每个位置的攻击和防守情况
        for row in range(10):
            for col in range(9):
                pos = (row, col)
                piece = board.board[row, col]
                
                if piece != 0:
                    # 统计攻击该位置的敌方棋子数量
                    attackers = self._count_attackers(board, pos, -current_player)
                    # 统计保护该位置的己方棋子数量
                    defenders = self._count_attackers(board, pos, current_player)
                    
                    # 归一化到[0, 1]范围
                    attack_channel[row, col] = min(attackers / 3.0, 1.0)
                    defend_channel[row, col] = min(defenders / 3.0, 1.0)
        
        return attack_channel, defend_channel
    
    def _count_attackers(self, board: ChessBoard, target_pos: Tuple[int, int], player: int) -> int:
        """
        统计攻击指定位置的棋子数量
        
        Args:
            board: 棋盘对象
            target_pos: 目标位置
            player: 玩家
            
        Returns:
            int: 攻击者数量
        """
        count = 0
        
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece != 0 and (piece > 0) == (player > 0):
                    # 简化的攻击判断（实际应该使用RuleEngine）
                    if self._can_simple_attack(board, (row, col), target_pos):
                        count += 1
        
        return count
    
    def _can_simple_attack(self, board: ChessBoard, attacker_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> bool:
        """
        简化的攻击判断
        
        Args:
            board: 棋盘对象
            attacker_pos: 攻击者位置
            target_pos: 目标位置
            
        Returns:
            bool: 是否能攻击
        """
        # 这里实现简化的攻击判断逻辑
        # 实际应用中应该使用RuleEngine的更精确判断
        
        from_row, from_col = attacker_pos
        to_row, to_col = target_pos
        
        if from_row == to_row or from_col == to_col:
            # 直线攻击（车、炮的简化判断）
            return True
        
        if abs(from_row - to_row) == abs(from_col - to_col):
            # 斜线攻击（相、象的简化判断）
            return True
        
        if (abs(from_row - to_row) == 2 and abs(from_col - to_col) == 1) or \
           (abs(from_row - to_row) == 1 and abs(from_col - to_col) == 2):
            # 马的攻击
            return True
        
        if abs(from_row - to_row) <= 1 and abs(from_col - to_col) <= 1:
            # 近距离攻击（帅、将、仕、士、兵、卒）
            return True
        
        return False
    
    def _encode_special_rules(self, board: ChessBoard) -> torch.Tensor:
        """
        编码特殊规则信息
        
        Args:
            board: 棋盘对象
            
        Returns:
            torch.Tensor: 特殊规则特征 [10, 9]
        """
        special_rules_channel = torch.zeros(10, 9)
        
        # 编码一些特殊规则信息
        metadata = board.metadata
        
        # 1. 无吃子步数信息
        moves_since_capture = (metadata.get('round_count', 0) - 
                              metadata.get('last_capture_round', 0))
        capture_ratio = min(moves_since_capture / 60.0, 1.0)  # 归一化到60步
        
        # 2. 重复局面信息
        repetition_info = 0.0
        if metadata.get('repetition_count'):
            max_repetitions = max(metadata['repetition_count'].values())
            repetition_info = min(max_repetitions / 3.0, 1.0)
        
        # 将信息编码到通道中（使用不同的模式）
        for row in range(10):
            for col in range(9):
                # 使用棋盘位置的奇偶性来编码不同信息
                if (row + col) % 2 == 0:
                    special_rules_channel[row, col] = capture_ratio
                else:
                    special_rules_channel[row, col] = repetition_info
        
        return special_rules_channel
    
    def encode_batch(
        self, 
        boards: List[ChessBoard], 
        legal_moves_list: Optional[List[List[Move]]] = None,
        move_histories: Optional[List[List[Move]]] = None
    ) -> torch.Tensor:
        """
        批量编码棋盘
        
        Args:
            boards: 棋盘列表
            legal_moves_list: 合法走法列表的列表
            move_histories: 走法历史列表
            
        Returns:
            torch.Tensor: 批量特征张量 [batch_size, channels, 10, 9]
        """
        batch_size = len(boards)
        batch_features = torch.zeros(batch_size, self.total_channels, 10, 9)
        
        for i, board in enumerate(boards):
            legal_moves = legal_moves_list[i] if legal_moves_list else None
            move_history = move_histories[i] if move_histories else None
            
            batch_features[i] = self.encode_board(board, legal_moves, move_history)
        
        return batch_features
    
    def decode_policy_to_moves(
        self, 
        policy_output: torch.Tensor, 
        board: ChessBoard,
        legal_moves: List[Move],
        top_k: int = 10
    ) -> List[Tuple[Move, float]]:
        """
        将策略输出解码为走法和概率
        
        Args:
            policy_output: 策略网络输出 [8100] 或 [1, 8100]
            board: 棋盘对象
            legal_moves: 合法走法列表
            top_k: 返回前k个走法
            
        Returns:
            List[Tuple[Move, float]]: [(走法, 概率), ...]
        """
        if policy_output.dim() == 2:
            policy_output = policy_output.squeeze(0)
        
        # 将策略输出映射到合法走法
        move_probs = []
        
        for move in legal_moves:
            # 计算走法对应的策略索引
            move_idx = self._move_to_policy_index(move)
            if 0 <= move_idx < len(policy_output):
                prob = policy_output[move_idx].item()
                move_probs.append((move, prob))
        
        # 按概率排序并返回前k个
        move_probs.sort(key=lambda x: x[1], reverse=True)
        return move_probs[:top_k]
    
    def _move_to_policy_index(self, move: Move) -> int:
        """
        将走法转换为策略索引
        
        Args:
            move: 走法对象
            
        Returns:
            int: 策略索引
        """
        from_row, from_col = move.from_pos
        to_row, to_col = move.to_pos
        
        # 计算移动方向
        dr = to_row - from_row
        dc = to_col - from_col
        
        # 将移动方向映射到索引（简化版本）
        # 实际实现中需要更精确的映射
        direction_idx = 0
        
        if dr == 0 and dc != 0:  # 水平移动
            direction_idx = 0 if dc > 0 else 1
        elif dr != 0 and dc == 0:  # 垂直移动
            direction_idx = 2 if dr > 0 else 3
        elif dr != 0 and dc != 0:  # 斜向移动
            if dr > 0 and dc > 0:
                direction_idx = 4
            elif dr > 0 and dc < 0:
                direction_idx = 5
            elif dr < 0 and dc > 0:
                direction_idx = 6
            else:
                direction_idx = 7
        
        # 计算最终索引
        policy_idx = from_row * 9 * 90 + from_col * 90 + direction_idx
        return policy_idx
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征编码器信息
        
        Returns:
            Dict[str, Any]: 特征信息
        """
        return {
            'total_channels': self.total_channels,
            'feature_channels': self.feature_channels,
            'history_length': self.history_length,
            'board_shape': (10, 9),
            'output_shape': (self.total_channels, 10, 9)
        }