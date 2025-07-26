"""
棋盘特征编码器

将棋盘状态转换为神经网络可以处理的张量格式。
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

from ..rules_engine import ChessBoard, Move


class BoardEncoder:
    """
    棋盘特征编码器
    
    将象棋棋盘状态编码为20通道的特征张量。
    """
    
    def __init__(self, history_length: int = 8):
        """
        初始化编码器
        
        Args:
            history_length: 历史步数长度
        """
        self.history_length = history_length
        self.logger = logging.getLogger(__name__)
        
        # 棋子类型到通道的映射
        self.piece_to_channel = {
            # 红方棋子 (通道0-6)
            1: 0,   # 帅
            2: 1,   # 仕
            3: 2,   # 相
            4: 3,   # 马
            5: 4,   # 车
            6: 5,   # 炮
            7: 6,   # 兵
            
            # 黑方棋子 (通道7-13)
            -1: 7,  # 将
            -2: 8,  # 士
            -3: 9,  # 象
            -4: 10, # 马
            -5: 11, # 车
            -6: 12, # 炮
            -7: 13  # 卒
        }
        
        # 总通道数: 14(棋子) + 2(重复局面) + 1(当前玩家) + 1(无吃子步数) + 2(王位置) = 20
        self.total_channels = 20
    
    def encode_board(self, board: ChessBoard) -> torch.Tensor:
        """
        编码单个棋盘状态
        
        Args:
            board: 棋盘状态
            
        Returns:
            torch.Tensor: 编码后的张量 [20, 10, 9]
        """
        # 创建特征张量
        features = torch.zeros(self.total_channels, 10, 9, dtype=torch.float32)
        
        # 1. 编码棋子位置 (通道0-13)
        self._encode_pieces(board, features)
        
        # 2. 编码重复局面信息 (通道14-15)
        self._encode_repetition(board, features)
        
        # 3. 编码当前玩家 (通道16)
        self._encode_current_player(board, features)
        
        # 4. 编码无吃子步数 (通道17)
        self._encode_no_capture_count(board, features)
        
        # 5. 编码王的位置 (通道18-19)
        self._encode_king_positions(board, features)
        
        return features
    
    def _encode_pieces(self, board: ChessBoard, features: torch.Tensor):
        """
        编码棋子位置
        
        Args:
            board: 棋盘状态
            features: 特征张量
        """
        board_matrix = board.to_matrix()
        
        for row in range(10):
            for col in range(9):
                piece = board_matrix[row, col]
                if piece != 0:
                    channel = self.piece_to_channel[piece]
                    features[channel, row, col] = 1.0
    
    def _encode_repetition(self, board: ChessBoard, features: torch.Tensor):
        """
        编码重复局面信息
        
        Args:
            board: 棋盘状态
            features: 特征张量
        """
        # 通道14: 当前局面是否重复过1次
        # 通道15: 当前局面是否重复过2次或以上
        
        current_hash = board.get_board_hash()
        repetition_count = board.metadata.get('repetition_count', {}).get(current_hash, 0)
        
        if repetition_count >= 1:
            features[14, :, :] = 1.0
        
        if repetition_count >= 2:
            features[15, :, :] = 1.0
    
    def _encode_current_player(self, board: ChessBoard, features: torch.Tensor):
        """
        编码当前玩家
        
        Args:
            board: 棋盘状态
            features: 特征张量
        """
        # 通道16: 当前玩家 (红方=1, 黑方=0)
        if board.current_player == 1:  # 红方
            features[16, :, :] = 1.0
        # 黑方时保持为0
    
    def _encode_no_capture_count(self, board: ChessBoard, features: torch.Tensor):
        """
        编码无吃子步数
        
        Args:
            board: 棋盘状态
            features: 特征张量
        """
        # 通道17: 无吃子步数 (归一化到[0,1])
        current_round = board.metadata.get('round_count', 0)
        last_capture_round = board.metadata.get('last_capture_round', 0)
        no_capture_count = current_round - last_capture_round
        
        # 归一化 (假设最大无吃子步数为100)
        normalized_count = min(no_capture_count / 100.0, 1.0)
        features[17, :, :] = normalized_count
    
    def _encode_king_positions(self, board: ChessBoard, features: torch.Tensor):
        """
        编码王的位置
        
        Args:
            board: 棋盘状态
            features: 特征张量
        """
        # 通道18: 红方帅的位置
        red_king_pos = board.find_king(1)
        if red_king_pos:
            row, col = red_king_pos
            features[18, row, col] = 1.0
        
        # 通道19: 黑方将的位置
        black_king_pos = board.find_king(-1)
        if black_king_pos:
            row, col = black_king_pos
            features[19, row, col] = 1.0
    
    def encode_board_with_history(
        self,
        board: ChessBoard,
        history_boards: Optional[List[ChessBoard]] = None
    ) -> torch.Tensor:
        """
        编码带历史信息的棋盘状态
        
        Args:
            board: 当前棋盘状态
            history_boards: 历史棋盘状态列表
            
        Returns:
            torch.Tensor: 编码后的张量 [20*history_length, 10, 9]
        """
        if history_boards is None:
            history_boards = []
        
        # 确保历史长度
        all_boards = history_boards[-self.history_length+1:] + [board]
        
        # 如果历史不足，用当前棋盘填充
        while len(all_boards) < self.history_length:
            all_boards.insert(0, board)
        
        # 编码每个历史状态
        encoded_features = []
        for hist_board in all_boards:
            features = self.encode_board(hist_board)
            encoded_features.append(features)
        
        # 拼接历史特征
        return torch.cat(encoded_features, dim=0)  # [20*history_length, 10, 9]
    
    def encode_move_to_policy_index(self, move: Move) -> int:
        """
        将走法编码为策略向量的索引
        
        Args:
            move: 走法
            
        Returns:
            int: 策略向量索引 (0-8099)
        """
        from_row, from_col = move.from_pos
        to_row, to_col = move.to_pos
        
        # 计算移动方向
        row_diff = to_row - from_row
        col_diff = to_col - from_col
        
        # 将移动方向映射到90个可能的方向之一
        direction_index = self._get_direction_index(row_diff, col_diff)
        
        # 计算最终索引: from_position * 90 + direction
        policy_index = from_row * 9 * 90 + from_col * 90 + direction_index
        
        return policy_index
    
    def _get_direction_index(self, row_diff: int, col_diff: int) -> int:
        """
        获取移动方向的索引
        
        Args:
            row_diff: 行差值
            col_diff: 列差值
            
        Returns:
            int: 方向索引 (0-89)
        """
        # 简化版本：将所有可能的移动方向映射到0-89
        # 这里使用一个简单的映射方案
        
        # 限制移动范围
        row_diff = max(-9, min(9, row_diff))
        col_diff = max(-8, min(8, col_diff))
        
        # 映射到索引 (这是一个简化的映射，实际实现可能需要更复杂的逻辑)
        direction_index = (row_diff + 9) * 17 + (col_diff + 8)
        
        # 确保索引在有效范围内
        return min(direction_index, 89)
    
    def decode_policy_index_to_move(
        self,
        policy_index: int,
        board: ChessBoard
    ) -> Optional[Move]:
        """
        将策略索引解码为走法
        
        Args:
            policy_index: 策略索引
            board: 当前棋盘状态
            
        Returns:
            Optional[Move]: 解码的走法，如果无效则返回None
        """
        # 解码起始位置和方向
        from_row = policy_index // (9 * 90)
        remaining = policy_index % (9 * 90)
        from_col = remaining // 90
        direction_index = remaining % 90
        
        # 解码移动方向
        row_diff, col_diff = self._decode_direction_index(direction_index)
        
        # 计算目标位置
        to_row = from_row + row_diff
        to_col = from_col + col_diff
        
        # 检查位置有效性
        if not (0 <= to_row <= 9 and 0 <= to_col <= 8):
            return None
        
        # 检查起始位置是否有棋子
        piece = board.get_piece_at((from_row, from_col))
        if piece == 0:
            return None
        
        # 创建走法
        move = Move(
            from_pos=(from_row, from_col),
            to_pos=(to_row, to_col),
            piece=piece
        )
        
        return move
    
    def _decode_direction_index(self, direction_index: int) -> Tuple[int, int]:
        """
        解码方向索引为行列差值
        
        Args:
            direction_index: 方向索引
            
        Returns:
            Tuple[int, int]: (行差值, 列差值)
        """
        # 与_get_direction_index相反的操作
        row_diff = (direction_index // 17) - 9
        col_diff = (direction_index % 17) - 8
        
        return row_diff, col_diff
    
    def create_policy_target(
        self,
        legal_moves: List[Move],
        move_probabilities: Dict[Move, float]
    ) -> np.ndarray:
        """
        创建策略目标向量
        
        Args:
            legal_moves: 合法走法列表
            move_probabilities: 走法概率字典
            
        Returns:
            np.ndarray: 策略目标向量 [8100]
        """
        policy_target = np.zeros(8100, dtype=np.float32)
        
        for move in legal_moves:
            policy_index = self.encode_move_to_policy_index(move)
            probability = move_probabilities.get(move, 0.0)
            
            # 确保索引在有效范围内
            if 0 <= policy_index < 8100:
                policy_target[policy_index] = probability
        
        # 归一化
        total_prob = policy_target.sum()
        if total_prob > 0:
            policy_target /= total_prob
        
        return policy_target
    
    def extract_legal_move_probabilities(
        self,
        policy_output: np.ndarray,
        legal_moves: List[Move]
    ) -> Dict[Move, float]:
        """
        从策略输出中提取合法走法的概率
        
        Args:
            policy_output: 神经网络策略输出 [8100]
            legal_moves: 合法走法列表
            
        Returns:
            Dict[Move, float]: 走法概率字典
        """
        move_probabilities = {}
        
        for move in legal_moves:
            policy_index = self.encode_move_to_policy_index(move)
            
            if 0 <= policy_index < 8100:
                probability = policy_output[policy_index]
                move_probabilities[move] = probability
        
        # 归一化合法走法的概率
        total_prob = sum(move_probabilities.values())
        if total_prob > 0:
            for move in move_probabilities:
                move_probabilities[move] /= total_prob
        
        return move_probabilities
    
    def get_feature_info(self) -> Dict[str, str]:
        """
        获取特征通道信息
        
        Returns:
            Dict[str, str]: 通道信息字典
        """
        return {
            '0-6': '红方棋子 (帅仕相马车炮兵)',
            '7-13': '黑方棋子 (将士象马车炮卒)',
            '14': '局面重复1次',
            '15': '局面重复2次以上',
            '16': '当前玩家 (红方=1)',
            '17': '无吃子步数 (归一化)',
            '18': '红方帅位置',
            '19': '黑方将位置'
        }
    
    def visualize_features(self, features: torch.Tensor, channel: int) -> str:
        """
        可视化特征通道
        
        Args:
            features: 特征张量
            channel: 要可视化的通道
            
        Returns:
            str: 可视化字符串
        """
        if channel >= features.shape[0]:
            return f"通道{channel}不存在"
        
        channel_data = features[channel].numpy()
        
        lines = [f"通道 {channel}:"]
        for row in range(10):
            line = ""
            for col in range(9):
                value = channel_data[row, col]
                if value > 0.5:
                    line += "■ "
                elif value > 0.1:
                    line += "□ "
                else:
                    line += "  "
            lines.append(line)
        
        return "\n".join(lines)