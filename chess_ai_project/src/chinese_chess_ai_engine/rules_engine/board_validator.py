"""
棋局合法性验证器

提供全面的棋局状态验证功能。
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .chess_board import ChessBoard
from .move import Move


class BoardValidator:
    """
    棋局合法性验证器
    
    提供各种棋局状态和走法的验证功能。
    """
    
    def __init__(self):
        """初始化验证器"""
        # 棋子数量限制
        self.piece_limits = {
            1: 1, -1: 1,    # 帅/将
            2: 2, -2: 2,    # 仕/士
            3: 2, -3: 2,    # 相/象
            4: 2, -4: 2,    # 马
            5: 2, -5: 2,    # 车
            6: 2, -6: 2,    # 炮
            7: 5, -7: 5     # 兵/卒
        }
        
        # 棋子名称
        self.piece_names = {
            0: "空", 1: "帅", 2: "仕", 3: "相", 4: "马", 5: "车", 6: "炮", 7: "兵",
            -1: "将", -2: "士", -3: "象", -4: "马", -5: "车", -6: "炮", -7: "卒"
        }
    
    def validate_board_structure(self, board: ChessBoard) -> Tuple[bool, List[str]]:
        """
        验证棋盘基本结构
        
        Args:
            board: 要验证的棋盘
            
        Returns:
            Tuple[bool, List[str]]: (是否合法, 错误信息列表)
        """
        errors = []
        
        # 检查棋盘尺寸
        if board.board.shape != (10, 9):
            errors.append(f"棋盘尺寸错误: {board.board.shape}, 应为(10, 9)")
        
        # 检查数据类型
        if board.board.dtype != np.int32 and board.board.dtype != int:
            errors.append(f"棋盘数据类型错误: {board.board.dtype}, 应为int")
        
        # 检查当前玩家
        if board.current_player not in [1, -1]:
            errors.append(f"当前玩家值错误: {board.current_player}, 应为1或-1")
        
        return len(errors) == 0, errors
    
    def validate_piece_counts(self, board: ChessBoard) -> Tuple[bool, List[str]]:
        """
        验证棋子数量
        
        Args:
            board: 要验证的棋盘
            
        Returns:
            Tuple[bool, List[str]]: (是否合法, 错误信息列表)
        """
        errors = []
        
        # 统计棋子数量
        piece_counts = {}
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece != 0:
                    piece_counts[piece] = piece_counts.get(piece, 0) + 1
        
        # 检查数量限制
        for piece, limit in self.piece_limits.items():
            count = piece_counts.get(piece, 0)
            if count > limit:
                piece_name = self.piece_names[piece]
                errors.append(f"{piece_name}数量超限: {count} > {limit}")
        
        # 检查必须存在的棋子
        required_pieces = [1, -1]  # 帅和将
        for piece in required_pieces:
            if piece_counts.get(piece, 0) != 1:
                piece_name = self.piece_names[piece]
                count = piece_counts.get(piece, 0)
                errors.append(f"{piece_name}数量错误: {count}, 应为1")
        
        return len(errors) == 0, errors
    
    def validate_piece_positions(self, board: ChessBoard) -> Tuple[bool, List[str]]:
        """
        验证棋子位置的合法性
        
        Args:
            board: 要验证的棋盘
            
        Returns:
            Tuple[bool, List[str]]: (是否合法, 错误信息列表)
        """
        errors = []
        
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece == 0:
                    continue
                
                # 验证帅/将的位置
                if piece == 1:  # 红帅
                    if not (7 <= row <= 9 and 3 <= col <= 5):
                        errors.append(f"红帅位置错误: ({row}, {col}), 应在九宫内")
                elif piece == -1:  # 黑将
                    if not (0 <= row <= 2 and 3 <= col <= 5):
                        errors.append(f"黑将位置错误: ({row}, {col}), 应在九宫内")
                
                # 验证仕/士的位置
                elif piece == 2:  # 红仕
                    if not (7 <= row <= 9 and 3 <= col <= 5):
                        errors.append(f"红仕位置错误: ({row}, {col}), 应在九宫内")
                elif piece == -2:  # 黑士
                    if not (0 <= row <= 2 and 3 <= col <= 5):
                        errors.append(f"黑士位置错误: ({row}, {col}), 应在九宫内")
                
                # 验证相/象的位置（不能过河）
                elif piece == 3:  # 红相
                    if row < 5:
                        errors.append(f"红相过河: ({row}, {col})")
                elif piece == -3:  # 黑象
                    if row > 4:
                        errors.append(f"黑象过河: ({row}, {col})")
        
        return len(errors) == 0, errors
    
    def validate_kings_facing(self, board: ChessBoard) -> Tuple[bool, List[str]]:
        """
        验证帅将是否照面
        
        Args:
            board: 要验证的棋盘
            
        Returns:
            Tuple[bool, List[str]]: (是否合法, 错误信息列表)
        """
        errors = []
        
        # 找到帅和将的位置
        red_king_pos = board.find_king(1)
        black_king_pos = board.find_king(-1)
        
        if red_king_pos and black_king_pos:
            red_row, red_col = red_king_pos
            black_row, black_col = black_king_pos
            
            # 如果在同一列，检查中间是否有棋子
            if red_col == black_col:
                # 检查两王之间是否有棋子
                has_piece_between = False
                start_row = min(red_row, black_row) + 1
                end_row = max(red_row, black_row)
                
                for row in range(start_row, end_row):
                    if board.board[row, red_col] != 0:
                        has_piece_between = True
                        break
                
                if not has_piece_between:
                    errors.append("帅将照面，中间无棋子阻挡")
        
        return len(errors) == 0, errors
    
    def validate_move_history(self, board: ChessBoard) -> Tuple[bool, List[str]]:
        """
        验证走法历史的一致性
        
        Args:
            board: 要验证的棋盘
            
        Returns:
            Tuple[bool, List[str]]: (是否合法, 错误信息列表)
        """
        errors = []
        
        # 检查历史长度一致性
        expected_history_length = len(board.move_history) + 1
        actual_history_length = len(board.board_history)
        
        if expected_history_length != actual_history_length:
            errors.append(f"历史记录长度不一致: 走法{len(board.move_history)}, "
                         f"棋局{actual_history_length}, 应为{expected_history_length}")
        
        # 验证走法序列的合法性
        if board.move_history:
            # 从初始状态重放所有走法
            try:
                temp_board = ChessBoard()
                for i, move in enumerate(board.move_history):
                    # 检查走法的基本有效性
                    if not self._is_valid_move_format(move):
                        errors.append(f"第{i+1}步走法格式错误: {move}")
                        continue
                    
                    # 执行走法
                    temp_board = temp_board.make_move(move)
                
                # 比较最终状态
                if not np.array_equal(temp_board.board, board.board):
                    errors.append("重放走法后的棋局状态与当前状态不一致")
                
                if temp_board.current_player != board.current_player:
                    errors.append("重放走法后的当前玩家与实际不一致")
                    
            except Exception as e:
                errors.append(f"重放走法时出错: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _is_valid_move_format(self, move: Move) -> bool:
        """
        检查走法格式是否有效
        
        Args:
            move: 要检查的走法
            
        Returns:
            bool: 格式是否有效
        """
        try:
            # 检查位置坐标
            from_row, from_col = move.from_pos
            to_row, to_col = move.to_pos
            
            # 坐标范围检查
            if not (0 <= from_row <= 9 and 0 <= from_col <= 8):
                return False
            if not (0 <= to_row <= 9 and 0 <= to_col <= 8):
                return False
            
            # 检查棋子类型
            if move.piece == 0:
                return False
            
            # 检查起始位置和目标位置不同
            if move.from_pos == move.to_pos:
                return False
            
            return True
            
        except (TypeError, ValueError):
            return False
    
    def validate_game_rules(self, board: ChessBoard) -> Tuple[bool, List[str]]:
        """
        验证游戏规则相关的状态
        
        Args:
            board: 要验证的棋盘
            
        Returns:
            Tuple[bool, List[str]]: (是否合法, 错误信息列表)
        """
        errors = []
        
        # 检查是否有无子可动的情况（困毙）
        # 这里可以添加更复杂的规则检查
        
        # 检查重复局面
        if board.metadata.get('repetition_count'):
            max_repetitions = max(board.metadata['repetition_count'].values())
            if max_repetitions >= 3:
                errors.append(f"局面重复次数过多: {max_repetitions}")
        
        # 检查无吃子步数（50步规则的变体）
        moves_since_capture = (board.metadata.get('round_count', 0) - 
                              board.metadata.get('last_capture_round', 0))
        if moves_since_capture > 120:  # 60回合无吃子
            errors.append(f"无吃子步数过多: {moves_since_capture}")
        
        return len(errors) == 0, errors
    
    def full_validation(self, board: ChessBoard) -> Tuple[bool, List[str]]:
        """
        完整的棋局验证
        
        Args:
            board: 要验证的棋盘
            
        Returns:
            Tuple[bool, List[str]]: (是否合法, 所有错误信息列表)
        """
        all_errors = []
        
        # 依次进行各项验证
        validations = [
            self.validate_board_structure,
            self.validate_piece_counts,
            self.validate_piece_positions,
            self.validate_kings_facing,
            self.validate_move_history,
            self.validate_game_rules
        ]
        
        for validation_func in validations:
            is_valid, errors = validation_func(board)
            all_errors.extend(errors)
        
        return len(all_errors) == 0, all_errors
    
    def get_validation_report(self, board: ChessBoard) -> Dict[str, Any]:
        """
        获取详细的验证报告
        
        Args:
            board: 要验证的棋盘
            
        Returns:
            Dict[str, Any]: 验证报告
        """
        report = {
            'overall_valid': True,
            'total_errors': 0,
            'validations': {}
        }
        
        # 各项验证
        validation_tests = {
            'structure': self.validate_board_structure,
            'piece_counts': self.validate_piece_counts,
            'piece_positions': self.validate_piece_positions,
            'kings_facing': self.validate_kings_facing,
            'move_history': self.validate_move_history,
            'game_rules': self.validate_game_rules
        }
        
        for test_name, test_func in validation_tests.items():
            is_valid, errors = test_func(board)
            report['validations'][test_name] = {
                'valid': is_valid,
                'errors': errors,
                'error_count': len(errors)
            }
            
            if not is_valid:
                report['overall_valid'] = False
                report['total_errors'] += len(errors)
        
        return report