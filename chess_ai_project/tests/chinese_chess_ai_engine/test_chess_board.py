"""
测试ChessBoard类的功能

测试棋局表示、序列化、走法历史等功能。
"""

import pytest
import numpy as np
import json
import tempfile
import os
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import ChessBoard, Move, BoardValidator


class TestChessBoard:
    """ChessBoard类的测试"""
    
    def test_initial_board_setup(self):
        """测试初始棋局设置"""
        board = ChessBoard()
        
        # 检查棋盘尺寸
        assert board.board.shape == (10, 9)
        
        # 检查初始玩家
        assert board.current_player == 1
        
        # 检查帅和将的位置
        assert board.find_king(1) == (9, 4)  # 红帅
        assert board.find_king(-1) == (0, 4)  # 黑将
        
        # 检查初始状态验证
        is_valid, errors = board.validate_board_state()
        assert is_valid, f"初始棋局应该是合法的，但发现错误: {errors}"
    
    def test_matrix_conversion(self):
        """测试矩阵格式转换"""
        board = ChessBoard()
        matrix = board.to_matrix()
        
        # 检查矩阵类型和形状
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (10, 9)
        
        # 从矩阵创建新棋盘
        new_board = ChessBoard.from_matrix(matrix)
        assert np.array_equal(board.board, new_board.board)
        assert board.current_player == new_board.current_player
    
    def test_fen_conversion(self):
        """测试FEN格式转换"""
        board = ChessBoard()
        fen = board.to_fen()
        
        # 检查FEN格式基本结构
        assert isinstance(fen, str)
        parts = fen.split()
        assert len(parts) >= 2
        
        # 从FEN创建新棋盘
        new_board = ChessBoard(fen=fen)
        assert np.array_equal(board.board, new_board.board)
        assert board.current_player == new_board.current_player
    
    def test_json_serialization(self):
        """测试JSON序列化"""
        board = ChessBoard()
        
        # 执行一些走法
        move1 = Move(from_pos=(6, 0), to_pos=(5, 0), piece=7)  # 兵前进
        board = board.make_move(move1)
        
        # 序列化为JSON
        json_str = board.to_json()
        assert isinstance(json_str, str)
        
        # 验证JSON格式
        data = json.loads(json_str)
        assert 'board' in data
        assert 'current_player' in data
        assert 'move_history' in data
        
        # 从JSON反序列化
        new_board = ChessBoard.from_json(json_str)
        assert np.array_equal(board.board, new_board.board)
        assert board.current_player == new_board.current_player
        assert len(board.move_history) == len(new_board.move_history)
    
    def test_binary_serialization(self):
        """测试二进制序列化"""
        board = ChessBoard()
        
        # 执行一些走法
        move1 = Move(from_pos=(6, 0), to_pos=(5, 0), piece=7)
        board = board.make_move(move1)
        
        # 序列化为二进制
        binary_data = board.to_binary()
        assert isinstance(binary_data, bytes)
        
        # 从二进制反序列化
        new_board = ChessBoard.from_binary(binary_data)
        assert np.array_equal(board.board, new_board.board)
        assert board.current_player == new_board.current_player
        assert len(board.move_history) == len(new_board.move_history)
    
    def test_file_operations(self):
        """测试文件保存和加载"""
        board = ChessBoard()
        
        # 执行一些走法
        move1 = Move(from_pos=(6, 0), to_pos=(5, 0), piece=7)
        board = board.make_move(move1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试JSON格式
            json_file = os.path.join(temp_dir, "test.json")
            board.save_to_file(json_file, format='json')
            loaded_board = ChessBoard.load_from_file(json_file, format='json')
            assert np.array_equal(board.board, loaded_board.board)
            
            # 测试二进制格式
            bin_file = os.path.join(temp_dir, "test.bin")
            board.save_to_file(bin_file, format='binary')
            loaded_board = ChessBoard.load_from_file(bin_file, format='binary')
            assert np.array_equal(board.board, loaded_board.board)
            
            # 测试FEN格式
            fen_file = os.path.join(temp_dir, "test.fen")
            board.save_to_file(fen_file, format='fen')
            loaded_board = ChessBoard.load_from_file(fen_file, format='fen')
            assert np.array_equal(board.board, loaded_board.board)
            
            # 测试自动格式检测
            auto_board = ChessBoard.load_from_file(json_file, format='auto')
            assert np.array_equal(board.board, auto_board.board)
    
    def test_move_operations(self):
        """测试走法操作"""
        board = ChessBoard()
        initial_board = board.copy()
        
        # 执行走法
        move1 = Move(from_pos=(6, 0), to_pos=(5, 0), piece=7)  # 兵前进
        board = board.make_move(move1)
        
        # 检查走法历史
        assert len(board.move_history) == 1
        assert board.get_last_move() == move1
        assert board.get_move_count() == 1
        
        # 检查玩家切换
        assert board.current_player == -1
        
        # 撤销走法
        board = board.undo_move()
        assert len(board.move_history) == 0
        assert board.current_player == 1
        assert np.array_equal(board.board, initial_board.board)
    
    def test_piece_operations(self):
        """测试棋子操作"""
        board = ChessBoard()
        
        # 测试获取棋子
        piece = board.get_piece_at((9, 4))
        assert piece == 1  # 红帅
        
        # 测试位置检查
        assert not board.is_empty((9, 4))
        assert board.is_empty((5, 5))
        
        # 测试己方/敌方棋子判断
        assert board.is_own_piece((9, 4), 1)  # 红帅对红方
        assert not board.is_own_piece((9, 4), -1)  # 红帅对黑方
        assert board.is_enemy_piece((0, 4), 1)  # 黑将对红方
        
        # 测试获取所有棋子
        all_pieces = board.get_all_pieces()
        assert len(all_pieces) == 32  # 初始局面有32个棋子
        
        red_pieces = board.get_all_pieces(player=1)
        black_pieces = board.get_all_pieces(player=-1)
        assert len(red_pieces) == 16
        assert len(black_pieces) == 16
    
    def test_board_validation(self):
        """测试棋局验证"""
        board = ChessBoard()
        
        # 正常棋局应该通过验证
        is_valid, errors = board.validate_board_state()
        assert is_valid, f"初始棋局验证失败: {errors}"
        
        # 获取详细验证报告
        report = board.get_validation_report()
        assert report['overall_valid']
        assert report['total_errors'] == 0
        
        # 测试无效棋局
        invalid_board = ChessBoard()
        invalid_board.board[9, 4] = 0  # 移除红帅
        is_valid, errors = invalid_board.validate_board_state()
        assert not is_valid
        assert len(errors) > 0
    
    def test_board_hash_and_repetition(self):
        """测试棋局哈希和重复检测"""
        board = ChessBoard()
        
        # 获取初始哈希
        initial_hash = board.get_board_hash()
        assert isinstance(initial_hash, str)
        assert len(initial_hash) == 32  # MD5哈希长度
        
        # 执行走法后哈希应该改变
        move1 = Move(from_pos=(6, 0), to_pos=(5, 0), piece=7)
        new_board = board.make_move(move1)
        new_hash = new_board.get_board_hash()
        assert new_hash != initial_hash
        
        # 测试重复检测
        assert not board.is_repetition()
    
    def test_material_value(self):
        """测试棋子价值计算"""
        board = ChessBoard()
        
        # 计算双方棋子价值
        red_value = board.get_material_value(1)
        black_value = board.get_material_value(-1)
        
        # 初始局面双方价值应该相等
        assert red_value == black_value
        assert red_value > 0
    
    def test_copy_and_equality(self):
        """测试拷贝和相等性"""
        board = ChessBoard()
        
        # 测试拷贝
        board_copy = board.copy()
        assert board == board_copy
        assert board is not board_copy
        assert np.array_equal(board.board, board_copy.board)
        
        # 修改拷贝不应影响原棋盘
        move1 = Move(from_pos=(6, 0), to_pos=(5, 0), piece=7)
        board_copy = board_copy.make_move(move1)
        assert board != board_copy
    
    def test_visual_representation(self):
        """测试可视化表示"""
        board = ChessBoard()
        
        # 测试字符串表示
        visual_str = board.to_visual_string()
        assert isinstance(visual_str, str)
        assert "帅" in visual_str
        assert "将" in visual_str
        
        # 测试__str__方法
        str_repr = str(board)
        assert str_repr == visual_str


class TestBoardValidator:
    """BoardValidator类的测试"""
    
    def test_validator_creation(self):
        """测试验证器创建"""
        validator = BoardValidator()
        assert validator is not None
        assert hasattr(validator, 'piece_limits')
        assert hasattr(validator, 'piece_names')
    
    def test_structure_validation(self):
        """测试结构验证"""
        validator = BoardValidator()
        board = ChessBoard()
        
        is_valid, errors = validator.validate_board_structure(board)
        assert is_valid
        assert len(errors) == 0
    
    def test_piece_count_validation(self):
        """测试棋子数量验证"""
        validator = BoardValidator()
        board = ChessBoard()
        
        is_valid, errors = validator.validate_piece_counts(board)
        assert is_valid
        assert len(errors) == 0
        
        # 测试无效数量
        board.board[5, 5] = 1  # 添加额外的帅
        is_valid, errors = validator.validate_piece_counts(board)
        assert not is_valid
        assert len(errors) > 0
    
    def test_position_validation(self):
        """测试位置验证"""
        validator = BoardValidator()
        board = ChessBoard()
        
        is_valid, errors = validator.validate_piece_positions(board)
        assert is_valid
        assert len(errors) == 0
        
        # 测试无效位置（帅出九宫）
        board.board[9, 4] = 0  # 移除原位置的帅
        board.board[9, 0] = 1  # 将帅放到九宫外
        is_valid, errors = validator.validate_piece_positions(board)
        assert not is_valid
        assert len(errors) > 0
    
    def test_full_validation(self):
        """测试完整验证"""
        validator = BoardValidator()
        board = ChessBoard()
        
        is_valid, errors = validator.full_validation(board)
        assert is_valid
        assert len(errors) == 0
        
        # 获取验证报告
        report = validator.get_validation_report(board)
        assert report['overall_valid']
        assert report['total_errors'] == 0
        assert 'validations' in report


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])