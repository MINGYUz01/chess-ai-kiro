"""
核心数据模型测试

测试Move类和ChessBoard类的基本功能。
"""

import pytest
import numpy as np
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine.move import Move
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine.chess_board import ChessBoard


class TestMove:
    """测试Move类"""
    
    def test_move_creation(self):
        """测试Move对象创建"""
        move = Move(
            from_pos=(9, 4),
            to_pos=(8, 4),
            piece=ChessBoard.RED_KING
        )
        
        assert move.from_pos == (9, 4)
        assert move.to_pos == (8, 4)
        assert move.piece == ChessBoard.RED_KING
        assert move.captured_piece is None
        assert not move.is_check
        assert not move.is_checkmate
    
    def test_move_validation(self):
        """测试Move位置验证"""
        # 有效位置
        move = Move((0, 0), (9, 8), ChessBoard.RED_PAWN)
        assert move.from_pos == (0, 0)
        
        # 无效位置应该抛出异常
        with pytest.raises(ValueError):
            Move((-1, 0), (0, 0), ChessBoard.RED_PAWN)
        
        with pytest.raises(ValueError):
            Move((0, 0), (10, 0), ChessBoard.RED_PAWN)
        
        with pytest.raises(ValueError):
            Move((0, 0), (0, 9), ChessBoard.RED_PAWN)
    
    def test_coordinate_notation(self):
        """测试坐标记法转换"""
        move = Move((0, 0), (1, 1), ChessBoard.RED_ROOK)
        coord_notation = move.to_coordinate_notation()
        assert coord_notation == "a0b1"
        
        # 测试从坐标记法创建Move
        move2 = Move.from_coordinate_notation("a0b1", ChessBoard.RED_ROOK)
        assert move2.from_pos == (0, 0)
        assert move2.to_pos == (1, 1)
        assert move2.piece == ChessBoard.RED_ROOK
    
    def test_chinese_notation(self):
        """测试中文记法转换"""
        move = Move((9, 4), (8, 4), ChessBoard.RED_KING)
        chinese_notation = move.to_chinese_notation()
        # 帅五进一
        assert "帅" in chinese_notation
        assert "进" in chinese_notation
    
    def test_move_equality(self):
        """测试Move相等性比较"""
        move1 = Move((0, 0), (1, 1), ChessBoard.RED_ROOK)
        move2 = Move((0, 0), (1, 1), ChessBoard.RED_ROOK)
        move3 = Move((0, 0), (1, 2), ChessBoard.RED_ROOK)
        
        assert move1 == move2
        assert move1 != move3
        assert hash(move1) == hash(move2)
        assert hash(move1) != hash(move3)


class TestChessBoard:
    """测试ChessBoard类"""
    
    def test_initial_board(self):
        """测试初始棋盘设置"""
        board = ChessBoard()
        
        # 检查棋盘大小
        assert board.board.shape == (10, 9)
        
        # 检查初始玩家
        assert board.current_player == 1
        
        # 检查一些关键位置的棋子
        assert board.board[9, 4] == ChessBoard.RED_KING      # 红帅
        assert board.board[0, 4] == ChessBoard.BLACK_KING    # 黑将
        assert board.board[9, 0] == ChessBoard.RED_ROOK      # 红车
        assert board.board[0, 0] == ChessBoard.BLACK_ROOK    # 黑车
    
    def test_matrix_conversion(self):
        """测试矩阵转换"""
        board = ChessBoard()
        matrix = board.to_matrix()
        
        # 检查矩阵类型和形状
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (10, 9)
        
        # 从矩阵创建新棋盘
        board2 = ChessBoard.from_matrix(matrix)
        assert np.array_equal(board.board, board2.board)
        assert board.current_player == board2.current_player
    
    def test_fen_conversion(self):
        """测试FEN格式转换"""
        board = ChessBoard()
        fen = board.to_fen()
        
        # 检查FEN格式基本结构
        assert isinstance(fen, str)
        assert "/" in fen  # 行分隔符
        assert " w " in fen or " b " in fen  # 玩家标识
        
        # 从FEN创建新棋盘
        board2 = ChessBoard()
        board2.from_fen(fen)
        
        # 应该得到相同的棋盘
        assert np.array_equal(board.board, board2.board)
        assert board.current_player == board2.current_player
    
    def test_piece_operations(self):
        """测试棋子操作"""
        board = ChessBoard()
        
        # 测试获取棋子
        piece = board.get_piece_at((9, 4))
        assert piece == ChessBoard.RED_KING
        
        # 测试空位置
        assert board.is_empty((5, 5))
        assert not board.is_empty((9, 4))
        
        # 测试己方/敌方棋子判断
        assert board.is_own_piece((9, 4), 1)  # 红方帅对红方玩家
        assert not board.is_own_piece((9, 4), -1)  # 红方帅对黑方玩家
        assert board.is_enemy_piece((0, 4), 1)  # 黑方将对红方玩家
    
    def test_find_king(self):
        """测试寻找王的位置"""
        board = ChessBoard()
        
        red_king_pos = board.find_king(1)
        assert red_king_pos == (9, 4)
        
        black_king_pos = board.find_king(-1)
        assert black_king_pos == (0, 4)
    
    def test_make_move(self):
        """测试执行走法"""
        board = ChessBoard()
        
        # 创建一个简单的走法（兵前进）
        move = Move((6, 0), (5, 0), ChessBoard.RED_PAWN)
        new_board = board.make_move(move)
        
        # 检查原棋盘没有改变
        assert board.board[6, 0] == ChessBoard.RED_PAWN
        assert board.board[5, 0] == ChessBoard.EMPTY
        
        # 检查新棋盘状态
        assert new_board.board[6, 0] == ChessBoard.EMPTY
        assert new_board.board[5, 0] == ChessBoard.RED_PAWN
        
        # 检查玩家切换
        assert new_board.current_player == -1
        
        # 检查历史记录
        assert len(new_board.move_history) == 1
        assert new_board.move_history[0] == move
    
    def test_undo_move(self):
        """测试撤销走法"""
        board = ChessBoard()
        
        # 执行一个走法
        move = Move((6, 0), (5, 0), ChessBoard.RED_PAWN)
        board_after_move = board.make_move(move)
        
        # 撤销走法
        board_after_undo = board_after_move.undo_move()
        
        # 应该回到原始状态
        assert np.array_equal(board.board, board_after_undo.board)
        assert board.current_player == board_after_undo.current_player
        assert len(board_after_undo.move_history) == 0
    
    def test_json_serialization(self):
        """测试JSON序列化"""
        board = ChessBoard()
        
        # 执行一些走法
        move1 = Move((6, 0), (5, 0), ChessBoard.RED_PAWN)
        board = board.make_move(move1)
        
        # 转换为JSON
        json_str = board.to_json()
        assert isinstance(json_str, str)
        
        # 从JSON恢复
        board2 = ChessBoard.from_json(json_str)
        
        # 检查状态一致性
        assert np.array_equal(board.board, board2.board)
        assert board.current_player == board2.current_player
        assert len(board.move_history) == len(board2.move_history)
    
    def test_visual_string(self):
        """测试可视化字符串"""
        board = ChessBoard()
        visual = board.to_visual_string()
        
        assert isinstance(visual, str)
        assert "帅" in visual  # 应该包含红帅
        assert "将" in visual  # 应该包含黑将
        assert "当前玩家" in visual  # 应该显示当前玩家
    
    def test_board_equality(self):
        """测试棋盘相等性比较"""
        board1 = ChessBoard()
        board2 = ChessBoard()
        
        assert board1 == board2
        
        # 执行不同的走法
        move = Move((6, 0), (5, 0), ChessBoard.RED_PAWN)
        board1_moved = board1.make_move(move)
        
        assert board1_moved != board2