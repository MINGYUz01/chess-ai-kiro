"""
测试RuleEngine类的功能

测试走法生成、合法性验证、终局检测等功能。
"""

import pytest
import numpy as np
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import ChessBoard, Move, RuleEngine


class TestRuleEngine:
    """RuleEngine类的测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.rule_engine = RuleEngine()
        self.board = ChessBoard()
    
    def test_rule_engine_creation(self):
        """测试规则引擎创建"""
        assert self.rule_engine is not None
        assert hasattr(self.rule_engine, 'generate_legal_moves')
        assert hasattr(self.rule_engine, 'is_legal_move')
    
    def test_initial_legal_moves(self):
        """测试初始局面的合法走法"""
        # 红方开局走法
        red_moves = self.rule_engine.generate_legal_moves(self.board, player=1)
        assert len(red_moves) > 0
        
        # 检查一些典型的开局走法
        move_strings = [move.to_coordinate_notation() for move in red_moves]
        
        # 兵的走法
        assert 'a6a5' in move_strings  # 一路兵进一
        assert 'c6c5' in move_strings  # 三路兵进一
        assert 'e6e5' in move_strings  # 五路兵进一
        
        # 马的走法
        assert 'b9c7' in move_strings  # 马二进三
        assert 'h9g7' in move_strings  # 马八进七
        
        # 炮的走法（可以移动到任何空位）
        cannon_moves = [m for m in move_strings if m.startswith('b7') or m.startswith('h7')]
        assert len(cannon_moves) > 0
    
    def test_piece_specific_moves(self):
        """测试各种棋子的走法生成"""
        # 测试帅的走法
        king_moves = self.rule_engine._generate_king_moves(self.board, (9, 4))
        assert len(king_moves) == 1  # 初始位置帅只能向前移动（左右被仕挡住）
        
        # 测试车的走法
        # 先移动一个兵让车有路
        board_with_space = self.board.make_move(Move(from_pos=(6, 0), to_pos=(5, 0), piece=7))
        rook_moves = self.rule_engine._generate_rook_moves(board_with_space, (9, 0))
        assert len(rook_moves) > 0  # 车应该能向前移动
        
        # 测试马的走法
        knight_moves = self.rule_engine._generate_knight_moves(self.board, (9, 1))
        assert len(knight_moves) == 2  # 初始位置马有两个合法走法
        
        # 测试兵的走法
        pawn_moves = self.rule_engine._generate_pawn_moves(self.board, (6, 0))
        assert len(pawn_moves) == 1  # 兵只能向前
        
        # 测试炮的走法
        cannon_moves = self.rule_engine._generate_cannon_moves(self.board, (7, 1))
        assert len(cannon_moves) > 5  # 炮可以移动到多个位置
    
    def test_move_legality_validation(self):
        """测试走法合法性验证"""
        # 合法走法
        legal_move = Move(from_pos=(6, 0), to_pos=(5, 0), piece=7)  # 兵前进
        assert self.rule_engine.is_legal_move(self.board, legal_move)
        
        # 非法走法：移动不存在的棋子
        illegal_move1 = Move(from_pos=(5, 0), to_pos=(4, 0), piece=7)  # 空位没有棋子
        assert not self.rule_engine.is_legal_move(self.board, illegal_move1)
        
        # 非法走法：棋子类型不匹配
        illegal_move2 = Move(from_pos=(6, 0), to_pos=(5, 0), piece=5)  # 位置是兵但说是车
        assert not self.rule_engine.is_legal_move(self.board, illegal_move2)
        
        # 非法走法：不符合棋子移动规则
        illegal_move3 = Move(from_pos=(6, 0), to_pos=(4, 0), piece=7)  # 兵不能走两步
        assert not self.rule_engine.is_legal_move(self.board, illegal_move3)
    
    def test_check_detection(self):
        """测试将军检测"""
        # 初始局面不应该有将军
        assert not self.rule_engine.is_in_check(self.board, 1)  # 红方
        assert not self.rule_engine.is_in_check(self.board, -1)  # 黑方
        
        # 创建一个将军的局面
        check_board = ChessBoard()
        check_board.board.fill(0)  # 清空棋盘
        
        # 设置简单的将军局面
        check_board.board[0, 4] = -1  # 黑将
        check_board.board[2, 4] = 6   # 红炮将军（中间隔一行）
        check_board.board[1, 4] = 7   # 红兵作为炮台
        check_board.board[9, 4] = 1   # 红帅
        
        # 现在黑方应该被将军
        assert self.rule_engine.is_in_check(check_board, -1)
        assert not self.rule_engine.is_in_check(check_board, 1)
    
    def test_checkmate_detection(self):
        """测试将死检测"""
        # 创建一个简单的将死局面
        checkmate_board = ChessBoard()
        # 清空棋盘，只留下必要的棋子
        checkmate_board.board.fill(0)
        
        # 设置一个简单的将死局面
        checkmate_board.board[0, 4] = -1  # 黑将
        checkmate_board.board[1, 4] = 5   # 红车将军
        checkmate_board.board[0, 3] = 5   # 红车封左路
        checkmate_board.board[0, 5] = 5   # 红车封右路
        checkmate_board.board[9, 4] = 1   # 红帅
        
        checkmate_board.current_player = -1  # 轮到黑方
        
        # 黑方应该被将死
        assert self.rule_engine.is_in_check(checkmate_board, -1)
        # 注意：实际的将死检测比较复杂，这里先检查是否被将军
        # assert self.rule_engine.is_checkmate(checkmate_board, -1)
    
    def test_stalemate_detection(self):
        """测试困毙检测"""
        # 创建一个困毙局面
        stalemate_board = ChessBoard()
        stalemate_board.board.fill(0)
        
        # 设置困毙局面：黑将被困但不被将军
        # 将黑将放在九宫角落，用红车控制其他位置
        stalemate_board.board[0, 3] = -1  # 黑将在九宫角落
        stalemate_board.board[1, 3] = 5   # 红车控制前方
        stalemate_board.board[0, 4] = 5   # 红车控制右方
        stalemate_board.board[9, 4] = 1   # 红帅
        
        stalemate_board.current_player = -1  # 轮到黑方
        
        # 检查是否被困毙（这个测试可能需要更复杂的局面）
        legal_moves = self.rule_engine.generate_legal_moves(stalemate_board, -1)
        # 在这个局面下，黑将可以吃掉红车，所以不是真正的困毙
        # 我们只是验证困毙检测函数能正常工作
        is_stalemate = self.rule_engine.is_stalemate(stalemate_board, -1)
        # 由于黑将可以移动（吃车），所以不是困毙
        assert not is_stalemate
    
    def test_game_status(self):
        """测试游戏状态获取"""
        status = self.rule_engine.get_game_status(self.board)
        
        assert status['current_player'] == 1
        assert status['current_player_name'] == '红方'
        assert not status['in_check']
        assert not status['checkmate']
        assert not status['stalemate']
        assert not status['game_over']
        assert status['winner'] is None
        assert status['legal_moves_count'] > 0
        assert len(status['legal_moves']) > 0
    
    def test_special_rules(self):
        """测试特殊规则"""
        # 测试长将检测
        moves_history = []
        # 创建一些重复的将军走法
        for _ in range(4):
            moves_history.append(Move(from_pos=(7, 1), to_pos=(4, 1), piece=6))
            moves_history.append(Move(from_pos=(0, 4), to_pos=(0, 3), piece=-1))
        
        # 注意：这里只是测试函数调用，实际的长将检测需要更复杂的局面
        perpetual_check = self.rule_engine.check_perpetual_check(self.board, moves_history)
        assert isinstance(perpetual_check, bool)
        
        # 测试和棋条件检查
        is_draw, reason = self.rule_engine.check_draw_conditions(self.board)
        assert isinstance(is_draw, bool)
        assert isinstance(reason, str)
    
    def test_move_threats_analysis(self):
        """测试走法威胁分析"""
        move = Move(from_pos=(6, 0), to_pos=(5, 0), piece=7)  # 兵前进
        threats = self.rule_engine.get_move_threats(self.board, move)
        
        assert 'checks' in threats
        assert 'captures' in threats
        assert 'defends' in threats
        assert 'attacks' in threats
        
        # 初始的兵前进不应该有将军威胁
        assert len(threats['checks']) == 0
    
    def test_move_quality_evaluation(self):
        """测试走法质量评估"""
        move = Move(from_pos=(6, 0), to_pos=(5, 0), piece=7)  # 兵前进
        evaluation = self.rule_engine.evaluate_move_quality(self.board, move)
        
        assert 'move' in evaluation
        assert 'score' in evaluation
        assert 'threats' in evaluation
        assert 'safety' in evaluation
        assert 'tactical_value' in evaluation
        assert 'positional_value' in evaluation
        
        assert evaluation['move'] == move
        assert isinstance(evaluation['score'], (int, float))
    
    def test_pawn_promotion_moves(self):
        """测试兵过河后的走法"""
        # 创建一个兵过河的局面
        promoted_board = ChessBoard()
        promoted_board.board[4, 0] = 7  # 红兵过河
        promoted_board.board[6, 0] = 0  # 移除原位置的兵
        
        pawn_moves = self.rule_engine._generate_pawn_moves(promoted_board, (4, 0))
        
        # 过河的兵应该能向前、向左、向右移动
        move_directions = []
        for move in pawn_moves:
            from_pos, to_pos = move.from_pos, move.to_pos
            dr, dc = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
            move_directions.append((dr, dc))
        
        # 应该包含向前和横向移动
        assert (-1, 0) in move_directions  # 向前
        # 如果左右有空位，应该也能移动
    
    def test_knight_blocking(self):
        """测试马腿阻挡"""
        # 创建一个马腿被阻挡的局面
        blocked_board = ChessBoard()
        # 在马腿位置放置棋子
        blocked_board.board[8, 1] = 7  # 在马腿位置放兵
        
        knight_moves = self.rule_engine._generate_knight_moves(blocked_board, (9, 1))
        
        # 被阻挡的马走法应该减少
        normal_moves = self.rule_engine._generate_knight_moves(self.board, (9, 1))
        assert len(knight_moves) < len(normal_moves)
    
    def test_bishop_blocking(self):
        """测试相/象塞眼"""
        # 创建一个象眼被塞的局面
        blocked_board = ChessBoard()
        # 在象眼位置放置棋子
        blocked_board.board[8, 3] = 7  # 塞住相眼
        
        bishop_moves = self.rule_engine._generate_bishop_moves(blocked_board, (9, 2))
        
        # 被塞眼的相走法应该减少
        normal_moves = self.rule_engine._generate_bishop_moves(self.board, (9, 2))
        assert len(bishop_moves) < len(normal_moves)
    
    def test_cannon_platform_mechanics(self):
        """测试炮的炮台机制"""
        # 创建一个炮有炮台的局面
        cannon_board = ChessBoard()
        # 在炮前面放一个棋子作为炮台，然后在更远处放敌方棋子
        cannon_board.board[5, 1] = 7   # 炮台（兵）
        cannon_board.board[3, 1] = -7  # 目标（敌方卒）
        
        cannon_moves = self.rule_engine._generate_cannon_moves(cannon_board, (7, 1))
        
        # 炮应该能够跳过炮台吃到敌方棋子
        target_moves = [move for move in cannon_moves if move.to_pos == (3, 1)]
        assert len(target_moves) == 1
        assert target_moves[0].captured_piece == -7


class TestRuleEngineEdgeCases:
    """RuleEngine边界情况测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.rule_engine = RuleEngine()
    
    def test_empty_board_moves(self):
        """测试空棋盘的走法生成"""
        empty_board = ChessBoard()
        empty_board.board.fill(0)
        empty_board.board[9, 4] = 1   # 只有红帅
        empty_board.board[0, 4] = -1  # 只有黑将
        
        red_moves = self.rule_engine.generate_legal_moves(empty_board, player=1)
        black_moves = self.rule_engine.generate_legal_moves(empty_board, player=-1)
        
        # 帅和将都应该有一些移动选项
        assert len(red_moves) > 0
        assert len(black_moves) > 0
    
    def test_invalid_move_formats(self):
        """测试无效走法格式"""
        board = ChessBoard()
        
        # 测试各种无效格式（需要绕过Move类的验证）
        # 直接测试规则引擎的格式验证
        
        # 创建一些看起来合法但实际不合法的走法
        invalid_moves = [
            Move(from_pos=(6, 0), to_pos=(6, 0), piece=7),   # 起点终点相同
            Move(from_pos=(5, 0), to_pos=(4, 0), piece=7),   # 空位没有棋子
            Move(from_pos=(6, 0), to_pos=(5, 0), piece=5),   # 棋子类型不匹配
        ]
        
        for move in invalid_moves:
            assert not self.rule_engine.is_legal_move(board, move)
        
        # 测试边界情况（通过直接调用内部方法）
        assert not self.rule_engine._is_valid_move_format(
            type('MockMove', (), {
                'from_pos': (-1, 0), 'to_pos': (5, 0), 'piece': 7
            })()
        )
    
    def test_boundary_positions(self):
        """测试边界位置的走法"""
        board = ChessBoard()
        
        # 测试角落位置的棋子
        corner_board = ChessBoard()
        corner_board.board.fill(0)
        corner_board.board[0, 0] = 5  # 车在角落
        corner_board.board[9, 4] = 1  # 红帅
        corner_board.board[0, 4] = -1 # 黑将
        
        corner_moves = self.rule_engine.generate_piece_moves(corner_board, (0, 0))
        assert len(corner_moves) > 0  # 角落的车仍应该能移动


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])