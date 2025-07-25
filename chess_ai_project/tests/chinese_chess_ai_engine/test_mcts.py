"""
MCTS安全测试

只测试不会卡住的基本功能。
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from chess_ai_project.src.chinese_chess_ai_engine.search_algorithm import (
    MCTSNode, MCTSConfig
)
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import ChessBoard, Move


class TestMCTSNodeSafe:
    """MCTS节点安全测试"""
    
    def test_node_creation(self):
        """测试节点创建"""
        board = ChessBoard()
        node = MCTSNode(board=board)
        
        assert node.board == board
        assert node.move is None
        assert node.parent is None
        assert len(node.children) == 0
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior_probability == 0.0
    
    def test_node_with_move_and_parent(self):
        """测试带走法和父节点的节点"""
        board = ChessBoard()
        parent = MCTSNode(board=board)
        move = Move((0, 0), (0, 1), 1)
        
        child = MCTSNode(
            board=board,
            move=move,
            parent=parent,
            prior_probability=0.3
        )
        
        assert child.move == move
        assert child.parent == parent
        assert child.prior_probability == 0.3
    
    def test_is_expanded_property(self):
        """测试节点扩展状态属性"""
        board = ChessBoard()
        node = MCTSNode(board=board)
        
        # 初始状态未扩展
        assert not node.is_expanded
        
        # 手动添加子节点
        move = Move((0, 0), (0, 1), 1)
        child = MCTSNode(board=board, move=move)
        node.children[move] = child
        
        # 现在应该是已扩展状态
        assert node.is_expanded
    
    def test_is_terminal_property(self):
        """测试终端节点属性"""
        # 使用模拟对象避免实际的游戏逻辑
        mock_board = Mock()
        mock_board.is_game_over.return_value = True
        
        node = MCTSNode(board=mock_board)
        assert node.is_terminal
        
        # 非终端状态
        mock_board.is_game_over.return_value = False
        node2 = MCTSNode(board=mock_board)
        assert not node2.is_terminal
    
    def test_average_value_calculation(self):
        """测试平均价值计算"""
        board = ChessBoard()
        node = MCTSNode(board=board)
        
        # 初始状态
        assert node.average_value == 0.0
        
        # 设置访问次数和价值总和
        node.visit_count = 4
        node.value_sum = 2.0
        assert node.average_value == 0.5
        
        # 负值测试
        node.value_sum = -1.0
        assert node.average_value == -0.25
    
    def test_ucb_score_basic(self):
        """测试基本UCB分数计算"""
        board = ChessBoard()
        parent = MCTSNode(board=board)
        parent.visit_count = 10
        
        child = MCTSNode(
            board=board,
            move=Move((0, 0), (0, 1), 1),
            parent=parent,
            prior_probability=0.4
        )
        
        # 未访问的节点应该返回无穷大
        assert child.ucb_score() == float('inf')
        
        # 设置访问次数和价值
        child.visit_count = 3
        child.value_sum = 1.5
        
        # 计算UCB分数
        score = child.ucb_score(c_puct=1.0, parent_visits=10)
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)
        assert score > 0  # 应该是正数
    
    def test_backup_single_node(self):
        """测试单个节点的价值回传"""
        board = ChessBoard()
        node = MCTSNode(board=board)
        
        # 回传正值
        node.backup(1.0)
        assert node.visit_count == 1
        assert node.value_sum == 1.0
        assert node.average_value == 1.0
        
        # 再次回传
        node.backup(0.5)
        assert node.visit_count == 2
        assert node.value_sum == 1.5
        assert node.average_value == 0.75
    
    def test_backup_parent_child(self):
        """测试父子节点的价值回传"""
        board = ChessBoard()
        parent = MCTSNode(board=board)
        child = MCTSNode(
            board=board,
            move=Move((0, 0), (0, 1), 1),
            parent=parent
        )
        
        # 从子节点回传价值
        child.backup(1.0)
        
        # 验证子节点
        assert child.visit_count == 1
        assert child.value_sum == 1.0
        assert child.average_value == 1.0
        
        # 验证父节点（价值应该相反）
        assert parent.visit_count == 1
        assert parent.value_sum == -1.0
        assert parent.average_value == -1.0
    
    def test_get_legal_moves_caching(self):
        """测试合法走法缓存"""
        # 使用模拟对象
        mock_board = Mock()
        mock_moves = [Move((0, 0), (0, 1), 1), Move((0, 0), (1, 0), 1)]
        mock_board.get_legal_moves.return_value = mock_moves
        
        node = MCTSNode(board=mock_board)
        
        # 第一次调用
        moves1 = node.get_legal_moves()
        assert moves1 == mock_moves
        assert mock_board.get_legal_moves.call_count == 1
        
        # 第二次调用应该使用缓存
        moves2 = node.get_legal_moves()
        assert moves2 == mock_moves
        assert mock_board.get_legal_moves.call_count == 1  # 没有增加


class TestMCTSConfigSafe:
    """MCTS配置安全测试"""
    
    def test_default_configuration(self):
        """测试默认配置"""
        config = MCTSConfig()
        
        assert config.num_simulations == 800
        assert config.c_puct == 1.0
        assert config.temperature == 1.0
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        assert config.max_depth == 100
        assert config.time_limit is None
    
    def test_custom_configuration(self):
        """测试自定义配置"""
        config = MCTSConfig(
            num_simulations=500,
            c_puct=1.5,
            temperature=0.8,
            dirichlet_alpha=0.2,
            dirichlet_epsilon=0.3,
            max_depth=50,
            time_limit=5.0
        )
        
        assert config.num_simulations == 500
        assert config.c_puct == 1.5
        assert config.temperature == 0.8
        assert config.dirichlet_alpha == 0.2
        assert config.dirichlet_epsilon == 0.3
        assert config.max_depth == 50
        assert config.time_limit == 5.0
    
    def test_to_dict_conversion(self):
        """测试转换为字典"""
        config = MCTSConfig(
            num_simulations=300,
            c_puct=2.0,
            temperature=0.5
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['num_simulations'] == 300
        assert config_dict['c_puct'] == 2.0
        assert config_dict['temperature'] == 0.5
        assert 'dirichlet_alpha' in config_dict
        assert 'max_depth' in config_dict
    
    def test_from_dict_creation(self):
        """测试从字典创建配置"""
        config_dict = {
            'num_simulations': 400,
            'c_puct': 1.8,
            'temperature': 0.6,
            'max_depth': 80
        }
        
        config = MCTSConfig.from_dict(config_dict)
        
        assert config.num_simulations == 400
        assert config.c_puct == 1.8
        assert config.temperature == 0.6
        assert config.max_depth == 80
        # 未指定的参数应该使用默认值
        assert config.dirichlet_alpha == 0.3
    
    def test_round_trip_conversion(self):
        """测试往返转换"""
        original_config = MCTSConfig(
            num_simulations=600,
            c_puct=1.2,
            temperature=0.9
        )
        
        # 转换为字典再转换回来
        config_dict = original_config.to_dict()
        restored_config = MCTSConfig.from_dict(config_dict)
        
        # 验证所有参数都相同
        assert restored_config.num_simulations == original_config.num_simulations
        assert restored_config.c_puct == original_config.c_puct
        assert restored_config.temperature == original_config.temperature
        assert restored_config.dirichlet_alpha == original_config.dirichlet_alpha
        assert restored_config.dirichlet_epsilon == original_config.dirichlet_epsilon
        assert restored_config.max_depth == original_config.max_depth
        assert restored_config.time_limit == original_config.time_limit


class TestMCTSNodeExpansionSafe:
    """MCTS节点扩展安全测试"""
    
    def test_expand_with_mock_board(self):
        """使用模拟棋盘测试节点扩展"""
        # 创建模拟棋盘
        mock_board = Mock()
        mock_moves = [
            Move((0, 0), (0, 1), 1),
            Move((0, 0), (1, 0), 1),
            Move((1, 0), (1, 1), 1)
        ]
        mock_board.get_legal_moves.return_value = mock_moves
        
        # 模拟make_move方法
        def mock_make_move(move):
            new_board = Mock()
            new_board.get_legal_moves.return_value = []
            new_board.is_game_over.return_value = False
            return new_board
        
        mock_board.make_move = mock_make_move
        mock_board.is_game_over.return_value = False
        
        # 创建节点
        node = MCTSNode(board=mock_board)
        
        # 准备先验概率
        move_priors = {move: 1.0 / len(mock_moves) for move in mock_moves}
        
        # 扩展节点
        children = node.expand(move_priors)
        
        # 验证扩展结果
        assert len(children) == len(mock_moves)
        assert len(node.children) == len(mock_moves)
        assert node.is_expanded
        
        # 验证每个子节点
        for move in mock_moves:
            assert move in node.children
            child = node.children[move]
            assert child.move == move
            assert child.parent == node
            assert child.prior_probability == move_priors[move]
    
    def test_expand_empty_moves(self):
        """测试没有合法走法时的扩展"""
        mock_board = Mock()
        mock_board.get_legal_moves.return_value = []
        mock_board.is_game_over.return_value = True
        
        node = MCTSNode(board=mock_board)
        
        # 扩展应该不会创建子节点
        children = node.expand({})
        
        assert len(children) == 0
        assert len(node.children) == 0
        assert not node.is_expanded  # 没有子节点，仍然未扩展
    
    def test_double_expansion(self):
        """测试重复扩展"""
        mock_board = Mock()
        mock_moves = [Move((0, 0), (0, 1), 1)]
        mock_board.get_legal_moves.return_value = mock_moves
        
        def mock_make_move(move):
            new_board = Mock()
            new_board.get_legal_moves.return_value = []
            return new_board
        
        mock_board.make_move = mock_make_move
        
        node = MCTSNode(board=mock_board)
        move_priors = {mock_moves[0]: 1.0}
        
        # 第一次扩展
        children1 = node.expand(move_priors)
        assert len(children1) == 1
        
        # 第二次扩展应该返回现有子节点
        children2 = node.expand(move_priors)
        assert len(children2) == 1
        assert children1[0] == children2[0]  # 应该是同一个对象


if __name__ == "__main__":
    pytest.main([__file__, "-v"])