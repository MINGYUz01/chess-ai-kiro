"""
ChessAI测试

测试象棋AI分析和决策核心功能。
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from chess_ai_project.src.chinese_chess_ai_engine.inference_interface import (
    ChessAI,
    AnalysisResult,
    AIConfig,
    DifficultyManager
)
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import ChessBoard, Move


class MockChessNet(nn.Module):
    """模拟的象棋神经网络"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(20, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.value_head = nn.Linear(128 * 10 * 9, 1)
        self.policy_head = nn.Linear(128 * 10 * 9, 8100)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        
        value = torch.tanh(self.value_head(x))
        policy = self.policy_head(x)
        
        return value, policy


class TestAnalysisResult:
    """测试分析结果类"""
    
    def test_analysis_result_creation(self):
        """测试分析结果创建"""
        move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        
        result = AnalysisResult(
            best_move=move,
            evaluation=0.5,
            win_probability=(0.7, 0.3),
            principal_variation=[move],
            top_moves=[(move, 0.8)],
            search_depth=5,
            nodes_searched=1000,
            time_used=2.5,
            metadata={'test': 'data'}
        )
        
        assert result.best_move == move
        assert result.evaluation == 0.5
        assert result.win_probability == (0.7, 0.3)
        assert len(result.principal_variation) == 1
        assert len(result.top_moves) == 1
        assert result.search_depth == 5
        assert result.nodes_searched == 1000
        assert result.time_used == 2.5
        assert result.metadata['test'] == 'data'
    
    def test_analysis_result_post_init(self):
        """测试分析结果初始化后处理"""
        move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        
        result = AnalysisResult(
            best_move=move,
            evaluation=0.0,
            win_probability=(0.5, 0.5),
            principal_variation=[],
            top_moves=[],
            search_depth=0,
            nodes_searched=0,
            time_used=0.0
        )
        
        # metadata应该被初始化为空字典
        assert result.metadata == {}


class TestAIConfig:
    """测试AI配置类"""
    
    def test_ai_config_creation(self):
        """测试AI配置创建"""
        config = AIConfig(
            model_path="test_model.pth",
            search_time=3.0,
            max_simulations=500,
            difficulty_level=7
        )
        
        assert config.model_path == "test_model.pth"
        assert config.search_time == 3.0
        assert config.max_simulations == 500
        assert config.difficulty_level == 7
        assert config.device in ['cpu', 'cuda', 'mps']
    
    def test_ai_config_defaults(self):
        """测试AI配置默认值"""
        config = AIConfig(model_path="test.pth")
        
        assert config.search_time == 5.0
        assert config.max_simulations == 5000  # 修正默认值
        assert config.difficulty_level == 5
        assert config.use_opening_book == True
        assert config.use_endgame_tablebase == True
        assert config.c_puct == 1.0
        assert config.temperature == 0.1
    
    def test_device_auto_selection(self):
        """测试设备自动选择"""
        config = AIConfig(model_path="test.pth", device='auto')
        
        # 设备应该被自动设置
        assert config.device in ['cpu', 'cuda', 'mps']


class TestDifficultyManager:
    """测试难度管理器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = AIConfig(model_path="test.pth")
        self.difficulty_manager = DifficultyManager(self.config)
    
    def test_get_search_params_level_1(self):
        """测试难度级别1的搜索参数"""
        params = self.difficulty_manager.get_search_params(1)
        
        assert params['search_time'] == self.config.min_search_time
        assert params['simulations'] == self.config.min_simulations
        assert params['temperature'] > 0.1  # 应该比较高
        assert params['randomness'] > 0  # 应该有随机性
    
    def test_get_search_params_level_10(self):
        """测试难度级别10的搜索参数"""
        params = self.difficulty_manager.get_search_params(10)
        
        assert params['search_time'] == self.config.max_search_time
        assert params['simulations'] == self.config.max_simulations
        assert abs(params['temperature'] - 0.1) < 0.001  # 使用近似相等
        assert params['randomness'] == 0  # 应该没有随机性
    
    def test_get_search_params_mid_level(self):
        """测试中等难度级别的搜索参数"""
        params = self.difficulty_manager.get_search_params(5)
        
        # 参数应该在最小值和最大值之间
        assert self.config.min_search_time < params['search_time'] < self.config.max_search_time
        assert self.config.min_simulations < params['simulations'] < self.config.max_simulations
        assert 0.1 <= params['temperature'] <= 0.5
    
    def test_difficulty_level_bounds(self):
        """测试难度级别边界"""
        # 测试超出范围的难度级别
        params_low = self.difficulty_manager.get_search_params(0)  # 应该被限制为1
        params_high = self.difficulty_manager.get_search_params(15)  # 应该被限制为10
        
        params_1 = self.difficulty_manager.get_search_params(1)
        params_10 = self.difficulty_manager.get_search_params(10)
        
        assert params_low == params_1
        assert params_high == params_10
    
    def test_should_add_randomness(self):
        """测试随机性判断"""
        # 低难度应该有更高的随机性概率
        with patch('random.random', return_value=0.05):
            assert self.difficulty_manager.should_add_randomness(1) == True
            assert self.difficulty_manager.should_add_randomness(10) == False


class TestChessAI:
    """测试ChessAI类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = AIConfig(
            model_path="test_model.pth",
            search_time=1.0,  # 减少测试时间
            max_simulations=100,
            difficulty_level=5
        )
        
        # 创建临时模型文件
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        self.temp_model_file.close()
        
        # 保存一个简单的模型
        model = MockChessNet()
        torch.save(model.state_dict(), self.temp_model_file.name)
        
        self.config.model_path = self.temp_model_file.name
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.ModelManager')
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.MCTSSearcher')
    def test_chess_ai_initialization(self, mock_searcher_class, mock_model_manager_class):
        """测试ChessAI初始化"""
        # 模拟ModelManager
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        mock_model_manager.load_model.return_value = MockChessNet()
        
        ai = ChessAI(self.config.model_path, self.config)
        
        assert ai.config == self.config
        assert ai.device.type == self.config.device
        assert ai.model is not None
        assert ai.board_encoder is not None
        assert ai.difficulty_manager is not None
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.ModelManager')
    def test_model_loading_fallback(self, mock_model_manager_class):
        """测试模型加载失败时的备选方案"""
        # 模拟ModelManager加载失败
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        mock_model_manager.load_model.side_effect = Exception("加载失败")
        
        # 应该创建默认模型而不是崩溃
        ai = ChessAI("nonexistent_model.pth", self.config)
        assert ai.model is not None
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.ModelManager')
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.MCTSSearcher')
    @patch('time.time')
    def test_analyze_position(self, mock_time, mock_searcher_class, mock_model_manager_class):
        """测试位置分析"""
        # 模拟时间流逝
        mock_time.side_effect = [0.0, 1.5]  # 开始时间和结束时间
        
        # 模拟ModelManager
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        mock_model_manager.load_model.return_value = MockChessNet()
        
        # 模拟MCTS搜索器
        mock_searcher = Mock()
        mock_searcher_class.return_value = mock_searcher
        
        # 模拟搜索结果
        mock_root_node = Mock()
        mock_root_node.visit_count = 100
        mock_root_node.children = {}  # 添加children属性以避免迭代错误
        mock_searcher.search.return_value = mock_root_node
        
        test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        mock_searcher.get_best_move.return_value = test_move
        mock_searcher.get_value_estimate.return_value = 0.3
        mock_searcher.get_action_probabilities.return_value = np.array([0.8, 0.2])
        
        # 创建AI
        ai = ChessAI(self.config.model_path, self.config)
        
        # 模拟棋盘
        board = ChessBoard()
        
        # 模拟合法走法
        with patch.object(board, 'get_legal_moves') as mock_get_legal_moves:
            mock_get_legal_moves.return_value = [test_move, Move(from_pos=(0, 3), to_pos=(1, 3), piece=2)]
            
            # 执行分析
            result = ai.analyze_position(board)
            
            # 验证结果
            assert isinstance(result, AnalysisResult)
            assert result.best_move == test_move
            assert result.evaluation == 0.3
            assert len(result.win_probability) == 2
            assert result.nodes_searched == 100
            assert result.time_used == 1.5  # 应该等于模拟的时间差
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.ModelManager')
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.MCTSSearcher')
    def test_get_best_move(self, mock_searcher_class, mock_model_manager_class):
        """测试获取最佳走法"""
        # 模拟ModelManager
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        mock_model_manager.load_model.return_value = MockChessNet()
        
        # 模拟MCTS搜索器
        mock_searcher = Mock()
        mock_searcher_class.return_value = mock_searcher
        
        mock_root_node = Mock()
        mock_searcher.search.return_value = mock_root_node
        
        test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        mock_searcher.get_best_move.return_value = test_move
        mock_searcher.get_value_estimate.return_value = 0.3
        mock_searcher.get_action_probabilities.return_value = np.array([0.8, 0.2])
        
        ai = ChessAI(self.config.model_path, self.config)
        board = ChessBoard()
        
        with patch.object(board, 'get_legal_moves') as mock_get_legal_moves:
            mock_get_legal_moves.return_value = [test_move]
            
            best_move = ai.get_best_move(board, time_limit=0.5)
            assert best_move == test_move
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.ModelManager')
    def test_evaluate_position(self, mock_model_manager_class):
        """测试位置评估"""
        # 模拟ModelManager
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        
        # 创建模拟模型
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.to.return_value = mock_model
        
        # 模拟模型输出
        mock_value = torch.tensor([[0.5]])
        mock_policy = torch.randn(1, 8100)
        mock_model.return_value = (mock_value, mock_policy)
        
        mock_model_manager.load_model.return_value = mock_model
        
        ai = ChessAI(self.config.model_path, self.config)
        board = ChessBoard()
        
        evaluation = ai.evaluate_position(board)
        
        # 评估值应该在合理范围内
        assert -1.0 <= evaluation <= 1.0
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.ModelManager')
    def test_difficulty_level_management(self, mock_model_manager_class):
        """测试难度级别管理"""
        # 模拟ModelManager
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        mock_model_manager.load_model.return_value = MockChessNet()
        
        ai = ChessAI(self.config.model_path, self.config)
        
        # 测试设置难度级别
        ai.set_difficulty_level(8)
        assert ai.get_difficulty_level() == 8
        
        # 测试边界值
        ai.set_difficulty_level(0)  # 应该被限制为1
        assert ai.get_difficulty_level() == 1
        
        ai.set_difficulty_level(15)  # 应该被限制为10
        assert ai.get_difficulty_level() == 10
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.ModelManager')
    def test_statistics(self, mock_model_manager_class):
        """测试统计信息"""
        # 模拟ModelManager
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        mock_model_manager.load_model.return_value = MockChessNet()
        
        ai = ChessAI(self.config.model_path, self.config)
        
        # 初始统计信息
        stats = ai.get_statistics()
        assert stats['positions_analyzed'] == 0
        assert stats['total_search_time'] == 0.0
        assert stats['difficulty_level'] == self.config.difficulty_level
        
        # 更新统计信息
        ai._update_stats(search_time=2.0, nodes_searched=1000, search_depth=5)
        
        stats = ai.get_statistics()
        assert stats['positions_analyzed'] == 1
        assert stats['total_search_time'] == 2.0
        assert stats['total_nodes_searched'] == 1000
        assert stats['average_depth'] == 5.0
        assert stats['average_search_time'] == 2.0
        
        # 重置统计信息
        ai.reset_statistics()
        stats = ai.get_statistics()
        assert stats['positions_analyzed'] == 0
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.ModelManager')
    def test_win_probability_calculation(self, mock_model_manager_class):
        """测试胜率计算"""
        # 模拟ModelManager
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        mock_model_manager.load_model.return_value = MockChessNet()
        
        ai = ChessAI(self.config.model_path, self.config)
        
        # 测试不同评估值的胜率计算
        red_prob, black_prob = ai._calculate_win_probability(0.0, 1)  # 红方回合，评估为0
        assert abs(red_prob - 0.5) < 0.1  # 应该接近50%
        assert abs(black_prob - 0.5) < 0.1
        
        red_prob, black_prob = ai._calculate_win_probability(1.0, 1)  # 红方回合，评估为1
        assert red_prob > 0.8  # 红方胜率应该很高
        assert black_prob < 0.2
        
        red_prob, black_prob = ai._calculate_win_probability(-1.0, 1)  # 红方回合，评估为-1
        assert red_prob < 0.2  # 红方胜率应该很低
        assert black_prob > 0.8
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.chess_ai.ModelManager')
    def test_error_handling(self, mock_model_manager_class):
        """测试错误处理"""
        # 模拟ModelManager
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        mock_model_manager.load_model.return_value = MockChessNet()
        
        ai = ChessAI(self.config.model_path, self.config)
        board = ChessBoard()
        
        # 测试获取最佳走法时的错误处理
        with patch.object(ai, 'analyze_position', side_effect=Exception("分析失败")):
            with patch.object(board, 'get_legal_moves') as mock_get_legal_moves:
                test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
                mock_get_legal_moves.return_value = [test_move]
                
                # 应该返回第一个合法走法作为备选
                best_move = ai.get_best_move(board)
                assert best_move == test_move
        
        # 测试位置评估时的错误处理
        with patch.object(ai.model, '__call__', side_effect=Exception("推理失败")):
            evaluation = ai.evaluate_position(board)
            # 由于模型推理失败，应该返回一个合理的默认值或随机值
            assert -1.0 <= evaluation <= 1.0  # 应该在合理范围内


if __name__ == '__main__':
    pytest.main([__file__])