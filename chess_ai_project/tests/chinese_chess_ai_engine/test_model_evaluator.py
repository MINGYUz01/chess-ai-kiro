"""
模型评估器测试

测试模型评估、ELO计算和基准测试功能。
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from chess_ai_project.src.chinese_chess_ai_engine.training_framework import (
    ModelEvaluator,
    EvaluationResult,
    BenchmarkPosition,
    ELOCalculator,
    EvaluationConfig
)
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import ChessBoard, Move


class MockChessNet(nn.Module):
    """模拟的象棋神经网络"""
    
    def __init__(self, name: str = "mock_model"):
        super().__init__()
        self.name = name
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


class TestEvaluationResult:
    """测试评估结果类"""
    
    def test_evaluation_result_creation(self):
        """测试评估结果创建"""
        result = EvaluationResult(
            model_name="model_a",
            opponent_name="model_b",
            wins=7,
            losses=2,
            draws=1,
            total_games=10,
            win_rate=0.7,
            elo_change=25.5,
            average_game_length=45.2,
            average_time_per_move=1.5
        )
        
        assert result.model_name == "model_a"
        assert result.opponent_name == "model_b"
        assert result.wins == 7
        assert result.losses == 2
        assert result.draws == 1
        assert result.total_games == 10
        assert result.win_rate == 0.7
        assert result.elo_change == 25.5
    
    def test_score_calculation(self):
        """测试得分计算"""
        result = EvaluationResult(
            model_name="model_a",
            opponent_name="model_b",
            wins=6,
            losses=2,
            draws=2,
            total_games=10,
            win_rate=0.6,
            elo_change=20.0,
            average_game_length=40.0,
            average_time_per_move=1.0
        )
        
        # 得分 = (胜利 + 0.5 * 平局) / 总局数 = (6 + 0.5 * 2) / 10 = 0.7
        assert result.score == 0.7
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = EvaluationResult(
            model_name="model_a",
            opponent_name="model_b",
            wins=5,
            losses=3,
            draws=2,
            total_games=10,
            win_rate=0.5,
            elo_change=10.0,
            average_game_length=50.0,
            average_time_per_move=2.0,
            metadata={'test': 'data'}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['model_name'] == "model_a"
        assert result_dict['opponent_name'] == "model_b"
        assert result_dict['wins'] == 5
        assert result_dict['score'] == 0.6  # (5 + 0.5 * 2) / 10
        assert result_dict['metadata']['test'] == 'data'


class TestBenchmarkPosition:
    """测试基准测试局面类"""
    
    def test_benchmark_position_creation(self):
        """测试基准局面创建"""
        position = BenchmarkPosition(
            fen="rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
            description="开局标准局面",
            best_moves=["h2e2", "b0c2"],
            difficulty=2,
            category="opening",
            expected_eval=0.1
        )
        
        assert position.fen.startswith("rnbakabnr")
        assert position.description == "开局标准局面"
        assert len(position.best_moves) == 2
        assert position.difficulty == 2
        assert position.category == "opening"
        assert position.expected_eval == 0.1
    
    def test_to_dict(self):
        """测试转换为字典"""
        position = BenchmarkPosition(
            fen="test_fen",
            description="test_description",
            best_moves=["move1", "move2"],
            difficulty=3,
            category="middlegame"
        )
        
        position_dict = position.to_dict()
        
        assert position_dict['fen'] == "test_fen"
        assert position_dict['description'] == "test_description"
        assert position_dict['best_moves'] == ["move1", "move2"]
        assert position_dict['difficulty'] == 3
        assert position_dict['category'] == "middlegame"


class TestELOCalculator:
    """测试ELO计算器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.elo_calc = ELOCalculator(k_factor=32.0, initial_rating=1500.0)
    
    def test_expected_score(self):
        """测试期望得分计算"""
        # 等级分相同时，期望得分应该是0.5
        expected = self.elo_calc.expected_score(1500, 1500)
        assert abs(expected - 0.5) < 1e-6
        
        # 高等级分对低等级分，期望得分应该大于0.5
        expected = self.elo_calc.expected_score(1600, 1400)
        assert expected > 0.5
        
        # 低等级分对高等级分，期望得分应该小于0.5
        expected = self.elo_calc.expected_score(1400, 1600)
        assert expected < 0.5
    
    def test_update_rating(self):
        """测试等级分更新"""
        # 等级分相同，获胜后应该增加
        new_rating = self.elo_calc.update_rating(1500, 1500, 1.0)  # 获胜
        assert new_rating > 1500
        
        # 等级分相同，失败后应该减少
        new_rating = self.elo_calc.update_rating(1500, 1500, 0.0)  # 失败
        assert new_rating < 1500
        
        # 等级分相同，平局后应该不变
        new_rating = self.elo_calc.update_rating(1500, 1500, 0.5)  # 平局
        assert abs(new_rating - 1500) < 1e-6
    
    def test_calculate_rating_change(self):
        """测试等级分变化计算"""
        # 等级分相同，获胜
        change = self.elo_calc.calculate_rating_change(1500, 1500, 1.0)
        assert change == 16.0  # K * (1.0 - 0.5) = 32 * 0.5 = 16
        
        # 等级分相同，失败
        change = self.elo_calc.calculate_rating_change(1500, 1500, 0.0)
        assert change == -16.0  # K * (0.0 - 0.5) = 32 * (-0.5) = -16
        
        # 等级分相同，平局
        change = self.elo_calc.calculate_rating_change(1500, 1500, 0.5)
        assert abs(change) < 1e-6
    
    def test_win_probability(self):
        """测试胜率计算"""
        # 等级分相同
        prob = self.elo_calc.win_probability(1500, 1500)
        assert abs(prob - 0.5) < 1e-6
        
        # 高等级分对低等级分
        prob = self.elo_calc.win_probability(1600, 1400)
        assert prob > 0.5
        
        # 低等级分对高等级分
        prob = self.elo_calc.win_probability(1400, 1600)
        assert prob < 0.5


class TestModelEvaluator:
    """测试模型评估器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = EvaluationConfig(
            num_games=4,  # 减少游戏数量以加快测试
            max_game_length=50,
            time_per_move=0.1
        )
        self.evaluator = ModelEvaluator(self.config)
        self.model1 = MockChessNet("model1")
        self.model2 = MockChessNet("model2")
    
    def test_evaluator_initialization(self):
        """测试评估器初始化"""
        assert self.evaluator.config == self.config
        assert isinstance(self.evaluator.elo_calculator, ELOCalculator)
        assert len(self.evaluator.benchmark_positions) > 0  # 应该有默认的基准局面
        assert isinstance(self.evaluator.model_ratings, dict)
    
    def test_add_benchmark_position(self):
        """测试添加基准局面"""
        initial_count = len(self.evaluator.benchmark_positions)
        
        position = BenchmarkPosition(
            fen="test_fen",
            description="test_position",
            best_moves=["test_move"],
            difficulty=1,
            category="test"
        )
        
        self.evaluator.add_benchmark_position(position)
        
        assert len(self.evaluator.benchmark_positions) == initial_count + 1
        assert self.evaluator.benchmark_positions[-1] == position
    
    def test_load_benchmark_positions(self):
        """测试从文件加载基准局面"""
        positions_data = [
            {
                'fen': 'test_fen_1',
                'description': 'test_position_1',
                'best_moves': ['move1'],
                'difficulty': 1,
                'category': 'test'
            },
            {
                'fen': 'test_fen_2',
                'description': 'test_position_2',
                'best_moves': ['move2'],
                'difficulty': 2,
                'category': 'test'
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(positions_data, f)
            temp_path = f.name
        
        try:
            initial_count = len(self.evaluator.benchmark_positions)
            self.evaluator.load_benchmark_positions(temp_path)
            
            assert len(self.evaluator.benchmark_positions) == initial_count + 2
            
            # 检查加载的局面
            loaded_positions = self.evaluator.benchmark_positions[-2:]
            assert loaded_positions[0].fen == 'test_fen_1'
            assert loaded_positions[1].fen == 'test_fen_2'
        finally:
            Path(temp_path).unlink()
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.training_framework.model_evaluator.MCTSSearcher')
    def test_evaluate_against_baseline(self, mock_searcher_class):
        """测试与基准模型对弈评估"""
        # 模拟MCTS搜索器
        mock_searcher = Mock()
        mock_searcher_class.return_value = mock_searcher
        
        # 模拟搜索结果
        mock_root_node = Mock()
        mock_searcher.search.return_value = mock_root_node
        mock_searcher.get_action_probabilities.return_value = np.array([0.8, 0.2])
        mock_searcher.get_value_estimate.return_value = 0.3
        
        # 模拟棋盘和走法
        with patch('chess_ai_project.src.chinese_chess_ai_engine.rules_engine.ChessBoard') as mock_board_class:
            mock_board = Mock()
            mock_board_class.return_value = mock_board
            
            # 设置游戏快速结束
            mock_board.is_game_over.side_effect = [False, True]
            mock_board.get_legal_moves.return_value = [
                Mock(to_coordinate_notation=lambda: "a0a1"),
                Mock(to_coordinate_notation=lambda: "b0b1")
            ]
            mock_board.get_winner.return_value = 1  # model1获胜
            mock_board.make_move.return_value = mock_board
            
            # 执行评估
            result = self.evaluator.evaluate_against_baseline(
                self.model1, self.model2, "model1", "model2"
            )
            
            # 验证结果
            assert isinstance(result, EvaluationResult)
            assert result.model_name == "model1"
            assert result.opponent_name == "model2"
            assert result.total_games == self.config.num_games
            assert 0 <= result.win_rate <= 1
            assert result.wins + result.losses + result.draws == result.total_games
    
    def test_tournament_evaluation(self):
        """测试锦标赛评估"""
        models = {
            "model_a": self.model1,
            "model_b": self.model2,
            "model_c": MockChessNet("model_c")
        }
        
        # 模拟evaluate_against_baseline方法
        def mock_evaluate(model1, model2, name1, name2):
            return EvaluationResult(
                model_name=name1,
                opponent_name=name2,
                wins=2,
                losses=1,
                draws=1,
                total_games=4,
                win_rate=0.5,
                elo_change=10.0,
                average_game_length=30.0,
                average_time_per_move=1.0
            )
        
        with patch.object(self.evaluator, 'evaluate_against_baseline', side_effect=mock_evaluate):
            result = self.evaluator.tournament_evaluation(models, num_games_per_pair=4)
            
            # 验证结果结构
            assert 'models' in result
            assert 'pairwise_results' in result
            assert 'rankings' in result
            assert 'final_ratings' in result
            
            # 验证模型列表
            assert len(result['models']) == 3
            assert set(result['models']) == {"model_a", "model_b", "model_c"}
            
            # 验证对弈结果数量 (3个模型，3对对弈)
            assert len(result['pairwise_results']) == 3
            
            # 验证排名
            assert len(result['rankings']) == 3
            for ranking in result['rankings']:
                assert 'model_name' in ranking
                assert 'rating' in ranking
                assert 'rank' in ranking
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.training_framework.model_evaluator.MCTSSearcher')
    def test_benchmark_performance(self, mock_searcher_class):
        """测试基准性能测试"""
        # 模拟MCTS搜索器
        mock_searcher = Mock()
        mock_searcher_class.return_value = mock_searcher
        
        mock_root_node = Mock()
        mock_searcher.search.return_value = mock_root_node
        mock_searcher.get_action_probabilities.return_value = np.array([0.8, 0.2])
        mock_searcher.get_value_estimate.return_value = 0.3
        
        # 创建测试局面
        test_positions = [
            BenchmarkPosition(
                fen="rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
                description="test_1",
                best_moves=["a0a1"],
                difficulty=1,
                category="test"
            ),
            BenchmarkPosition(
                fen="rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
                description="test_2",
                best_moves=["b0b1"],
                difficulty=2,
                category="test"
            )
        ]
        
        # 模拟棋盘
        with patch('chess_ai_project.src.chinese_chess_ai_engine.rules_engine.ChessBoard') as mock_board_class:
            mock_board = Mock()
            mock_board_class.return_value = mock_board
            
            # 模拟合法走法
            mock_moves = [
                Mock(to_coordinate_notation=lambda: "a0a1"),
                Mock(to_coordinate_notation=lambda: "b0b1")
            ]
            mock_board.get_legal_moves.return_value = mock_moves
            
            # 执行基准测试
            result = self.evaluator.benchmark_performance(
                self.model1, "test_model", test_positions
            )
            
            # 验证结果
            assert result['model_name'] == "test_model"
            assert result['total_positions'] == 2
            assert 'overall_accuracy' in result
            assert 'average_search_time' in result
            assert 'category_stats' in result
            assert 'difficulty_stats' in result
            assert 'detailed_results' in result
            
            # 验证详细结果
            assert len(result['detailed_results']) == 2
            for detail in result['detailed_results']:
                assert 'fen' in detail
                assert 'predicted_move' in detail
                assert 'is_correct' in detail
                assert 'search_time' in detail
    
    def test_model_rating_management(self):
        """测试模型等级分管理"""
        # 测试获取不存在的模型等级分
        rating = self.evaluator.get_model_rating("new_model")
        assert rating == self.config.initial_elo
        
        # 测试设置模型等级分
        self.evaluator.set_model_rating("test_model", 1600.0)
        rating = self.evaluator.get_model_rating("test_model")
        assert rating == 1600.0
        
        # 测试获取所有等级分
        all_ratings = self.evaluator.get_all_ratings()
        assert "test_model" in all_ratings
        assert all_ratings["test_model"] == 1600.0
    
    def test_save_evaluation_report(self):
        """测试保存评估报告"""
        test_results = {
            'model_name': 'test_model',
            'accuracy': 0.85,
            'details': ['result1', 'result2']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.evaluator.save_evaluation_report(test_results, temp_path)
            
            # 验证文件是否创建
            assert Path(temp_path).exists()
            
            # 验证文件内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded_results = json.load(f)
            
            assert loaded_results['model_name'] == 'test_model'
            assert loaded_results['accuracy'] == 0.85
            assert loaded_results['details'] == ['result1', 'result2']
        finally:
            Path(temp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__])