"""
训练框架模块测试

测试自对弈数据生成、棋盘编码器和训练样本等功能。
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from chess_ai_project.src.chinese_chess_ai_engine.training_framework import (
    TrainingExample,
    TrainingDataset,
    BoardEncoder,
    SelfPlayGenerator,
    SelfPlayConfig,
    GameResult
)
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import ChessBoard, Move
from chess_ai_project.src.chinese_chess_ai_engine.neural_network import ChessNet


class TestTrainingExample:
    """测试训练样本类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.board_tensor = torch.randn(20, 10, 9)
        self.policy_target = np.random.random(8100).astype(np.float32)
        self.policy_target /= self.policy_target.sum()  # 归一化
        
        self.example = TrainingExample(
            board_tensor=self.board_tensor,
            policy_target=self.policy_target,
            value_target=0.5,
            game_result=1,
            move_number=10,
            current_player=1,
            metadata={'test': 'data'}
        )
    
    def test_training_example_creation(self):
        """测试训练样本创建"""
        assert self.example.board_tensor.shape == (20, 10, 9)
        assert self.example.policy_target.shape == (8100,)
        assert self.example.value_target == 0.5
        assert self.example.game_result == 1
        assert self.example.move_number == 10
        assert self.example.current_player == 1
        assert self.example.metadata['test'] == 'data'
    
    def test_dimension_validation(self):
        """测试维度验证"""
        # 错误的棋盘张量维度
        with pytest.raises(ValueError, match="棋盘张量维度错误"):
            TrainingExample(
                board_tensor=torch.randn(10, 10, 9),  # 错误维度
                policy_target=self.policy_target,
                value_target=0.5,
                game_result=1,
                move_number=1,
                current_player=1
            )
        
        # 错误的策略目标维度
        with pytest.raises(ValueError, match="策略目标维度错误"):
            TrainingExample(
                board_tensor=self.board_tensor,
                policy_target=np.random.random(1000),  # 错误维度
                value_target=0.5,
                game_result=1,
                move_number=1,
                current_player=1
            )
        
        # 价值目标超出范围
        with pytest.raises(ValueError, match="价值目标超出范围"):
            TrainingExample(
                board_tensor=self.board_tensor,
                policy_target=self.policy_target,
                value_target=2.0,  # 超出范围
                game_result=1,
                move_number=1,
                current_player=1
            )
    
    def test_to_dict_and_from_dict(self):
        """测试字典转换"""
        # 添加实际走法
        move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        self.example.actual_move = move
        
        # 转换为字典
        data_dict = self.example.to_dict()
        
        # 验证字典内容
        assert 'board_tensor' in data_dict
        assert 'policy_target' in data_dict
        assert 'value_target' in data_dict
        assert 'actual_move' in data_dict
        assert data_dict['actual_move']['from_pos'] == (0, 4)
        
        # 从字典重建
        reconstructed = TrainingExample.from_dict(data_dict)
        
        # 验证重建的对象
        assert torch.equal(reconstructed.board_tensor, self.example.board_tensor)
        assert np.array_equal(reconstructed.policy_target, self.example.policy_target)
        assert reconstructed.value_target == self.example.value_target
        assert reconstructed.actual_move.from_pos == move.from_pos
    
    def test_file_operations(self):
        """测试文件保存和加载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试pickle格式
            pickle_path = os.path.join(temp_dir, 'example.pkl')
            self.example.save_to_file(pickle_path, 'pickle')
            loaded_example = TrainingExample.load_from_file(pickle_path, 'pickle')
            
            assert torch.equal(loaded_example.board_tensor, self.example.board_tensor)
            assert np.array_equal(loaded_example.policy_target, self.example.policy_target)
            
            # 测试JSON格式
            json_path = os.path.join(temp_dir, 'example.json')
            self.example.save_to_file(json_path, 'json')
            loaded_example = TrainingExample.load_from_file(json_path, 'json')
            
            assert torch.equal(loaded_example.board_tensor, self.example.board_tensor)
            assert np.array_equal(loaded_example.policy_target, self.example.policy_target)
    
    def test_policy_entropy(self):
        """测试策略熵计算"""
        # 创建均匀分布的策略
        uniform_policy = np.ones(8100) / 8100
        example = TrainingExample(
            board_tensor=self.board_tensor,
            policy_target=uniform_policy,
            value_target=0.0,
            game_result=0,
            move_number=1,
            current_player=1
        )
        
        entropy = example.get_policy_entropy()
        expected_entropy = np.log(8100)  # 均匀分布的熵
        assert abs(entropy - expected_entropy) < 0.1
    
    def test_top_policy_moves(self):
        """测试获取最高概率走法"""
        # 创建有明显最高概率的策略
        policy = np.zeros(8100)
        policy[100] = 0.5
        policy[200] = 0.3
        policy[300] = 0.2
        
        example = TrainingExample(
            board_tensor=self.board_tensor,
            policy_target=policy,
            value_target=0.0,
            game_result=0,
            move_number=1,
            current_player=1
        )
        
        top_moves = example.get_top_policy_moves(3)
        assert len(top_moves) == 3
        assert top_moves[0] == (100, 0.5)
        assert top_moves[1] == (200, 0.3)
        assert top_moves[2] == (300, 0.2)


class TestTrainingDataset:
    """测试训练数据集类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.examples = []
        for i in range(10):
            example = TrainingExample(
                board_tensor=torch.randn(20, 10, 9),
                policy_target=np.random.random(8100).astype(np.float32),
                value_target=np.random.uniform(-1, 1),
                game_result=np.random.choice([-1, 0, 1]),
                move_number=i + 1,
                current_player=1 if i % 2 == 0 else -1,
                metadata={'game_id': f'game_{i // 5}'}
            )
            self.examples.append(example)
        
        self.dataset = TrainingDataset(self.examples)
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        assert len(self.dataset) == 10
        assert self.dataset.metadata['total_moves'] == 10
    
    def test_add_examples(self):
        """测试添加样本"""
        new_example = TrainingExample(
            board_tensor=torch.randn(20, 10, 9),
            policy_target=np.random.random(8100).astype(np.float32),
            value_target=0.0,
            game_result=0,
            move_number=1,
            current_player=1
        )
        
        initial_size = len(self.dataset)
        self.dataset.add_example(new_example)
        assert len(self.dataset) == initial_size + 1
    
    def test_shuffle(self):
        """测试数据集打乱"""
        original_order = [ex.move_number for ex in self.dataset.examples]
        self.dataset.shuffle()
        shuffled_order = [ex.move_number for ex in self.dataset.examples]
        
        # 打乱后顺序应该不同（除非极小概率相同）
        assert original_order != shuffled_order or len(original_order) <= 2
    
    def test_split(self):
        """测试数据集分割"""
        train_dataset, val_dataset = self.dataset.split(train_ratio=0.8)
        
        assert len(train_dataset) == 8
        assert len(val_dataset) == 2
        assert len(train_dataset) + len(val_dataset) == len(self.dataset)
    
    def test_get_batch(self):
        """测试获取批次"""
        batch = self.dataset.get_batch(batch_size=3, start_idx=2)
        assert len(batch) == 3
        assert batch[0].move_number == self.examples[2].move_number
    
    def test_statistics(self):
        """测试统计信息"""
        stats = self.dataset.get_statistics()
        
        assert 'total_examples' in stats
        assert 'value_stats' in stats
        assert 'move_stats' in stats
        assert 'policy_entropy' in stats
        
        assert stats['total_examples'] == 10
        assert 'mean' in stats['value_stats']
        assert 'std' in stats['value_stats']


class TestBoardEncoder:
    """测试棋盘编码器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.encoder = BoardEncoder(history_length=8)
        self.board = ChessBoard()
    
    def test_encoder_initialization(self):
        """测试编码器初始化"""
        assert self.encoder.history_length == 8
        assert self.encoder.total_channels == 20
        assert len(self.encoder.piece_to_channel) == 14
    
    def test_encode_board(self):
        """测试棋盘编码"""
        features = self.encoder.encode_board(self.board)
        
        assert features.shape == (20, 10, 9)
        assert features.dtype == torch.float32
        
        # 检查棋子编码
        board_matrix = self.board.to_matrix()
        for row in range(10):
            for col in range(9):
                piece = board_matrix[row, col]
                if piece != 0:
                    channel = self.encoder.piece_to_channel[piece]
                    assert features[channel, row, col] == 1.0
    
    def test_encode_current_player(self):
        """测试当前玩家编码"""
        # 红方回合
        red_board = ChessBoard()
        red_features = self.encoder.encode_board(red_board)
        assert torch.all(red_features[16, :, :] == 1.0)  # 红方通道应该全为1
        
        # 黑方回合
        black_board = ChessBoard()
        black_board.current_player = -1
        black_features = self.encoder.encode_board(black_board)
        assert torch.all(black_features[16, :, :] == 0.0)  # 黑方时应该全为0
    
    def test_encode_with_history(self):
        """测试带历史的编码"""
        history_boards = [self.board.copy() for _ in range(3)]
        features = self.encoder.encode_board_with_history(self.board, history_boards)
        
        # 应该有8个历史状态 * 20通道 = 160通道
        assert features.shape == (160, 10, 9)
    
    def test_move_encoding_decoding(self):
        """测试走法编码和解码"""
        move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        
        # 编码走法
        policy_index = self.encoder.encode_move_to_policy_index(move)
        assert 0 <= policy_index < 8100
        
        # 解码走法
        decoded_move = self.encoder.decode_policy_index_to_move(policy_index, self.board)
        
        # 验证解码结果
        if decoded_move:  # 如果解码成功
            assert decoded_move.from_pos == move.from_pos
            assert decoded_move.to_pos == move.to_pos
    
    def test_policy_target_creation(self):
        """测试策略目标创建"""
        legal_moves = self.board.get_legal_moves()[:5]  # 取前5个合法走法
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        policy_target = self.encoder.create_policy_target(legal_moves, move_probs)
        
        assert policy_target.shape == (8100,)
        assert abs(policy_target.sum() - 1.0) < 1e-6  # 应该归一化
        
        # 检查非零元素数量
        non_zero_count = np.count_nonzero(policy_target)
        assert non_zero_count <= len(legal_moves)
    
    def test_extract_legal_move_probabilities(self):
        """测试提取合法走法概率"""
        legal_moves = self.board.get_legal_moves()[:3]
        policy_output = np.random.random(8100)
        
        move_probs = self.encoder.extract_legal_move_probabilities(policy_output, legal_moves)
        
        assert len(move_probs) == len(legal_moves)
        assert abs(sum(move_probs.values()) - 1.0) < 1e-6  # 应该归一化
        
        for move in legal_moves:
            assert move in move_probs
            assert move_probs[move] >= 0
    
    def test_feature_info(self):
        """测试特征信息"""
        info = self.encoder.get_feature_info()
        
        assert isinstance(info, dict)
        assert len(info) > 0
        assert '0-6' in info  # 红方棋子通道
        assert '7-13' in info  # 黑方棋子通道
    
    def test_visualize_features(self):
        """测试特征可视化"""
        features = self.encoder.encode_board(self.board)
        
        # 可视化第0通道（红方帅）
        visualization = self.encoder.visualize_features(features, 0)
        assert isinstance(visualization, str)
        assert "通道 0" in visualization
        
        # 测试无效通道
        invalid_vis = self.encoder.visualize_features(features, 100)
        assert "不存在" in invalid_vis


class TestSelfPlayGenerator:
    """测试自对弈生成器"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建模拟的神经网络模型
        self.mock_model = Mock(spec=ChessNet)
        
        # 创建配置
        self.config = SelfPlayConfig(
            num_games=2,
            max_game_length=50,
            mcts_simulations=10,  # 减少模拟次数以加快测试
            num_workers=1  # 单线程测试
        )
        
        # 创建生成器
        self.generator = SelfPlayGenerator(
            model=self.mock_model,
            config=self.config
        )
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.training_framework.self_play_generator.MCTSSearcher')
    def test_generate_single_game(self, mock_searcher_class):
        """测试生成单个游戏"""
        # 模拟MCTS搜索器
        mock_searcher = Mock()
        mock_searcher_class.return_value = mock_searcher
        
        # 模拟搜索结果
        mock_root_node = Mock()
        mock_searcher.search.return_value = mock_root_node
        
        # 确保action_probs与legal_moves长度匹配
        def mock_get_action_probs(*args, **kwargs):
            return np.array([0.8, 0.2])  # 与下面的legal_moves长度匹配
        
        mock_searcher.get_action_probabilities.side_effect = mock_get_action_probs
        mock_searcher.get_value_estimate.return_value = 0.3
        
        # 模拟棋盘状态
        with patch('chess_ai_project.src.chinese_chess_ai_engine.rules_engine.ChessBoard') as mock_board_class:
            mock_board = Mock()
            mock_board_class.return_value = mock_board
            
            # 设置游戏在几步后结束
            mock_board.is_game_over.side_effect = [False, False, True]
            mock_board.get_legal_moves.return_value = [
                Move(from_pos=(0, 0), to_pos=(1, 0), piece=1),
                Move(from_pos=(0, 1), to_pos=(1, 1), piece=2)
            ]
            mock_board.get_winner.return_value = 1
            mock_board.current_player = 1
            mock_board.copy.return_value = mock_board
            mock_board.make_move.return_value = mock_board
            
            # 生成游戏
            game_result = self.generator.generate_game("test_game")
            
            # 验证结果
            assert isinstance(game_result, GameResult)
            assert game_result.game_id == "test_game"
            assert game_result.winner in [-1, 0, 1]
            assert len(game_result.training_examples) > 0
            assert game_result.game_length == len(game_result.training_examples)
    
    def test_config_validation(self):
        """测试配置验证"""
        config = SelfPlayConfig(
            num_games=5,
            max_game_length=100,
            mcts_simulations=50
        )
        
        assert config.num_games == 5
        assert config.max_game_length == 100
        assert config.mcts_simulations == 50
        assert config.temperature_threshold == 30
        assert config.c_puct == 1.0
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        initial_stats = self.generator.get_statistics()
        
        assert initial_stats['games_played'] == 0
        assert initial_stats['total_moves'] == 0
        assert initial_stats['red_wins'] == 0
        assert initial_stats['black_wins'] == 0
        assert initial_stats['draws'] == 0
        
        # 重置统计
        self.generator.reset_statistics()
        reset_stats = self.generator.get_statistics()
        assert reset_stats['games_played'] == 0
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.training_framework.self_play_generator.MCTSSearcher')
    def test_collect_training_data(self, mock_searcher_class):
        """测试收集训练数据"""
        # 模拟简单的游戏生成
        with patch.object(self.generator, 'generate_games_parallel') as mock_generate:
            # 创建模拟的游戏结果
            mock_examples = [
                TrainingExample(
                    board_tensor=torch.randn(20, 10, 9),
                    policy_target=np.random.random(8100).astype(np.float32),
                    value_target=0.5,
                    game_result=1,
                    move_number=1,
                    current_player=1
                )
            ]
            
            mock_game_result = GameResult(
                winner=1,
                game_length=1,
                training_examples=mock_examples,
                game_id="test_game"
            )
            
            mock_generate.return_value = [mock_game_result]
            
            # 收集训练数据
            dataset = self.generator.collect_training_data(num_games=1)
            
            assert isinstance(dataset, TrainingDataset)
            assert len(dataset) == 1
    
    def test_save_training_data(self):
        """测试保存训练数据"""
        # 创建模拟的游戏结果
        mock_examples = [
            TrainingExample(
                board_tensor=torch.randn(20, 10, 9),
                policy_target=np.random.random(8100).astype(np.float32),
                value_target=0.5,
                game_result=1,
                move_number=1,
                current_player=1
            )
        ]
        
        game_results = [
            GameResult(
                winner=1,
                game_length=1,
                training_examples=mock_examples,
                game_id="test_game"
            )
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'training_data.pkl')
            
            self.generator.save_training_data(game_results, output_path)
            
            # 验证文件是否创建
            assert os.path.exists(output_path)
            
            # 验证可以加载
            loaded_dataset = TrainingDataset.load_from_file(output_path)
            assert len(loaded_dataset) == 1


if __name__ == '__main__':
    pytest.main([__file__])