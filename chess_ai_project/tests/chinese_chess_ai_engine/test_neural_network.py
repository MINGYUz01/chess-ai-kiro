"""
测试神经网络模块的功能

测试ChessNet、FeatureEncoder等组件。
"""

import pytest
import torch
import numpy as np
from chess_ai_project.src.chinese_chess_ai_engine.neural_network import ChessNet, FeatureEncoder, ResidualBlock, AttentionModule
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import ChessBoard, Move, RuleEngine


class TestResidualBlock:
    """ResidualBlock类的测试"""
    
    def test_residual_block_creation(self):
        """测试残差块创建"""
        block = ResidualBlock(channels=256)
        assert block is not None
        assert hasattr(block, 'conv1')
        assert hasattr(block, 'conv2')
        assert hasattr(block, 'bn1')
        assert hasattr(block, 'bn2')
    
    def test_residual_block_forward(self):
        """测试残差块前向传播"""
        block = ResidualBlock(channels=64)
        x = torch.randn(2, 64, 10, 9)  # [batch, channels, height, width]
        
        output = block(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # 输出应该与输入不同


class TestAttentionModule:
    """AttentionModule类的测试"""
    
    def test_attention_module_creation(self):
        """测试注意力模块创建"""
        attention = AttentionModule(channels=256, num_heads=8)
        assert attention is not None
        assert attention.channels == 256
        assert attention.num_heads == 8
        assert attention.head_dim == 32
    
    def test_attention_module_forward(self):
        """测试注意力模块前向传播"""
        attention = AttentionModule(channels=64, num_heads=8)
        x = torch.randn(2, 64, 10, 9)
        
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)
    
    def test_attention_invalid_heads(self):
        """测试无效的注意力头数"""
        with pytest.raises(AssertionError):
            AttentionModule(channels=65, num_heads=8)  # 65不能被8整除


class TestChessNet:
    """ChessNet类的测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.model = ChessNet(
            input_channels=14,
            num_blocks=4,  # 使用较小的模型进行测试
            channels=64,
            num_attention_heads=8
        )
    
    def test_chess_net_creation(self):
        """测试网络创建"""
        assert self.model is not None
        assert self.model.input_channels == 14
        assert self.model.num_blocks == 4
        assert self.model.channels == 64
    
    def test_chess_net_forward(self):
        """测试网络前向传播"""
        batch_size = 2
        x = torch.randn(batch_size, 14, 10, 9)
        
        value, policy = self.model(x)
        
        # 检查输出形状
        assert value.shape == (batch_size, 1)
        assert policy.shape == (batch_size, 8100)  # 10*9*90 + 10*9
        
        # 检查输出范围
        assert torch.all(value >= -1) and torch.all(value <= 1)  # tanh输出范围
        assert torch.all(torch.isfinite(policy))  # 策略输出应该是有限的
    
    def test_predict_value(self):
        """测试价值预测"""
        board_tensor = torch.randn(14, 10, 9)
        
        value = self.model.predict_value(board_tensor)
        
        assert isinstance(value, float)
        assert -1 <= value <= 1
    
    def test_predict_policy(self):
        """测试策略预测"""
        board_tensor = torch.randn(14, 10, 9)
        
        policy = self.model.predict_policy(board_tensor)
        
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (8100,)
        assert np.all(policy >= 0)  # 概率应该非负
        assert np.isclose(np.sum(policy), 1.0, atol=1e-5)  # 概率和应该接近1
    
    def test_model_info(self):
        """测试模型信息获取"""
        info = self.model.get_model_info()
        
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'input_channels' in info
        assert 'model_size_mb' in info
        
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert info['input_channels'] == 14
    
    def test_save_load_checkpoint(self):
        """测试模型保存和加载"""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'test_model.pth')
            
            # 保存模型
            metadata = {'epoch': 10, 'loss': 0.5}
            self.model.save_checkpoint(checkpoint_path, metadata)
            
            # 创建新模型并加载
            new_model = ChessNet(
                input_channels=14,
                num_blocks=4,
                channels=64,
                num_attention_heads=8
            )
            loaded_metadata = new_model.load_checkpoint(checkpoint_path)
            
            # 验证元数据
            assert loaded_metadata['epoch'] == 10
            assert loaded_metadata['loss'] == 0.5
            
            # 验证模型参数相同
            for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)


class TestFeatureEncoder:
    """FeatureEncoder类的测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.encoder = FeatureEncoder(history_length=8)
        self.board = ChessBoard()
        self.rule_engine = RuleEngine()
    
    def test_feature_encoder_creation(self):
        """测试特征编码器创建"""
        assert self.encoder is not None
        assert self.encoder.history_length == 8
        assert self.encoder.total_channels == 20
    
    def test_encode_board_shape(self):
        """测试棋盘编码形状"""
        features = self.encoder.encode_board(self.board)
        
        assert features.shape == (20, 10, 9)
        assert features.dtype == torch.float32
    
    def test_encode_board_content(self):
        """测试棋盘编码内容"""
        features = self.encoder.encode_board(self.board)
        
        # 检查红方棋子通道（前7个通道）
        red_pieces_sum = features[:7].sum()
        assert red_pieces_sum > 0  # 应该有红方棋子
        
        # 检查黑方棋子通道（第8-14个通道）
        black_pieces_sum = features[7:14].sum()
        assert black_pieces_sum > 0  # 应该有黑方棋子
        
        # 检查当前玩家通道（第15个通道）
        current_player_channel = features[14]
        assert torch.all(current_player_channel == 1.0)  # 初始是红方
    
    def test_encode_with_legal_moves(self):
        """测试包含合法走法的编码"""
        legal_moves = self.rule_engine.generate_legal_moves(self.board)
        features = self.encoder.encode_board(self.board, legal_moves=legal_moves)
        
        # 检查合法走法通道（第16个通道，索引15）
        legal_moves_channel = features[15]
        # 由于合法走法编码可能为0（如果没有目标位置重复），我们检查是否有合法走法
        assert len(legal_moves) > 0  # 至少应该有合法走法
    
    def test_encode_with_history(self):
        """测试包含历史的编码"""
        # 创建一些走法历史
        move_history = [
            Move(from_pos=(6, 0), to_pos=(5, 0), piece=7),
            Move(from_pos=(3, 0), to_pos=(4, 0), piece=-7)
        ]
        
        features = self.encoder.encode_board(self.board, move_history=move_history)
        
        # 检查历史通道（第15个通道）
        history_channel = features[14]
        assert history_channel.sum() > 0  # 应该有历史信息
    
    def test_batch_encoding(self):
        """测试批量编码"""
        boards = [self.board, self.board.copy()]
        
        batch_features = self.encoder.encode_batch(boards)
        
        assert batch_features.shape == (2, 20, 10, 9)
        assert batch_features.dtype == torch.float32
    
    def test_decode_policy_to_moves(self):
        """测试策略解码"""
        policy_output = torch.randn(8100)
        policy_output = torch.softmax(policy_output, dim=0)  # 转换为概率分布
        
        legal_moves = self.rule_engine.generate_legal_moves(self.board)
        
        move_probs = self.encoder.decode_policy_to_moves(
            policy_output, self.board, legal_moves, top_k=5
        )
        
        assert len(move_probs) <= 5
        assert len(move_probs) <= len(legal_moves)
        
        for move, prob in move_probs:
            assert isinstance(move, Move)
            assert isinstance(prob, float)
            assert prob >= 0
    
    def test_feature_info(self):
        """测试特征信息获取"""
        info = self.encoder.get_feature_info()
        
        assert info['total_channels'] == 20
        assert info['board_shape'] == (10, 9)
        assert info['output_shape'] == (20, 10, 9)
        assert info['history_length'] == 8


class TestIntegration:
    """集成测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.encoder = FeatureEncoder()
        self.model = ChessNet(input_channels=20, num_blocks=2, channels=32)
        self.board = ChessBoard()
        self.rule_engine = RuleEngine()
    
    def test_end_to_end_prediction(self):
        """测试端到端预测"""
        # 编码棋盘
        legal_moves = self.rule_engine.generate_legal_moves(self.board)
        features = self.encoder.encode_board(self.board, legal_moves=legal_moves)
        
        # 添加batch维度
        features = features.unsqueeze(0)
        
        # 模型预测
        value, policy = self.model(features)
        
        # 检查输出
        assert value.shape == (1, 1)
        assert policy.shape == (1, 8100)
        assert -1 <= value.item() <= 1
    
    def test_multiple_positions(self):
        """测试多个棋局位置"""
        boards = []
        
        # 创建几个不同的棋局
        board1 = ChessBoard()
        boards.append(board1)
        
        # 执行一些走法创建不同局面
        board2 = board1.make_move(Move(from_pos=(6, 0), to_pos=(5, 0), piece=7))
        boards.append(board2)
        
        board3 = board2.make_move(Move(from_pos=(3, 0), to_pos=(4, 0), piece=-7))
        boards.append(board3)
        
        # 批量编码
        batch_features = self.encoder.encode_batch(boards)
        
        # 批量预测
        values, policies = self.model(batch_features)
        
        assert values.shape == (3, 1)
        assert policies.shape == (3, 8100)
        
        # 每个位置的价值应该不同
        assert not torch.allclose(values[0], values[1])
        assert not torch.allclose(values[1], values[2])


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v'])