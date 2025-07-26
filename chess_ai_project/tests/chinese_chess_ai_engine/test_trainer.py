"""
训练器模块测试

测试神经网络训练器的各种功能。
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from chess_ai_project.src.chinese_chess_ai_engine.training_framework import (
    TrainingConfig,
    EvaluationConfig,
    Trainer,
    ChessTrainingDataset,
    EarlyStopping,
    ModelEMA,
    TrainingExample,
    TrainingDataset
)
from chess_ai_project.src.chinese_chess_ai_engine.neural_network import ChessNet


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


class TestTrainingConfig:
    """测试训练配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.optimizer == 'adam'
        assert config.lr_scheduler == 'cosine'
        assert config.device in ['cpu', 'cuda', 'mps']
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            learning_rate=0.01,
            batch_size=64,
            num_epochs=50,
            optimizer='sgd',
            lr_scheduler='step'
        )
        
        assert config.learning_rate == 0.01
        assert config.batch_size == 64
        assert config.num_epochs == 50
        assert config.optimizer == 'sgd'
        assert config.lr_scheduler == 'step'
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效学习率
        with pytest.raises(ValueError, match="学习率必须大于0"):
            TrainingConfig(learning_rate=-0.001)
        
        # 测试无效批次大小
        with pytest.raises(ValueError, match="批次大小必须大于0"):
            TrainingConfig(batch_size=0)
        
        # 测试无效优化器
        with pytest.raises(ValueError, match="不支持的优化器"):
            TrainingConfig(optimizer='invalid')
        
        # 测试无效调度器
        with pytest.raises(ValueError, match="不支持的学习率调度器"):
            TrainingConfig(lr_scheduler='invalid')
    
    def test_optimizer_params(self):
        """测试优化器参数"""
        # Adam优化器
        config = TrainingConfig(optimizer='adam', learning_rate=0.01, weight_decay=1e-3)
        params = config.get_optimizer_params()
        
        assert params['lr'] == 0.01
        assert params['weight_decay'] == 1e-3
        assert params['betas'] == (0.9, 0.999)
        assert params['eps'] == 1e-8
        
        # SGD优化器
        config = TrainingConfig(optimizer='sgd', momentum=0.95)
        params = config.get_optimizer_params()
        
        assert params['momentum'] == 0.95
        assert params['nesterov'] == True
    
    def test_scheduler_params(self):
        """测试调度器参数"""
        # Cosine调度器
        config = TrainingConfig(lr_scheduler='cosine', num_epochs=100)
        params = config.get_scheduler_params()
        
        assert params['T_max'] == 100
        assert 'eta_min' in params
        
        # Step调度器
        config = TrainingConfig(lr_scheduler='step', num_epochs=90)
        params = config.get_scheduler_params()
        
        assert params['step_size'] == 30  # num_epochs // 3
        assert params['gamma'] == 0.1
    
    def test_config_serialization(self):
        """测试配置序列化"""
        config = TrainingConfig(learning_rate=0.01, batch_size=64)
        
        # 转换为字典
        config_dict = config.to_dict()
        assert config_dict['learning_rate'] == 0.01
        assert config_dict['batch_size'] == 64
        
        # 从字典重建
        new_config = TrainingConfig.from_dict(config_dict)
        assert new_config.learning_rate == 0.01
        assert new_config.batch_size == 64
    
    def test_config_file_operations(self):
        """测试配置文件操作"""
        config = TrainingConfig(learning_rate=0.01, batch_size=64)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'config.json')
            
            # 保存配置
            config.save_to_file(config_path)
            assert os.path.exists(config_path)
            
            # 加载配置
            loaded_config = TrainingConfig.load_from_file(config_path)
            assert loaded_config.learning_rate == 0.01
            assert loaded_config.batch_size == 64


class TestEvaluationConfig:
    """测试评估配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = EvaluationConfig()
        
        assert config.batch_size == 64
        assert config.num_games == 100
        assert config.initial_elo == 1500.0
        assert config.k_factor == 32.0
        assert config.device in ['cpu', 'cuda', 'mps']
    
    def test_config_serialization(self):
        """测试配置序列化"""
        config = EvaluationConfig(num_games=50, initial_elo=1600.0)
        
        config_dict = config.to_dict()
        assert config_dict['num_games'] == 50
        assert config_dict['initial_elo'] == 1600.0
        
        new_config = EvaluationConfig.from_dict(config_dict)
        assert new_config.num_games == 50
        assert new_config.initial_elo == 1600.0


class TestChessTrainingDataset:
    """测试象棋训练数据集"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建训练样本
        examples = []
        for i in range(10):
            example = TrainingExample(
                board_tensor=torch.randn(20, 10, 9),
                policy_target=np.random.random(8100).astype(np.float32),
                value_target=np.random.uniform(-1, 1),
                game_result=np.random.choice([-1, 0, 1]),
                move_number=i + 1,
                current_player=1 if i % 2 == 0 else -1
            )
            examples.append(example)
        
        self.training_dataset = TrainingDataset(examples)
        self.chess_dataset = ChessTrainingDataset(self.training_dataset)
    
    def test_dataset_length(self):
        """测试数据集长度"""
        assert len(self.chess_dataset) == 10
    
    def test_dataset_getitem(self):
        """测试获取数据项"""
        board_tensor, policy_target, value_target = self.chess_dataset[0]
        
        assert isinstance(board_tensor, torch.Tensor)
        assert isinstance(policy_target, torch.Tensor)
        assert isinstance(value_target, torch.Tensor)
        
        assert board_tensor.shape == (20, 10, 9)
        assert policy_target.shape == (8100,)
        assert value_target.shape == ()


class TestEarlyStopping:
    """测试早停机制"""
    
    def test_early_stopping_min_mode(self):
        """测试最小化模式的早停"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        
        # 初始分数
        assert not early_stopping(1.0)
        assert not early_stopping.early_stop
        
        # 改善的分数
        assert not early_stopping(0.8)
        assert not early_stopping.early_stop
        
        # 没有改善的分数
        result1 = early_stopping(0.85)  # 第1次，不应该早停
        assert not result1
        assert not early_stopping.early_stop
        
        result2 = early_stopping(0.82)  # 第2次，不应该早停
        assert not result2
        assert not early_stopping.early_stop
        
        result3 = early_stopping(0.83)  # 第3次，应该早停
        assert result3
        assert early_stopping.early_stop
    
    def test_early_stopping_max_mode(self):
        """测试最大化模式的早停"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='max')
        
        # 初始分数
        assert not early_stopping(0.5)
        
        # 改善的分数
        assert not early_stopping(0.7)
        
        # 没有改善的分数
        result1 = early_stopping(0.65)  # 第1次，不应该早停
        assert not result1
        assert not early_stopping.early_stop
        
        result2 = early_stopping(0.68)  # 第2次，应该早停
        assert result2
        assert early_stopping.early_stop


class TestModelEMA:
    """测试模型指数移动平均"""
    
    def test_model_ema_initialization(self):
        """测试EMA初始化"""
        model = MockChessNet()
        ema = ModelEMA(model, decay=0.999)
        
        assert ema.decay == 0.999
        assert len(ema.shadow) > 0
        
        # 检查shadow参数是否正确初始化
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.equal(ema.shadow[name], param.data)
    
    def test_model_ema_update(self):
        """测试EMA更新"""
        model = MockChessNet()
        ema = ModelEMA(model, decay=0.9)
        
        # 保存原始参数
        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()
        
        # 修改模型参数
        with torch.no_grad():
            for param in model.parameters():
                param.data.fill_(1.0)
        
        # 更新EMA
        ema.update()
        
        # 检查shadow参数是否正确更新
        for name, param in model.named_parameters():
            if param.requires_grad:
                expected = 0.1 * param.data + 0.9 * original_params[name]
                assert torch.allclose(ema.shadow[name], expected, atol=1e-6)
    
    def test_model_ema_apply_restore(self):
        """测试EMA应用和恢复"""
        model = MockChessNet()
        ema = ModelEMA(model, decay=0.9)
        
        # 保存原始参数
        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()
        
        # 修改shadow参数
        for name in ema.shadow:
            ema.shadow[name].fill_(2.0)
        
        # 应用shadow参数
        ema.apply_shadow()
        
        # 检查模型参数是否被替换
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param.data, torch.full_like(param.data, 2.0))
        
        # 恢复原始参数
        ema.restore()
        
        # 检查参数是否恢复
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param.data, original_params[name])


class TestTrainer:
    """测试训练器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.model = MockChessNet()
        self.config = TrainingConfig(
            learning_rate=0.01,
            batch_size=4,
            num_epochs=2,
            log_interval=1,
            validation_interval=1,
            save_interval=1,
            tensorboard_log=False,
            early_stopping=False,
            model_ema=False,
            mixed_precision=False,
            compile_model=False
        )
        
        # 创建训练数据
        examples = []
        for i in range(8):
            example = TrainingExample(
                board_tensor=torch.randn(20, 10, 9),
                policy_target=np.random.random(8100).astype(np.float32),
                value_target=np.random.uniform(-1, 1),
                game_result=np.random.choice([-1, 0, 1]),
                move_number=i + 1,
                current_player=1 if i % 2 == 0 else -1
            )
            examples.append(example)
        
        self.training_dataset = TrainingDataset(examples)
        self.validation_dataset = TrainingDataset(examples[:4])
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(self.model, self.config, checkpoint_dir=temp_dir)
            
            assert trainer.model == self.model
            assert trainer.config == self.config
            assert trainer.device.type == self.config.device
            assert trainer.optimizer is not None
            assert trainer.current_epoch == 0
            assert trainer.global_step == 0
            
            # 检查配置文件是否保存
            config_path = Path(temp_dir) / 'config.json'
            assert config_path.exists()
    
    def test_optimizer_creation(self):
        """测试优化器创建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试Adam优化器
            config = TrainingConfig(optimizer='adam', learning_rate=0.01)
            trainer = Trainer(self.model, config, checkpoint_dir=temp_dir)
            assert isinstance(trainer.optimizer, torch.optim.Adam)
            
            # 测试SGD优化器
            config = TrainingConfig(optimizer='sgd', learning_rate=0.01)
            trainer = Trainer(self.model, config, checkpoint_dir=temp_dir)
            assert isinstance(trainer.optimizer, torch.optim.SGD)
            
            # 测试AdamW优化器
            config = TrainingConfig(optimizer='adamw', learning_rate=0.01)
            trainer = Trainer(self.model, config, checkpoint_dir=temp_dir)
            assert isinstance(trainer.optimizer, torch.optim.AdamW)
    
    def test_scheduler_creation(self):
        """测试学习率调度器创建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试Cosine调度器
            config = TrainingConfig(lr_scheduler='cosine')
            trainer = Trainer(self.model, config, checkpoint_dir=temp_dir)
            assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
            
            # 测试Step调度器
            config = TrainingConfig(lr_scheduler='step')
            trainer = Trainer(self.model, config, checkpoint_dir=temp_dir)
            assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)
            
            # 测试常量调度器（无调度器）
            config = TrainingConfig(lr_scheduler='constant')
            trainer = Trainer(self.model, config, checkpoint_dir=temp_dir)
            assert trainer.scheduler is None
    
    def test_train_epoch(self):
        """测试训练一个epoch"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(self.model, self.config, checkpoint_dir=temp_dir)
            
            # 创建数据加载器
            chess_dataset = ChessTrainingDataset(self.training_dataset)
            train_loader = torch.utils.data.DataLoader(
                chess_dataset, batch_size=self.config.batch_size, shuffle=True
            )
            
            # 训练一个epoch
            metrics = trainer.train_epoch(train_loader, epoch=0)
            
            # 检查返回的指标
            assert 'loss' in metrics
            assert 'value_loss' in metrics
            assert 'policy_loss' in metrics
            assert 'learning_rate' in metrics
            
            assert isinstance(metrics['loss'], float)
            assert metrics['loss'] >= 0
            assert trainer.global_step > 0
    
    def test_validate(self):
        """测试验证"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(self.model, self.config, checkpoint_dir=temp_dir)
            
            # 创建验证数据加载器
            chess_dataset = ChessTrainingDataset(self.validation_dataset)
            val_loader = torch.utils.data.DataLoader(
                chess_dataset, batch_size=self.config.batch_size, shuffle=False
            )
            
            # 验证
            metrics = trainer.validate(val_loader, epoch=0)
            
            # 检查返回的指标
            assert 'loss' in metrics
            assert 'value_loss' in metrics
            assert 'policy_loss' in metrics
            assert 'value_mae' in metrics
            assert 'policy_accuracy' in metrics
            
            assert isinstance(metrics['loss'], float)
            assert metrics['loss'] >= 0
            assert 0 <= metrics['policy_accuracy'] <= 1
    
    def test_full_training(self):
        """测试完整训练流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(self.model, self.config, checkpoint_dir=temp_dir)
            
            # 执行训练
            history = trainer.train(self.training_dataset, self.validation_dataset)
            
            # 检查训练历史
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) == self.config.num_epochs
            assert len(history['val_loss']) == self.config.num_epochs
            
            # 检查检查点文件
            checkpoint_dir = Path(temp_dir)
            assert (checkpoint_dir / 'best_model.pth').exists()
            assert (checkpoint_dir / 'final_model.pth').exists()
    
    def test_checkpoint_save_load(self):
        """测试检查点保存和加载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(self.model, self.config, checkpoint_dir=temp_dir)
            
            # 设置一些训练状态
            trainer.current_epoch = 5
            trainer.global_step = 100
            trainer.best_loss = 0.5
            
            # 保存检查点
            metrics = {'loss': 0.4, 'accuracy': 0.8}
            trainer.save_checkpoint(5, metrics, is_best=True)
            
            # 检查文件是否存在
            checkpoint_path = Path(temp_dir) / 'best_model.pth'
            assert checkpoint_path.exists()
            
            # 创建新的训练器并加载检查点
            new_trainer = Trainer(MockChessNet(), self.config, checkpoint_dir=temp_dir)
            checkpoint_info = new_trainer.load_checkpoint(str(checkpoint_path))
            
            # 检查状态是否正确恢复
            assert new_trainer.current_epoch == 6  # epoch + 1
            assert new_trainer.global_step == 100
            assert new_trainer.best_loss == 0.5
            assert checkpoint_info['metrics'] == metrics
    
    def test_early_stopping_integration(self):
        """测试早停机制集成"""
        config = TrainingConfig(
            batch_size=4,
            num_epochs=10,
            early_stopping=True,
            early_stopping_patience=2,
            validation_interval=1,
            tensorboard_log=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer(self.model, config, checkpoint_dir=temp_dir)
            
            # 模拟训练过程中验证损失不再改善
            with patch.object(trainer, 'validate') as mock_validate:
                mock_validate.side_effect = [
                    {'loss': 1.0, 'value_loss': 0.5, 'policy_loss': 0.5, 'value_mae': 0.3, 'policy_accuracy': 0.7},
                    {'loss': 0.9, 'value_loss': 0.4, 'policy_loss': 0.5, 'value_mae': 0.3, 'policy_accuracy': 0.7},
                    {'loss': 0.95, 'value_loss': 0.45, 'policy_loss': 0.5, 'value_mae': 0.3, 'policy_accuracy': 0.7},
                    {'loss': 0.96, 'value_loss': 0.46, 'policy_loss': 0.5, 'value_mae': 0.3, 'policy_accuracy': 0.7},
                ]
                
                with patch.object(trainer, 'train_epoch') as mock_train_epoch:
                    mock_train_epoch.return_value = {
                        'loss': 0.8, 'value_loss': 0.4, 'policy_loss': 0.4, 'learning_rate': 0.01
                    }
                    
                    # 执行训练
                    history = trainer.train(self.training_dataset, self.validation_dataset)
                    
                    # 检查是否提前停止
                    assert len(history['train_loss']) < config.num_epochs


if __name__ == '__main__':
    pytest.main([__file__])