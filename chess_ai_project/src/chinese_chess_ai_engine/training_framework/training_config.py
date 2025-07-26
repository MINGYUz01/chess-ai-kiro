"""
训练配置类

定义神经网络训练的各种配置参数。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch


@dataclass
class TrainingConfig:
    """
    训练配置类
    
    包含神经网络训练的所有配置参数。
    """
    
    # 基础训练参数
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-4
    
    # 学习率调度器
    lr_scheduler: str = 'cosine'  # 'cosine', 'step', 'exponential', 'plateau'
    lr_scheduler_params: Dict[str, Any] = None
    
    # 优化器设置
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    momentum: float = 0.9  # 仅用于SGD
    beta1: float = 0.9     # 仅用于Adam/AdamW
    beta2: float = 0.999   # 仅用于Adam/AdamW
    eps: float = 1e-8      # 仅用于Adam/AdamW
    
    # 损失函数权重
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    
    # 正则化
    gradient_clip_norm: Optional[float] = 1.0
    dropout_rate: float = 0.0
    
    # 设备和精度
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    mixed_precision: bool = False
    compile_model: bool = False
    
    # 验证和保存
    validation_interval: int = 1  # 每几个epoch验证一次
    save_interval: int = 5        # 每几个epoch保存一次
    max_checkpoints: int = 5      # 最多保存几个检查点
    
    # 早停
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # 日志和监控
    log_interval: int = 100  # 每几个batch记录一次日志
    tensorboard_log: bool = True
    wandb_log: bool = False
    wandb_project: Optional[str] = None
    
    # 模型相关
    model_ema: bool = False      # 指数移动平均
    model_ema_decay: float = 0.9999
    
    # 高级设置
    accumulate_grad_batches: int = 1  # 梯度累积
    find_unused_parameters: bool = False  # DDP设置
    
    def __post_init__(self):
        """初始化后处理"""
        if self.lr_scheduler_params is None:
            self.lr_scheduler_params = {}
        
        # 自动设置设备
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        
        # 验证参数
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        if self.learning_rate <= 0:
            raise ValueError("学习率必须大于0")
        
        if self.batch_size <= 0:
            raise ValueError("批次大小必须大于0")
        
        if self.num_epochs <= 0:
            raise ValueError("训练轮数必须大于0")
        
        if self.weight_decay < 0:
            raise ValueError("权重衰减必须非负")
        
        if self.lr_scheduler not in ['cosine', 'step', 'exponential', 'plateau', 'linear', 'constant']:
            raise ValueError(f"不支持的学习率调度器: {self.lr_scheduler}")
        
        if self.optimizer not in ['adam', 'sgd', 'adamw']:
            raise ValueError(f"不支持的优化器: {self.optimizer}")
        
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("梯度裁剪范数必须大于0")
        
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("Dropout率必须在[0,1]范围内")
        
        if self.accumulate_grad_batches <= 0:
            raise ValueError("梯度累积批次数必须大于0")
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """
        获取优化器参数
        
        Returns:
            Dict[str, Any]: 优化器参数字典
        """
        base_params = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        if self.optimizer == 'sgd':
            base_params.update({
                'momentum': self.momentum,
                'nesterov': True
            })
        elif self.optimizer in ['adam', 'adamw']:
            base_params.update({
                'betas': (self.beta1, self.beta2),
                'eps': self.eps
            })
        
        return base_params
    
    def get_scheduler_params(self) -> Dict[str, Any]:
        """
        获取学习率调度器参数
        
        Returns:
            Dict[str, Any]: 调度器参数字典
        """
        params = self.lr_scheduler_params.copy()
        
        # 设置默认参数
        if self.lr_scheduler == 'cosine':
            params.setdefault('T_max', self.num_epochs)
            params.setdefault('eta_min', self.learning_rate * 0.01)
        elif self.lr_scheduler == 'step':
            params.setdefault('step_size', self.num_epochs // 3)
            params.setdefault('gamma', 0.1)
        elif self.lr_scheduler == 'exponential':
            params.setdefault('gamma', 0.95)
        elif self.lr_scheduler == 'plateau':
            params.setdefault('mode', 'min')
            params.setdefault('factor', 0.5)
            params.setdefault('patience', 5)
            params.setdefault('min_lr', self.learning_rate * 0.001)
        
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'lr_scheduler': self.lr_scheduler,
            'lr_scheduler_params': self.lr_scheduler_params,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'value_loss_weight': self.value_loss_weight,
            'policy_loss_weight': self.policy_loss_weight,
            'gradient_clip_norm': self.gradient_clip_norm,
            'dropout_rate': self.dropout_rate,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'compile_model': self.compile_model,
            'validation_interval': self.validation_interval,
            'save_interval': self.save_interval,
            'max_checkpoints': self.max_checkpoints,
            'early_stopping': self.early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'prefetch_factor': self.prefetch_factor,
            'log_interval': self.log_interval,
            'tensorboard_log': self.tensorboard_log,
            'wandb_log': self.wandb_log,
            'wandb_project': self.wandb_project,
            'model_ema': self.model_ema,
            'model_ema_decay': self.model_ema_decay,
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'find_unused_parameters': self.find_unused_parameters
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """
        从字典创建配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            TrainingConfig: 配置对象
        """
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """
        保存配置到文件
        
        Args:
            filepath: 文件路径
        """
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrainingConfig':
        """
        从文件加载配置
        
        Args:
            filepath: 文件路径
            
        Returns:
            TrainingConfig: 配置对象
        """
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"TrainingConfig(lr={self.learning_rate}, batch_size={self.batch_size}, epochs={self.num_epochs})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


@dataclass
class EvaluationConfig:
    """
    评估配置类
    
    包含模型评估的各种配置参数。
    """
    
    # 基础评估参数
    batch_size: int = 64
    num_workers: int = 4
    device: str = 'auto'
    
    # 对弈评估
    num_games: int = 100
    max_game_length: int = 200
    time_per_move: float = 1.0
    
    # ELO计算
    initial_elo: float = 1500.0
    k_factor: float = 32.0
    
    # 基准测试
    benchmark_positions: Optional[List[str]] = None  # FEN格式的测试局面
    benchmark_timeout: float = 5.0
    
    # 统计分析
    confidence_level: float = 0.95
    min_games_for_rating: int = 50
    
    def __post_init__(self):
        """初始化后处理"""
        # 自动设置设备
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'device': self.device,
            'num_games': self.num_games,
            'max_game_length': self.max_game_length,
            'time_per_move': self.time_per_move,
            'initial_elo': self.initial_elo,
            'k_factor': self.k_factor,
            'benchmark_positions': self.benchmark_positions,
            'benchmark_timeout': self.benchmark_timeout,
            'confidence_level': self.confidence_level,
            'min_games_for_rating': self.min_games_for_rating
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)