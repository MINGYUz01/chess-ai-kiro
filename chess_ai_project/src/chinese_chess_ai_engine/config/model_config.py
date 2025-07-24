"""
模型配置数据结构

定义各种配置类和默认参数。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class MCTSConfig:
    """蒙特卡洛树搜索配置"""
    num_simulations: int = 800          # 模拟次数
    c_puct: float = 1.0                 # UCB公式中的探索常数
    temperature: float = 1.0            # 温度参数，控制随机性
    dirichlet_alpha: float = 0.3        # Dirichlet噪声的alpha参数
    dirichlet_epsilon: float = 0.25     # Dirichlet噪声的权重
    max_depth: int = 100                # 最大搜索深度
    time_limit: float = 5.0             # 时间限制(秒)
    use_virtual_loss: bool = True       # 是否使用虚拟损失
    virtual_loss_value: float = 3.0     # 虚拟损失值


@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 0.001        # 学习率
    batch_size: int = 32                # 批次大小
    num_epochs: int = 100               # 训练轮数
    weight_decay: float = 1e-4          # 权重衰减
    lr_scheduler: str = 'cosine'        # 学习率调度器
    momentum: float = 0.9               # 动量
    gradient_clip: float = 1.0          # 梯度裁剪
    
    # 自对弈配置
    self_play_games: int = 1000         # 自对弈游戏数量
    self_play_workers: int = 4          # 自对弈工作进程数
    
    # 模型评估配置
    evaluation_games: int = 100         # 评估游戏数量
    evaluation_threshold: float = 0.55  # 新模型胜率阈值
    
    # 数据管理
    max_training_examples: int = 500000 # 最大训练样本数
    training_data_ratio: float = 0.8    # 训练数据比例
    
    # 检查点配置
    save_checkpoint_every: int = 10     # 每N轮保存检查点
    keep_checkpoint_count: int = 5      # 保留检查点数量


@dataclass
class ModelConfig:
    """神经网络模型配置"""
    input_channels: int = 14            # 输入通道数
    num_blocks: int = 20                # ResNet块数量
    hidden_channels: int = 256          # 隐藏层通道数
    value_head_hidden: int = 256        # 价值头隐藏层大小
    policy_head_hidden: int = 256       # 策略头隐藏层大小
    
    # 注意力机制配置
    use_attention: bool = True          # 是否使用注意力机制
    attention_heads: int = 8            # 注意力头数量
    attention_dim: int = 256            # 注意力维度
    
    # 正则化配置
    dropout_rate: float = 0.1           # Dropout率
    batch_norm: bool = True             # 是否使用批归一化
    
    # 激活函数
    activation: str = 'relu'            # 激活函数类型


@dataclass
class AIConfig:
    """AI引擎配置"""
    model_path: str = ""                # 模型文件路径
    search_time: float = 5.0            # 搜索时间限制
    max_simulations: int = 1000         # 最大模拟次数
    difficulty_level: int = 5           # 难度级别 (1-10)
    
    # 开局库和残局库
    use_opening_book: bool = True       # 是否使用开局库
    use_endgame_tablebase: bool = True  # 是否使用残局库
    opening_book_path: str = ""         # 开局库路径
    endgame_tablebase_path: str = ""    # 残局库路径
    
    # 分析配置
    analysis_depth: int = 12            # 分析深度
    multi_pv: int = 3                   # 多主变数量
    
    # 性能配置
    num_threads: int = 4                # 线程数
    hash_size: int = 128                # 哈希表大小(MB)
    
    # 设备配置
    device: str = 'auto'                # 计算设备 ('cpu', 'cuda', 'auto')
    use_tensorrt: bool = False          # 是否使用TensorRT优化
    use_onnx: bool = False              # 是否使用ONNX运行时


@dataclass
class SystemConfig:
    """系统配置"""
    # 日志配置
    log_level: str = 'INFO'             # 日志级别
    log_file: str = 'chess_ai.log'      # 日志文件
    log_max_size: int = 10              # 日志文件最大大小(MB)
    log_backup_count: int = 5           # 日志备份数量
    
    # 数据目录
    data_dir: str = 'data'              # 数据目录
    model_dir: str = 'models'           # 模型目录
    log_dir: str = 'logs'               # 日志目录
    temp_dir: str = 'temp'              # 临时目录
    
    # 性能监控
    enable_profiling: bool = False      # 是否启用性能分析
    memory_limit: int = 2048            # 内存限制(MB)
    
    # API服务器配置
    api_host: str = 'localhost'         # API服务器主机
    api_port: int = 8000                # API服务器端口
    api_workers: int = 1                # API工作进程数
    api_timeout: float = 30.0           # API超时时间
    
    # 安全配置
    enable_auth: bool = False           # 是否启用身份验证
    api_key: str = ""                   # API密钥
    rate_limit: int = 100               # 速率限制(请求/分钟)


@dataclass
class GameConfig:
    """游戏配置"""
    # 时间控制
    time_control: str = 'fixed'         # 时间控制类型 ('fixed', 'increment', 'tournament')
    base_time: float = 300.0            # 基础时间(秒)
    increment: float = 5.0              # 增量时间(秒)
    
    # 游戏规则
    enable_repetition_draw: bool = True # 是否启用重复局面和棋
    max_moves: int = 300                # 最大走法数
    
    # 开局设置
    random_opening: bool = False        # 是否随机开局
    opening_moves: int = 0              # 开局走法数
    
    # 调试选项
    save_game_pgn: bool = True          # 是否保存PGN格式的对局
    save_analysis: bool = False         # 是否保存分析结果
    verbose_output: bool = False        # 是否输出详细信息


# 默认配置实例
DEFAULT_MCTS_CONFIG = MCTSConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_AI_CONFIG = AIConfig()
DEFAULT_SYSTEM_CONFIG = SystemConfig()
DEFAULT_GAME_CONFIG = GameConfig()