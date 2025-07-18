# 中国象棋AI引擎设计文档

## 概述

中国象棋AI引擎是一个基于深度强化学习的高性能象棋对弈系统，采用AlphaZero架构结合蒙特卡洛树搜索(MCTS)算法。系统包括规则引擎、神经网络模型、搜索算法、训练框架和推理接口五个核心模块，能够提供专业级别的象棋分析和对弈能力。

## 架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    中国象棋AI引擎                                │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   规则引擎模块   │   神经网络模块   │      搜索算法模块            │
│                │                │                            │
│ • 棋局表示器     │ • 价值网络       │ • MCTS搜索器                │
│ • 走法生成器     │ • 策略网络       │ • 搜索树管理器              │
│ • 规则验证器     │ • 模型管理器     │ • 并行搜索器                │
│ • 状态评估器     │ • 推理引擎       │ • 时间管理器                │
└─────────────────┼─────────────────┼─────────────────────────────┤
│      训练框架模块                  │      推理接口模块            │
│                                  │                            │
│ • 自对弈生成器                    │ • API服务器                 │
│ • 数据管理器                      │ • 对弈接口                  │
│ • 训练调度器                      │ • 分析接口                  │
│ • 模型评估器                      │ • 配置管理器                │
└───────────────────────────────────┴─────────────────────────────┘
```

### 技术栈

- **深度学习框架**: PyTorch
- **神经网络**: ResNet + Attention机制
- **搜索算法**: 蒙特卡洛树搜索(MCTS)
- **并行计算**: multiprocessing, threading
- **数据存储**: HDF5, SQLite
- **API框架**: FastAPI
- **配置管理**: YAML, Pydantic
- **性能优化**: ONNX Runtime, TensorRT

## 组件和接口

### 1. 规则引擎模块

#### ChessBoard类
```python
class ChessBoard:
    def __init__(self, fen: str = None)
    def from_matrix(self, matrix: np.ndarray) -> 'ChessBoard'
    def to_matrix(self) -> np.ndarray
    def to_fen(self) -> str
    def make_move(self, move: Move) -> 'ChessBoard'
    def undo_move(self) -> 'ChessBoard'
    def is_legal_move(self, move: Move) -> bool
    def get_legal_moves(self) -> List[Move]
    def is_game_over(self) -> bool
    def get_winner(self) -> Optional[int]
    def is_check(self, color: int) -> bool
    def is_checkmate(self, color: int) -> bool
    def is_stalemate(self, color: int) -> bool
```

**职责**:
- 维护棋局状态和历史
- 生成所有合法走法
- 验证走法的合法性
- 检测游戏结束条件
- 支持多种棋局表示格式

#### Move类
```python
@dataclass
class Move:
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    piece: int
    captured_piece: Optional[int] = None
    is_check: bool = False
    is_checkmate: bool = False
    
    def to_chinese_notation(self) -> str
    def to_coordinate_notation(self) -> str
    def from_string(cls, move_str: str) -> 'Move'
```

#### RuleEngine类
```python
class RuleEngine:
    def __init__(self)
    def validate_board_state(self, board: ChessBoard) -> bool
    def generate_moves_for_piece(self, board: ChessBoard, pos: Tuple[int, int]) -> List[Move]
    def is_move_legal(self, board: ChessBoard, move: Move) -> bool
    def detect_repetition(self, board_history: List[ChessBoard]) -> bool
    def evaluate_endgame(self, board: ChessBoard) -> Optional[int]
```

### 2. 神经网络模块

#### ChessNet类
```python
class ChessNet(nn.Module):
    def __init__(self, input_channels: int = 14, num_blocks: int = 20)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
    def predict_value(self, board_tensor: torch.Tensor) -> float
    def predict_policy(self, board_tensor: torch.Tensor) -> np.ndarray
    def save_checkpoint(self, path: str) -> None
    def load_checkpoint(self, path: str) -> None
```

**网络架构**:
- **输入层**: 14通道的10x9特征图（棋子类型、颜色、历史位置等）
- **主干网络**: 20层ResNet块，每块包含卷积、批归一化和ReLU激活
- **注意力机制**: 自注意力层增强位置关系理解
- **价值头**: 全连接层输出局面评估值(-1到1)
- **策略头**: 卷积层输出所有可能走法的概率分布

#### ModelManager类
```python
class ModelManager:
    def __init__(self, model_dir: str)
    def save_model(self, model: ChessNet, version: str, metadata: Dict) -> None
    def load_model(self, version: str = 'latest') -> ChessNet
    def list_models(self) -> List[Dict]
    def delete_model(self, version: str) -> None
    def export_onnx(self, model: ChessNet, output_path: str) -> None
    def quantize_model(self, model: ChessNet) -> ChessNet
```

#### InferenceEngine类
```python
class InferenceEngine:
    def __init__(self, model_path: str, device: str = 'auto')
    def preprocess_board(self, board: ChessBoard) -> torch.Tensor
    def predict(self, board: ChessBoard) -> Tuple[float, np.ndarray]
    def batch_predict(self, boards: List[ChessBoard]) -> List[Tuple[float, np.ndarray]]
    def set_batch_size(self, batch_size: int) -> None
```

### 3. 搜索算法模块

#### MCTSNode类
```python
@dataclass
class MCTSNode:
    board: ChessBoard
    move: Optional[Move]
    parent: Optional['MCTSNode']
    children: Dict[Move, 'MCTSNode']
    visit_count: int = 0
    value_sum: float = 0.0
    prior_probability: float = 0.0
    
    def is_expanded(self) -> bool
    def is_terminal(self) -> bool
    def ucb_score(self, c_puct: float = 1.0) -> float
    def backup(self, value: float) -> None
```

#### MCTSSearcher类
```python
class MCTSSearcher:
    def __init__(self, model: ChessNet, config: MCTSConfig)
    def search(self, root_board: ChessBoard, num_simulations: int) -> MCTSNode
    def select_leaf(self, root: MCTSNode) -> MCTSNode
    def expand_and_evaluate(self, node: MCTSNode) -> float
    def backup_value(self, node: MCTSNode, value: float) -> None
    def get_action_probabilities(self, root: MCTSNode, temperature: float = 1.0) -> np.ndarray
    def get_best_move(self, root: MCTSNode) -> Move
```

#### ParallelSearcher类
```python
class ParallelSearcher:
    def __init__(self, model: ChessNet, num_workers: int = 4)
    def parallel_search(self, board: ChessBoard, simulations: int) -> Tuple[Move, float, np.ndarray]
    def worker_search(self, worker_id: int, shared_tree: MCTSNode) -> None
    def merge_search_results(self, results: List[MCTSNode]) -> MCTSNode
```

### 4. 训练框架模块

#### SelfPlayGenerator类
```python
class SelfPlayGenerator:
    def __init__(self, model: ChessNet, config: SelfPlayConfig)
    def generate_game(self) -> List[TrainingExample]
    def play_game(self, model1: ChessNet, model2: ChessNet) -> GameResult
    def collect_training_data(self, num_games: int) -> List[TrainingExample]
    def save_training_data(self, data: List[TrainingExample], path: str) -> None
```

#### TrainingExample类
```python
@dataclass
class TrainingExample:
    board_tensor: torch.Tensor
    policy_target: np.ndarray
    value_target: float
    game_result: int
    move_number: int
```

#### Trainer类
```python
class Trainer:
    def __init__(self, model: ChessNet, config: TrainingConfig)
    def train_epoch(self, training_data: List[TrainingExample]) -> Dict[str, float]
    def validate(self, validation_data: List[TrainingExample]) -> Dict[str, float]
    def train(self, num_epochs: int, training_data: List[TrainingExample]) -> None
    def save_checkpoint(self, epoch: int, metrics: Dict) -> None
    def load_checkpoint(self, checkpoint_path: str) -> None
```

#### ModelEvaluator类
```python
class ModelEvaluator:
    def __init__(self, config: EvaluationConfig)
    def evaluate_against_baseline(self, new_model: ChessNet, baseline_model: ChessNet) -> Dict
    def tournament_evaluation(self, models: List[ChessNet]) -> Dict
    def calculate_elo_rating(self, game_results: List[GameResult]) -> Dict[str, float]
    def benchmark_performance(self, model: ChessNet) -> Dict
```

### 5. 推理接口模块

#### ChessAI类
```python
class ChessAI:
    def __init__(self, model_path: str, config: AIConfig)
    def analyze_position(self, board: ChessBoard, depth: int = None) -> AnalysisResult
    def get_best_move(self, board: ChessBoard, time_limit: float = 5.0) -> Move
    def get_top_moves(self, board: ChessBoard, num_moves: int = 5) -> List[Tuple[Move, float]]
    def evaluate_position(self, board: ChessBoard) -> float
    def calculate_win_probability(self, board: ChessBoard) -> Tuple[float, float]
    def set_difficulty_level(self, level: int) -> None
```

#### GameInterface类
```python
class GameInterface:
    def __init__(self, ai: ChessAI)
    def start_new_game(self, ai_color: int = None) -> str
    def make_move(self, game_id: str, move: Move) -> GameState
    def get_ai_move(self, game_id: str) -> Move
    def get_game_state(self, game_id: str) -> GameState
    def end_game(self, game_id: str) -> GameResult
    def get_game_history(self, game_id: str) -> List[Move]
```

#### APIServer类
```python
class APIServer:
    def __init__(self, ai: ChessAI, host: str = "localhost", port: int = 8000)
    async def analyze_position_endpoint(self, request: AnalysisRequest) -> AnalysisResponse
    async def get_best_move_endpoint(self, request: MoveRequest) -> MoveResponse
    async def start_game_endpoint(self, request: GameStartRequest) -> GameStartResponse
    async def make_move_endpoint(self, request: GameMoveRequest) -> GameMoveResponse
    def start_server(self) -> None
```

## 数据模型

### 配置数据结构
```python
@dataclass
class MCTSConfig:
    num_simulations: int = 800
    c_puct: float = 1.0
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-4
    lr_scheduler: str = 'cosine'

@dataclass
class AIConfig:
    model_path: str
    search_time: float = 5.0
    max_simulations: int = 1000
    difficulty_level: int = 5
    use_opening_book: bool = True
    use_endgame_tablebase: bool = True
```

### 分析结果数据结构
```python
@dataclass
class AnalysisResult:
    best_move: Move
    evaluation: float
    win_probability: Tuple[float, float]  # (red_win_prob, black_win_prob)
    principal_variation: List[Move]
    top_moves: List[Tuple[Move, float]]
    search_depth: int
    nodes_searched: int
    time_used: float
```

### 游戏状态数据结构
```python
@dataclass
class GameState:
    board: ChessBoard
    current_player: int
    move_history: List[Move]
    game_status: str  # 'ongoing', 'red_wins', 'black_wins', 'draw'
    last_move: Optional[Move]
    check_status: bool
    legal_moves: List[Move]
```

## 错误处理

### 异常类型定义
```python
class ChessAIError(Exception):
    """象棋AI引擎基础异常"""
    pass

class InvalidMoveError(ChessAIError):
    """非法走法异常"""
    pass

class ModelLoadError(ChessAIError):
    """模型加载异常"""
    pass

class SearchTimeoutError(ChessAIError):
    """搜索超时异常"""
    pass

class TrainingError(ChessAIError):
    """训练过程异常"""
    pass
```

### 错误处理策略
1. **非法走法**: 返回错误信息，要求重新输入合法走法
2. **模型加载失败**: 尝试加载备用模型或使用规则引擎
3. **搜索超时**: 返回当前最佳走法，记录超时警告
4. **内存不足**: 减少搜索深度，清理缓存
5. **训练中断**: 保存当前状态，支持断点续训

## 测试策略

### 单元测试
- 规则引擎的走法生成和验证测试
- 神经网络的前向传播和梯度测试
- MCTS搜索算法的正确性测试
- 数据转换和序列化测试

### 集成测试
- 完整对弈流程测试
- 训练和推理流程测试
- API接口的功能测试
- 多线程并发安全测试

### 性能测试
- 搜索速度基准测试（目标：1000次模拟/秒）
- 内存使用量监控（目标：<2GB）
- 模型推理延迟测试（目标：<100ms）
- 并发处理能力测试

### 棋力测试
- 与开源象棋引擎对弈测试
- 标准测试局面的分析准确性
- 不同时间控制下的表现测试
- ELO等级分评估（目标：>2000）

## 部署和优化

### 模型优化
1. **量化**: INT8量化减少模型大小50%
2. **剪枝**: 移除不重要的连接，保持95%准确率
3. **蒸馏**: 训练轻量级学生模型
4. **ONNX导出**: 支持跨平台高性能推理

### 搜索优化
1. **并行搜索**: 多线程MCTS提升搜索效率
2. **缓存机制**: 缓存常见局面的评估结果
3. **剪枝策略**: 实现Alpha-Beta剪枝优化
4. **时间管理**: 动态调整搜索时间分配

### 部署配置
```python
# deployment_config.py
DEPLOYMENT_CONFIG = {
    'model': {
        'format': 'onnx',  # pytorch, onnx, tensorrt
        'precision': 'fp16',  # fp32, fp16, int8
        'batch_size': 1,
    },
    'search': {
        'max_simulations': 1000,
        'time_limit': 5.0,
        'num_threads': 4,
    },
    'hardware': {
        'device': 'auto',  # cpu, cuda, auto
        'memory_limit': '2GB',
        'use_tensorrt': False,
    }
}
```

### 性能监控
```python
class PerformanceMonitor:
    def __init__(self)
    def log_search_metrics(self, simulations: int, time_used: float, nodes_per_second: float) -> None
    def log_model_metrics(self, inference_time: float, memory_usage: float) -> None
    def generate_performance_report(self) -> Dict
    def alert_on_performance_degradation(self, threshold: float) -> None
```