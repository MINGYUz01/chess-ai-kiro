# 棋盘识别系统设计文档

## 概述

棋盘识别系统采用模块化架构，基于YOLO11深度学习框架实现中国象棋棋局的实时识别。系统分为数据收集模块、数据处理模块、模型训练模块、推理引擎模块和系统管理模块五个核心组件。

## 架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    棋盘识别系统                              │
├─────────────────┬─────────────────┬─────────────────────────┤
│   数据收集模块   │   数据处理模块   │      模型训练模块        │
│                │                │                        │
│ • 屏幕截图工具   │ • 数据预处理     │ • YOLO11训练器          │
│ • 区域选择器     │ • 标注管理器     │ • 超参数优化器          │
│ • 自动/手动截图  │ • 数据增强器     │ • 模型评估器            │
└─────────────────┼─────────────────┼─────────────────────────┤
│      推理引擎模块                  │      系统管理模块        │
│                                  │                        │
│ • 模型加载器                      │ • 配置管理器            │
│ • 图像预处理器                    │ • 日志系统              │
│ • 后处理器                        │ • 性能监控器            │
│ • 结果输出器                      │ • 异常处理器            │
└───────────────────────────────────┴─────────────────────────┘
```

### 技术栈

- **深度学习框架**: Ultralytics YOLO11
- **图像处理**: OpenCV, PIL
- **GUI框架**: Tkinter (截图工具界面)
- **数据处理**: NumPy, Pandas
- **配置管理**: YAML, JSON
- **日志系统**: Python logging
- **模型优化**: ONNX Runtime

## 组件和接口

### 1. 数据收集模块

#### ScreenCapture类
```python
class ScreenCapture:
    def __init__(self, config_path: str)
    def select_region(self) -> Tuple[int, int, int, int]
    def start_auto_capture(self, interval: int) -> None
    def manual_capture(self) -> str
    def stop_capture(self) -> None
    def get_capture_stats(self) -> Dict
```

**职责**:
- 提供图形界面选择截图区域
- 支持自动和手动截图模式
- 管理截图文件的存储和命名
- 监控存储空间和截图统计

#### RegionSelector类
```python
class RegionSelector:
    def __init__(self)
    def show_selection_overlay(self) -> None
    def get_selected_region(self) -> Tuple[int, int, int, int]
    def save_region_config(self, region: Tuple) -> None
    def load_region_config(self) -> Tuple[int, int, int, int]
```

### 2. 数据处理模块

#### DataManager类
```python
class DataManager:
    def __init__(self, data_dir: str)
    def create_labelimg_structure(self) -> None
    def validate_annotations(self, annotation_dir: str) -> List[str]
    def split_dataset(self, train_ratio: float = 0.8) -> Dict
    def augment_data(self, augmentation_config: Dict) -> None
    def get_class_statistics(self) -> Dict
```

**职责**:
- 创建和管理训练数据目录结构
- 验证标注文件的格式和完整性
- 自动划分训练集、验证集和测试集
- 提供数据增强功能
- 统计各类别的数据分布

#### AnnotationValidator类
```python
class AnnotationValidator:
    def __init__(self, class_names: List[str])
    def validate_yolo_format(self, annotation_file: str) -> bool
    def check_class_consistency(self, annotation_dir: str) -> List[str]
    def generate_validation_report(self) -> Dict
```

### 3. 模型训练模块

#### YOLO11Trainer类
```python
class YOLO11Trainer:
    def __init__(self, config_path: str)
    def prepare_training_data(self, data_yaml_path: str) -> None
    def train(self, epochs: int, batch_size: int, **kwargs) -> None
    def validate(self, model_path: str) -> Dict
    def export_model(self, format: str = 'onnx') -> str
    def get_training_metrics(self) -> Dict
```

**职责**:
- 配置YOLO11训练参数
- 执行模型训练和验证
- 监控训练进度和指标
- 导出训练好的模型
- 生成训练报告

#### HyperparameterOptimizer类
```python
class HyperparameterOptimizer:
    def __init__(self, search_space: Dict)
    def optimize(self, objective_function: Callable) -> Dict
    def get_best_params(self) -> Dict
```

### 4. 推理引擎模块

#### ChessboardDetector类
```python
class ChessboardDetector:
    def __init__(self, model_path: str)
    def load_model(self, model_path: str) -> None
    def preprocess_image(self, image: np.ndarray) -> np.ndarray
    def detect(self, image: np.ndarray) -> Dict
    def postprocess_results(self, raw_results: Dict) -> Dict
    def convert_to_matrix(self, detections: Dict) -> np.ndarray
```

**职责**:
- 加载训练好的YOLO11模型
- 预处理输入图像
- 执行目标检测
- 后处理检测结果
- 转换为标准化的棋局矩阵

#### ResultProcessor类
```python
class ResultProcessor:
    def __init__(self, board_size: Tuple[int, int] = (10, 9))
    def map_detections_to_board(self, detections: List) -> np.ndarray
    def validate_board_state(self, board_matrix: np.ndarray) -> bool
    def get_selected_piece_info(self, detections: List) -> Dict
    def calculate_confidence_score(self, detections: List) -> float
```

### 5. 系统管理模块

#### ConfigManager类
```python
class ConfigManager:
    def __init__(self, config_file: str)
    def load_config(self) -> Dict
    def save_config(self, config: Dict) -> None
    def validate_config(self, config: Dict) -> bool
    def get_default_config(self) -> Dict
```

#### PerformanceMonitor类
```python
class PerformanceMonitor:
    def __init__(self)
    def start_monitoring(self) -> None
    def log_inference_time(self, time_ms: float) -> None
    def log_accuracy_metrics(self, metrics: Dict) -> None
    def generate_performance_report(self) -> Dict
```

## 数据模型

### 检测结果数据结构
```python
@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]  # x, y

@dataclass
class BoardState:
    matrix: np.ndarray  # 10x9 matrix
    selected_piece: Optional[Tuple[int, int]]
    confidence: float
    timestamp: datetime
    detections: List[Detection]
```

### 配置文件结构
```yaml
# config.yaml
capture:
  region: [x, y, width, height]
  auto_interval: 2  # seconds
  save_path: "./data/captures"
  
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  image_size: 640
  
model:
  path: "./models/best.pt"
  confidence_threshold: 0.5
  nms_threshold: 0.4
  
classes:
  - board
  - grid_lines
  - red_king
  - red_advisor
  # ... 其他类别
```

## 错误处理

### 异常类型定义
```python
class ChessboardRecognitionError(Exception):
    """棋盘识别系统基础异常"""
    pass

class ModelLoadError(ChessboardRecognitionError):
    """模型加载异常"""
    pass

class InferenceError(ChessboardRecognitionError):
    """推理过程异常"""
    pass

class DataValidationError(ChessboardRecognitionError):
    """数据验证异常"""
    pass
```

### 错误处理策略
1. **模型加载失败**: 尝试加载备用模型或提示用户重新训练
2. **推理异常**: 记录错误信息，返回空结果并继续处理
3. **数据格式错误**: 提供详细的错误报告和修复建议
4. **资源不足**: 自动降低处理质量或提示用户释放资源

## 测试策略

### 单元测试
- 每个类和方法的独立功能测试
- 数据处理和转换的正确性测试
- 配置管理和异常处理测试

### 集成测试
- 端到端的数据流测试
- 模型训练和推理流程测试
- 不同模块间的接口测试

### 性能测试
- 推理速度基准测试
- 内存使用量监控
- 并发处理能力测试

### 准确性测试
- 使用标准测试集验证识别准确率
- 不同光照和角度条件下的鲁棒性测试
- 边界情况和异常棋局的处理测试

## 部署和优化

### 模型优化
1. **量化**: 使用INT8量化减少模型大小
2. **剪枝**: 移除不重要的网络连接
3. **蒸馏**: 使用知识蒸馏技术压缩模型
4. **ONNX导出**: 支持跨平台部署

### 性能优化
1. **GPU加速**: 支持CUDA和OpenCL加速
2. **批处理**: 支持批量图像处理
3. **内存管理**: 优化内存使用和垃圾回收
4. **缓存机制**: 缓存常用的处理结果

### 部署配置
```python
# deployment_config.py
DEPLOYMENT_CONFIG = {
    'model_format': 'onnx',  # pt, onnx, tensorrt
    'device': 'auto',  # cpu, cuda, auto
    'batch_size': 1,
    'optimization_level': 'balanced',  # speed, balanced, accuracy
    'memory_limit': '2GB'
}
```