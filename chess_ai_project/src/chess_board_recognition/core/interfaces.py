"""
棋盘识别系统核心接口定义

定义了系统中所有核心组件的接口和数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 异常类定义
# ============================================================================

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


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class Detection:
    """检测结果数据结构"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]  # x, y
    
    def __post_init__(self):
        """验证数据有效性"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"置信度必须在0-1之间，当前值: {self.confidence}")
        if len(self.bbox) != 4:
            raise ValueError(f"边界框必须包含4个坐标值，当前: {self.bbox}")
        if len(self.center) != 2:
            raise ValueError(f"中心点必须包含2个坐标值，当前: {self.center}")


@dataclass
class BoardState:
    """棋局状态数据结构"""
    matrix: np.ndarray  # 10x9 棋局矩阵
    selected_piece: Optional[Tuple[int, int]]  # 选中棋子位置 (row, col)
    confidence: float  # 整体置信度
    timestamp: datetime  # 识别时间戳
    detections: List[Detection]  # 原始检测结果
    
    def __post_init__(self):
        """验证棋局状态数据"""
        if self.matrix.shape != (10, 9):
            raise ValueError(f"棋局矩阵必须是10x9，当前形状: {self.matrix.shape}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"置信度必须在0-1之间，当前值: {self.confidence}")
        if self.selected_piece is not None:
            row, col = self.selected_piece
            if not (0 <= row < 10 and 0 <= col < 9):
                raise ValueError(f"选中棋子位置超出棋盘范围: {self.selected_piece}")


# ============================================================================
# 核心接口定义
# ============================================================================

class ChessboardDetector(ABC):
    """棋盘检测器抽象基类"""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
            
        Raises:
            ModelLoadError: 模型加载失败时抛出
        """
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理输入图像
        
        Args:
            image: 输入图像数组
            
        Returns:
            预处理后的图像数组
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Dict:
        """
        执行目标检测
        
        Args:
            image: 输入图像数组
            
        Returns:
            检测结果字典
            
        Raises:
            InferenceError: 推理过程出错时抛出
        """
        pass
    
    @abstractmethod
    def postprocess_results(self, raw_results: Dict) -> Dict:
        """
        后处理检测结果
        
        Args:
            raw_results: 原始检测结果
            
        Returns:
            后处理后的结果字典
        """
        pass
    
    @abstractmethod
    def convert_to_matrix(self, detections: Dict) -> BoardState:
        """
        将检测结果转换为标准化的棋局状态
        
        Args:
            detections: 检测结果字典
            
        Returns:
            棋局状态对象
        """
        pass


class DataManager(ABC):
    """数据管理器抽象基类"""
    
    @abstractmethod
    def create_labelimg_structure(self) -> None:
        """创建labelImg兼容的目录结构"""
        pass
    
    @abstractmethod
    def validate_annotations(self, annotation_dir: str) -> List[str]:
        """
        验证标注文件
        
        Args:
            annotation_dir: 标注文件目录
            
        Returns:
            验证错误列表
        """
        pass
    
    @abstractmethod
    def split_dataset(self, train_ratio: float = 0.8) -> Dict:
        """
        划分数据集
        
        Args:
            train_ratio: 训练集比例
            
        Returns:
            数据集划分结果
        """
        pass
    
    @abstractmethod
    def get_class_statistics(self) -> Dict:
        """
        获取类别统计信息
        
        Returns:
            类别统计字典
        """
        pass


class ModelTrainer(ABC):
    """模型训练器抽象基类"""
    
    @abstractmethod
    def prepare_training_data(self, data_yaml_path: str) -> None:
        """
        准备训练数据
        
        Args:
            data_yaml_path: 数据配置文件路径
        """
        pass
    
    @abstractmethod
    def train(self, epochs: int, batch_size: int, **kwargs) -> None:
        """
        执行模型训练
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            **kwargs: 其他训练参数
        """
        pass
    
    @abstractmethod
    def validate(self, model_path: str) -> Dict:
        """
        验证模型性能
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            验证结果字典
        """
        pass
    
    @abstractmethod
    def export_model(self, format: str = 'onnx') -> str:
        """
        导出模型
        
        Args:
            format: 导出格式
            
        Returns:
            导出文件路径
        """
        pass


class ScreenCapture(ABC):
    """屏幕截图抽象基类"""
    
    @abstractmethod
    def select_region(self) -> Tuple[int, int, int, int]:
        """
        选择截图区域
        
        Returns:
            区域坐标 (x, y, width, height)
        """
        pass
    
    @abstractmethod
    def start_auto_capture(self, interval: int) -> None:
        """
        开始自动截图
        
        Args:
            interval: 截图间隔（秒）
        """
        pass
    
    @abstractmethod
    def manual_capture(self) -> str:
        """
        手动截图
        
        Returns:
            截图文件路径
        """
        pass
    
    @abstractmethod
    def stop_capture(self) -> None:
        """停止截图"""
        pass
    
    @abstractmethod
    def get_capture_stats(self) -> Dict:
        """
        获取截图统计信息
        
        Returns:
            统计信息字典
        """
        pass


class PerformanceMonitor(ABC):
    """性能监控器抽象基类"""
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """开始性能监控"""
        pass
    
    @abstractmethod
    def log_inference_time(self, time_ms: float) -> None:
        """
        记录推理时间
        
        Args:
            time_ms: 推理时间（毫秒）
        """
        pass
    
    @abstractmethod
    def log_accuracy_metrics(self, metrics: Dict) -> None:
        """
        记录准确率指标
        
        Args:
            metrics: 指标字典
        """
        pass
    
    @abstractmethod
    def generate_performance_report(self) -> Dict:
        """
        生成性能报告
        
        Returns:
            性能报告字典
        """
        pass


# ============================================================================
# 常量定义
# ============================================================================

# 棋子类别定义
CHESS_CLASSES = {
    0: "board",           # 棋盘边界
    1: "grid_lines",      # 网格线
    2: "red_king",        # 红帅
    3: "red_advisor",     # 红仕
    4: "red_bishop",      # 红相
    5: "red_knight",      # 红马
    6: "red_rook",        # 红车
    7: "red_cannon",      # 红炮
    8: "red_pawn",        # 红兵
    9: "black_king",      # 黑将
    10: "black_advisor",  # 黑士
    11: "black_bishop",   # 黑象
    12: "black_knight",   # 黑马
    13: "black_rook",     # 黑车
    14: "black_cannon",   # 黑炮
    15: "black_pawn",     # 黑卒
    16: "selected_piece", # 选中状态
}

# 棋盘尺寸
BOARD_SIZE = (10, 9)  # 行数, 列数

# 默认配置
DEFAULT_CONFIG = {
    "model": {
        "path": "./models/best.pt",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
        "device": "auto",  # auto, cpu, cuda
    },
    "capture": {
        "region": [0, 0, 800, 600],  # x, y, width, height
        "auto_interval": 2,  # 秒
        "save_path": "./data/captures",
        "format": "jpg",
    },
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.001,
        "image_size": 640,
        "device": "auto",
    },
    "logging": {
        "level": "INFO",
        "file": "./logs/chess_board_recognition.log",
        "max_size": "10MB",
        "backup_count": 5,
    }
}