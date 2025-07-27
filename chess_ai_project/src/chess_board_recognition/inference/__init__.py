"""
推理引擎模块

提供棋局检测和识别功能，包括：
- 模型加载和推理
- 图像预处理和后处理
- 结果转换和验证
- 棋局状态输出
"""

# 导入检测器相关类
from .chessboard_detector import (
    ChessboardDetector, 
    DetectionBox, 
    ImagePreprocessor
)

# 导入映射器相关类
from .board_mapper import BoardMapper

# 导入结果处理器相关类
from .result_processor import (
    ResultProcessor,
    QualityMetrics,
    StateValidator,
    ConfidenceCalculator
)

__all__ = [
    # 核心检测器
    'ChessboardDetector',
    'DetectionBox',
    'ImagePreprocessor',
    
    # 棋盘映射
    'BoardMapper',
    
    # 结果处理
    'ResultProcessor',
    'QualityMetrics',
    'StateValidator',
    'ConfidenceCalculator'
]