"""
数据处理模块

提供数据管理、标注验证和数据增强功能，包括：
- 数据集管理和划分
- 标注文件验证
- 数据增强
- 类别统计
"""

from .data_manager import DataManagerImpl
from .annotation_validator import AnnotationValidator

__all__ = [
    "DataManagerImpl", 
    "AnnotationValidator",
]