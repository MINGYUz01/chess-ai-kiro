# -*- coding: utf-8 -*-
"""
数据处理模块

提供数据管理、验证和预处理功能，支持YOLO格式的标注数据处理。
"""

from .data_manager import DataManager, DatasetSplit, ClassStatistics, YOLOAnnotation
from .annotation_validator import AnnotationValidator, ValidationResult, ClassConsistencyResult

__all__ = [
    'DataManager',
    'DatasetSplit', 
    'ClassStatistics',
    'YOLOAnnotation',
    'AnnotationValidator',
    'ValidationResult',
    'ClassConsistencyResult'
]