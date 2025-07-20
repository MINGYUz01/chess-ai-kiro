# -*- coding: utf-8 -*-
"""
数据处理模块

提供数据管理、验证和预处理功能，支持YOLO格式的标注数据处理。
"""

from .data_manager import DataManager, DatasetSplit, ClassStatistics, YOLOAnnotation
from .annotation_validator import AnnotationValidator, ValidationResult, ClassConsistencyResult
from .quality_controller import QualityController, QualityMetrics, QualityIssue, QualityIssueType, LabelImgConfigGenerator
from .data_augmentation import DataAugmentor, AugmentationConfig, YOLOBbox, create_default_augmentation_config

__all__ = [
    # 数据管理
    'DataManager',
    'DatasetSplit', 
    'ClassStatistics',
    'YOLOAnnotation',
    
    # 标注验证
    'AnnotationValidator',
    'ValidationResult',
    'ClassConsistencyResult',
    
    # 质量控制
    'QualityController',
    'QualityMetrics',
    'QualityIssue',
    'QualityIssueType',
    'LabelImgConfigGenerator',
    
    # 数据增强
    'DataAugmentor',
    'AugmentationConfig',
    'YOLOBbox',
    'create_default_augmentation_config'
]