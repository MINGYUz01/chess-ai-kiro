# -*- coding: utf-8 -*-
"""
质量控制器模块

提供标注质量检查、质量报告生成和labelImg配置文件生成功能。
"""

import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """标注质量指标"""
    # 基本统计
    total_annotations: int = 0                # 总标注数量
    valid_annotations: int = 0                # 有效标注数量
    invalid_annotations: int = 0              # 无效标注数量
    validation_rate: float = 1.0              # 有效标注比例
    
    # 类别分布
    class_distribution: Dict[int, int] = field(default_factory=dict)  # 各类别标注数量
    class_balance_score: float = 1.0          # 类别平衡得分 (0-1)
    
    # 标注密度
    annotation_density: float = 0.0           # 每张图像的平均标注数
    
    # 边界框特征
    avg_bbox_size: float = 0.0                # 平均边界框大小（相对于图像）
    bbox_size_variance: float = 0.0           # 边界框大小方差
    
    # 重复和重叠
    duplicate_rate: float = 0.0               # 重复标注率
    overlap_rate: float = 0.0                 # 重叠标注率
    
    # 总体质量
    overall_quality_score: float = 100.0      # 总体质量得分 (0-100)


class QualityIssueType(str, Enum):
    """质量问题类型"""
    LOW_VALIDATION_RATE = "low_validation_rate"
    CLASS_IMBALANCE = "class_imbalance"
    LOW_ANNOTATION_DENSITY = "low_annotation_density"
    SMALL_BBOX_SIZE = "small_bbox_size"
    HIGH_BBOX_VARIANCE = "high_bbox_variance"
    HIGH_DUPLICATE_RATE = "high_duplicate_rate"
    HIGH_OVERLAP_RATE = "high_overlap_rate"
    MISSING_CLASSES = "missing_classes"
    INCONSISTENT_LABELING = "inconsistent_labeling"


@dataclass
class QualityIssue:
    """质量问题"""
    issue_type: str                           # 问题类型
    severity: str                             # 严重程度 (low, medium, high)
    description: str                          # 问题描述
    affected_files: List[str] = field(default_factory=list)  # 受影响的文件
    recommendation: str = ""                  # 改进建议


class LabelImgConfigGenerator:
    """
    LabelImg配置生成器
    
    为LabelImg标注工具生成预定义类别和配置文件。
    """
    
    def __init__(self, class_names: List[str]):
        """
        初始化LabelImg配置生成器
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
        logger.info(f"初始化LabelImg配置生成器，类别数量: {len(class_names)}")
    
    def generate_predefined_classes(self, output_file: Union[str, Path]) -> None:
        """
        生成预定义类别文件
        
        Args:
            output_file: 输出文件路径
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        logger.info(f"已生成预定义类别文件: {output_file}")
    
    def generate_config_file(self, output_dir: Union[str, Path], 
                           image_dir: Union[str, Path], 
                           label_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        生成LabelImg配置文件
        
        Args:
            output_dir: 输出目录
            image_dir: 图像目录
            label_dir: 标注目录
            
        Returns:
            配置信息字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成预定义类别文件
        predefined_classes_file = output_dir / "predefined_classes.txt"
        self.generate_predefined_classes(predefined_classes_file)
        
        # 创建配置字典
        config = {
            "image_dir": str(Path(image_dir).absolute()),
            "label_dir": str(Path(label_dir).absolute()),
            "classes": self.class_names,
            "class_count": len(self.class_names),
            "save_format": "YOLO",
            "last_open_dir": str(Path(image_dir).absolute()),
            "auto_save": True,
            "single_class_mode": False,
            "display_label_popup": True,
            "keep_prev": False,
            "keep_prev_scale": False,
            "keep_prev_brightness": True,
            "store_data": True
        }
        
        # 保存配置文件
        config_file = output_dir / "labelimg_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已生成LabelImg配置文件: {config_file}")
        return config


class QualityController:
    """
    质量控制器
    
    提供标注质量检查、质量报告生成和质量问题识别功能。
    """
    
    def __init__(self, class_names: List[str], quality_thresholds: Optional[Dict[str, float]] = None):
        """
        初始化质量控制器
        
        Args:
            class_names: 类别名称列表
            quality_thresholds: 质量阈值字典，如果为None则使用默认阈值
        """
        self.class_names = class_names
        self.class_ids = set(range(len(class_names)))
        
        # 设置默认质量阈值
        self.quality_thresholds = {
            'min_validation_rate': 0.95,      # 最低有效标注率
            'min_class_balance': 0.3,         # 最低类别平衡得分
            'min_annotation_density': 1.0,    # 最低标注密度
            'min_bbox_size': 0.01,            # 最小边界框大小（相对于图像面积）
            'max_duplicate_rate': 0.05,       # 最大重复标注率
            'max_overlap_rate': 0.7,          # 最大重叠标注率
            'min_quality_score': 70.0         # 最低质量得分
        }
        
        # 更新用户提供的阈值
        if quality_thresholds:
            self.quality_thresholds.update(quality_thresholds)
        
        logger.info(f"初始化质量控制器，类别数量: {len(class_names)}")
    
    def check_annotation_quality(self, annotation_dir: Union[str, Path]) -> QualityMetrics:
        """
        检查标注质量
        
        Args:
            annotation_dir: 标注目录路径
            
        Returns:
            质量指标对象
        """
        annotation_dir = Path(annotation_dir)
        
        if not annotation_dir.exists():
            raise ValueError(f"标注目录不存在: {annotation_dir}")
        
        # 查找所有标注文件
        annotation_files = list(annotation_dir.glob("*.txt"))
        
        if not annotation_files:
            raise ValueError(f"在目录 {annotation_dir} 中未找到标注文件")
        
        logger.info(f"开始检查标注质量，文件数量: {len(annotation_files)}")
        
        # 初始化指标
        metrics = QualityMetrics()
        
        # 类别分布计数器
        class_counter = Counter()
        
        # 边界框大小列表
        bbox_sizes = []
        
        # 重复标注检测
        duplicate_count = 0
        
        # 重叠标注检测
        overlap_count = 0
        total_bbox_pairs = 0
        
        # 处理每个标注文件
        for annotation_file in annotation_files:
            try:
                # 读取标注
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                # 当前文件的边界框列表
                file_bboxes = []
                
                # 处理每行标注
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        metrics.invalid_annotations += 1
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 检查类别ID是否有效
                        if class_id not in self.class_ids:
                            metrics.invalid_annotations += 1
                            continue
                        
                        # 检查坐标是否在有效范围内
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                               0 < width <= 1 and 0 < height <= 1):
                            metrics.invalid_annotations += 1
                            continue
                        
                        # 有效标注
                        metrics.valid_annotations += 1
                        class_counter[class_id] += 1
                        
                        # 计算边界框大小（相对于图像面积）
                        bbox_size = width * height
                        bbox_sizes.append(bbox_size)
                        
                        # 添加到当前文件的边界框列表
                        file_bboxes.append((class_id, x_center, y_center, width, height))
                        
                    except (ValueError, IndexError):
                        metrics.invalid_annotations += 1
                
                # 检查重复标注
                seen_bboxes = set()
                for bbox in file_bboxes:
                    # 使用四舍五入减少浮点误差影响
                    rounded_bbox = (bbox[0], round(bbox[1], 4), round(bbox[2], 4), 
                                   round(bbox[3], 4), round(bbox[4], 4))
                    if rounded_bbox in seen_bboxes:
                        duplicate_count += 1
                    else:
                        seen_bboxes.add(rounded_bbox)
                
                # 检查重叠标注
                for i in range(len(file_bboxes)):
                    for j in range(i + 1, len(file_bboxes)):
                        total_bbox_pairs += 1
                        
                        # 计算IoU
                        iou = self._calculate_iou(file_bboxes[i], file_bboxes[j])
                        if iou > 0.5:  # IoU阈值
                            overlap_count += 1
                
            except Exception as e:
                logger.warning(f"处理标注文件失败 {annotation_file}: {e}")
        
        # 计算总标注数
        metrics.total_annotations = metrics.valid_annotations + metrics.invalid_annotations
        
        # 计算有效标注率
        if metrics.total_annotations > 0:
            metrics.validation_rate = metrics.valid_annotations / metrics.total_annotations
        
        # 设置类别分布
        metrics.class_distribution = dict(class_counter)
        
        # 计算类别平衡得分
        if class_counter:
            max_count = max(class_counter.values())
            min_count = min(class_counter.values())
            if max_count > 0:
                metrics.class_balance_score = min_count / max_count
        
        # 计算标注密度
        if annotation_files:
            metrics.annotation_density = metrics.valid_annotations / len(annotation_files)
        
        # 计算边界框大小统计
        if bbox_sizes:
            metrics.avg_bbox_size = sum(bbox_sizes) / len(bbox_sizes)
            if len(bbox_sizes) > 1:
                variance = sum((size - metrics.avg_bbox_size) ** 2 for size in bbox_sizes) / len(bbox_sizes)
                metrics.bbox_size_variance = variance
        
        # 计算重复率
        if metrics.valid_annotations > 0:
            metrics.duplicate_rate = duplicate_count / metrics.valid_annotations
        
        # 计算重叠率
        if total_bbox_pairs > 0:
            metrics.overlap_rate = overlap_count / total_bbox_pairs
        
        # 计算总体质量得分
        metrics.overall_quality_score = self._calculate_quality_score(metrics)
        
        logger.info(f"标注质量检查完成，总体质量得分: {metrics.overall_quality_score:.2f}")
        
        return metrics
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            bbox1: 第一个边界框 (class_id, x_center, y_center, width, height)
            bbox2: 第二个边界框 (class_id, x_center, y_center, width, height)
            
        Returns:
            IoU值
        """
        # 提取坐标
        _, x1, y1, w1, h1 = bbox1
        _, x2, y2, w2, h2 = bbox2
        
        # 计算边界框的左上角和右下角坐标
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2
        
        # 计算交集区域
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        # 检查是否有交集
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        # 计算交集面积
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算两个边界框的面积
        area1 = w1 * h1
        area2 = w2 * h2
        
        # 计算并集面积
        union_area = area1 + area2 - inter_area
        
        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _calculate_quality_score(self, metrics: QualityMetrics) -> float:
        """
        计算总体质量得分
        
        Args:
            metrics: 质量指标对象
            
        Returns:
            质量得分 (0-100)
        """
        # 各指标的权重
        weights = {
            'validation_rate': 0.25,
            'class_balance': 0.15,
            'annotation_density': 0.15,
            'bbox_size': 0.10,
            'duplicate_rate': 0.15,
            'overlap_rate': 0.10,
            'class_coverage': 0.10
        }
        
        # 计算各指标的得分
        scores = {}
        
        # 有效标注率得分
        scores['validation_rate'] = min(metrics.validation_rate / self.quality_thresholds['min_validation_rate'], 1.0)
        
        # 类别平衡得分
        scores['class_balance'] = min(metrics.class_balance_score / self.quality_thresholds['min_class_balance'], 1.0)
        
        # 标注密度得分
        scores['annotation_density'] = min(metrics.annotation_density / self.quality_thresholds['min_annotation_density'], 1.0)
        
        # 边界框大小得分
        scores['bbox_size'] = min(metrics.avg_bbox_size / self.quality_thresholds['min_bbox_size'], 1.0)
        
        # 重复率得分（越低越好）
        if metrics.duplicate_rate <= self.quality_thresholds['max_duplicate_rate']:
            scores['duplicate_rate'] = 1.0
        else:
            scores['duplicate_rate'] = max(0.0, 1.0 - (metrics.duplicate_rate - self.quality_thresholds['max_duplicate_rate']) / 0.2)
        
        # 重叠率得分（越低越好）
        if metrics.overlap_rate <= self.quality_thresholds['max_overlap_rate']:
            scores['overlap_rate'] = 1.0
        else:
            scores['overlap_rate'] = max(0.0, 1.0 - (metrics.overlap_rate - self.quality_thresholds['max_overlap_rate']) / 0.3)
        
        # 类别覆盖率得分
        class_coverage = len(metrics.class_distribution) / len(self.class_names) if self.class_names else 1.0
        scores['class_coverage'] = class_coverage
        
        # 计算加权总分
        total_score = 0.0
        for key, weight in weights.items():
            total_score += scores.get(key, 0.0) * weight
        
        # 转换为0-100分
        return total_score * 100.0
    
    def identify_quality_issues(self, metrics: QualityMetrics) -> List[QualityIssue]:
        """
        识别质量问题
        
        Args:
            metrics: 质量指标对象
            
        Returns:
            质量问题列表
        """
        issues = []
        
        # 检查有效标注率
        if metrics.validation_rate < self.quality_thresholds['min_validation_rate']:
            severity = "high" if metrics.validation_rate < 0.8 else "medium"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.LOW_VALIDATION_RATE,
                severity=severity,
                description=f"有效标注率过低 ({metrics.validation_rate:.2f})",
                recommendation="检查标注格式和坐标范围，确保所有标注都符合YOLO格式要求"
            ))
        
        # 检查类别平衡
        if metrics.class_balance_score < self.quality_thresholds['min_class_balance']:
            severity = "high" if metrics.class_balance_score < 0.1 else "medium"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.CLASS_IMBALANCE,
                severity=severity,
                description=f"类别分布不平衡 (平衡得分: {metrics.class_balance_score:.2f})",
                recommendation="增加少数类别的样本数量，或对少数类别进行数据增强"
            ))
        
        # 检查标注密度
        if metrics.annotation_density < self.quality_thresholds['min_annotation_density']:
            severity = "medium" if metrics.annotation_density < 0.5 else "low"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.LOW_ANNOTATION_DENSITY,
                severity=severity,
                description=f"标注密度过低 (每图平均 {metrics.annotation_density:.2f} 个标注)",
                recommendation="增加每张图像的标注数量，或确保所有目标都被标注"
            ))
        
        # 检查边界框大小
        if metrics.avg_bbox_size < self.quality_thresholds['min_bbox_size']:
            severity = "medium"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.SMALL_BBOX_SIZE,
                severity=severity,
                description=f"边界框平均大小过小 ({metrics.avg_bbox_size:.4f})",
                recommendation="考虑使用更高分辨率的图像，或确保目标在图像中占据足够大的区域"
            ))
        
        # 检查边界框大小方差
        if metrics.bbox_size_variance > 0.05:
            severity = "low"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.HIGH_BBOX_VARIANCE,
                severity=severity,
                description=f"边界框大小变化过大 (方差: {metrics.bbox_size_variance:.4f})",
                recommendation="尝试保持目标在图像中的大小相对一致，或考虑按大小分类训练"
            ))
        
        # 检查重复标注
        if metrics.duplicate_rate > self.quality_thresholds['max_duplicate_rate']:
            severity = "high" if metrics.duplicate_rate > 0.1 else "medium"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.HIGH_DUPLICATE_RATE,
                severity=severity,
                description=f"重复标注率过高 ({metrics.duplicate_rate:.2f})",
                recommendation="检查并删除重复的标注，确保每个目标只被标注一次"
            ))
        
        # 检查重叠标注
        if metrics.overlap_rate > self.quality_thresholds['max_overlap_rate']:
            severity = "medium"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.HIGH_OVERLAP_RATE,
                severity=severity,
                description=f"重叠标注率过高 ({metrics.overlap_rate:.2f})",
                recommendation="检查并修正重叠的标注，确保每个目标的边界框准确"
            ))
        
        # 检查缺失类别
        missing_classes = set(range(len(self.class_names))) - set(metrics.class_distribution.keys())
        if missing_classes:
            missing_class_names = [self.class_names[i] for i in missing_classes]
            severity = "high" if len(missing_classes) > len(self.class_names) / 3 else "medium"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_CLASSES,
                severity=severity,
                description=f"缺失类别: {', '.join(missing_class_names)}",
                recommendation="添加缺失类别的样本，确保所有类别都有足够的训练数据"
            ))
        
        # 检查总体质量得分
        if metrics.overall_quality_score < self.quality_thresholds['min_quality_score']:
            # 不添加额外的问题，因为具体问题已经在上面列出
            pass
        
        return issues
    
    def generate_quality_report(self, annotation_dir: Union[str, Path], 
                              output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        生成质量报告
        
        Args:
            annotation_dir: 标注目录路径
            output_dir: 输出目录路径
            
        Returns:
            报告字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查标注质量
        metrics = self.check_annotation_quality(annotation_dir)
        
        # 识别质量问题
        issues = self.identify_quality_issues(metrics)
        
        # 生成改进建议
        recommendations = self._generate_recommendations(metrics, issues)
        
        # 创建报告字典
        report = {
            'summary': {
                'total_annotations': metrics.total_annotations,
                'valid_annotations': metrics.valid_annotations,
                'invalid_annotations': metrics.invalid_annotations,
                'validation_rate': metrics.validation_rate,
                'overall_quality_score': metrics.overall_quality_score,
                'issue_count': len(issues)
            },
            'metrics': {
                'class_distribution': metrics.class_distribution,
                'class_balance_score': metrics.class_balance_score,
                'annotation_density': metrics.annotation_density,
                'avg_bbox_size': metrics.avg_bbox_size,
                'bbox_size_variance': metrics.bbox_size_variance,
                'duplicate_rate': metrics.duplicate_rate,
                'overlap_rate': metrics.overlap_rate
            },
            'issues': [
                {
                    'type': issue.issue_type,
                    'severity': issue.severity,
                    'description': issue.description,
                    'recommendation': issue.recommendation
                }
                for issue in issues
            ],
            'recommendations': recommendations
        }
        
        # 保存JSON报告
        json_report_path = output_dir / "quality_report.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可读文本报告
        text_report_path = output_dir / "quality_report.txt"
        self._generate_text_report(metrics, issues, recommendations, text_report_path)
        
        logger.info(f"质量报告已生成: {json_report_path}, {text_report_path}")
        
        return report
    
    def _generate_recommendations(self, metrics: QualityMetrics, 
                                issues: List[QualityIssue]) -> List[str]:
        """
        生成改进建议
        
        Args:
            metrics: 质量指标对象
            issues: 质量问题列表
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 根据问题严重程度排序
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_issues = sorted(issues, key=lambda x: severity_order.get(x.severity, 3))
        
        # 添加问题相关建议
        for issue in sorted_issues:
            if issue.recommendation and issue.recommendation not in recommendations:
                recommendations.append(issue.recommendation)
        
        # 添加通用建议
        if metrics.overall_quality_score < 70:
            recommendations.append("考虑重新审查标注质量，特别是高严重度的问题")
        
        if metrics.validation_rate < 0.9:
            recommendations.append("使用自动化工具验证标注格式，确保符合YOLO格式要求")
        
        if len(metrics.class_distribution) < len(self.class_names):
            recommendations.append("确保数据集包含所有目标类别的样本")
        
        # 数据增强建议
        if metrics.class_balance_score < 0.3 or metrics.total_annotations < 1000:
            recommendations.append("考虑使用数据增强技术增加训练样本，特别是少数类别")
        
        return recommendations
    
    def _generate_text_report(self, metrics: QualityMetrics, issues: List[QualityIssue],
                            recommendations: List[str], output_file: Path) -> None:
        """
        生成文本报告
        
        Args:
            metrics: 质量指标对象
            issues: 质量问题列表
            recommendations: 建议列表
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("标注质量报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 摘要
            f.write("摘要\n")
            f.write("-" * 30 + "\n")
            f.write(f"总标注数量: {metrics.total_annotations}\n")
            f.write(f"有效标注数量: {metrics.valid_annotations}\n")
            f.write(f"无效标注数量: {metrics.invalid_annotations}\n")
            f.write(f"有效标注率: {metrics.validation_rate:.2f}\n")
            f.write(f"总体质量得分: {metrics.overall_quality_score:.2f}/100\n")
            f.write(f"问题数量: {len(issues)}\n\n")
            
            # 详细指标
            f.write("详细指标\n")
            f.write("-" * 30 + "\n")
            
            # 类别分布
            f.write("类别分布:\n")
            for class_id, count in metrics.class_distribution.items():
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"未知类别({class_id})"
                f.write(f"  - {class_name}: {count}\n")
            
            f.write(f"类别平衡得分: {metrics.class_balance_score:.2f}\n")
            f.write(f"标注密度: {metrics.annotation_density:.2f} 个/图\n")
            f.write(f"平均边界框大小: {metrics.avg_bbox_size:.4f}\n")
            f.write(f"边界框大小方差: {metrics.bbox_size_variance:.4f}\n")
            f.write(f"重复标注率: {metrics.duplicate_rate:.2f}\n")
            f.write(f"重叠标注率: {metrics.overlap_rate:.2f}\n\n")
            
            # 质量问题
            if issues:
                f.write("质量问题\n")
                f.write("-" * 30 + "\n")
                
                for i, issue in enumerate(issues, 1):
                    f.write(f"{i}. [{issue.severity.upper()}] {issue.description}\n")
                    f.write(f"   建议: {issue.recommendation}\n\n")
            else:
                f.write("未发现质量问题\n\n")
            
            # 改进建议
            if recommendations:
                f.write("改进建议\n")
                f.write("-" * 30 + "\n")
                
                for i, recommendation in enumerate(recommendations, 1):
                    f.write(f"{i}. {recommendation}\n")
            
            # 结论
            f.write("\n结论\n")
            f.write("-" * 30 + "\n")
            
            if metrics.overall_quality_score >= 90:
                f.write("标注质量优秀，可以直接用于训练。\n")
            elif metrics.overall_quality_score >= 70:
                f.write("标注质量良好，建议修复发现的问题后再用于训练。\n")
            else:
                f.write("标注质量需要改进，请先解决高优先级问题再用于训练。\n")
    
    def analyze_annotation_consistency(self, annotation_dirs: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        分析多个标注目录之间的一致性
        
        Args:
            annotation_dirs: 标注目录路径列表
            
        Returns:
            一致性分析结果
        """
        if len(annotation_dirs) < 2:
            raise ValueError("至少需要两个标注目录进行一致性分析")
        
        # 收集每个目录的类别分布
        dir_class_distributions = []
        dir_metrics = []
        
        for dir_path in annotation_dirs:
            try:
                metrics = self.check_annotation_quality(dir_path)
                dir_metrics.append(metrics)
                dir_class_distributions.append(metrics.class_distribution)
            except Exception as e:
                logger.warning(f"处理目录 {dir_path} 失败: {e}")
                return {"error": str(e)}
        
        # 计算类别分布相似度
        similarity_matrix = []
        for i in range(len(dir_class_distributions)):
            row = []
            for j in range(len(dir_class_distributions)):
                if i == j:
                    row.append(1.0)
                else:
                    similarity = self._calculate_distribution_similarity(
                        dir_class_distributions[i], dir_class_distributions[j]
                    )
                    row.append(similarity)
            similarity_matrix.append(row)
        
        # 计算平均边界框大小差异
        bbox_size_diffs = []
        for i in range(len(dir_metrics)):
            for j in range(i + 1, len(dir_metrics)):
                if dir_metrics[i].avg_bbox_size > 0 and dir_metrics[j].avg_bbox_size > 0:
                    diff = abs(dir_metrics[i].avg_bbox_size - dir_metrics[j].avg_bbox_size) / max(
                        dir_metrics[i].avg_bbox_size, dir_metrics[j].avg_bbox_size
                    )
                    bbox_size_diffs.append(diff)
        
        avg_bbox_size_diff = sum(bbox_size_diffs) / len(bbox_size_diffs) if bbox_size_diffs else 0
        
        # 计算平均类别分布相似度
        avg_similarity = 0
        count = 0
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix[i])):
                avg_similarity += similarity_matrix[i][j]
                count += 1
        
        if count > 0:
            avg_similarity /= count
        
        # 生成结果
        result = {
            "directory_count": len(annotation_dirs),
            "similarity_matrix": similarity_matrix,
            "average_similarity": avg_similarity,
            "average_bbox_size_difference": avg_bbox_size_diff,
            "consistency_score": (avg_similarity + (1 - avg_bbox_size_diff)) / 2 * 100,
            "class_distributions": [dict(d) for d in dir_class_distributions]
        }
        
        return result
    
    def _calculate_distribution_similarity(self, dist1: Dict[int, int], dist2: Dict[int, int]) -> float:
        """
        计算两个分布的相似度
        
        Args:
            dist1: 第一个分布
            dist2: 第二个分布
            
        Returns:
            相似度 (0-1)
        """
        # 获取所有类别
        all_classes = set(dist1.keys()) | set(dist2.keys())
        
        if not all_classes:
            return 1.0
        
        # 计算每个类别的比例
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        # 计算余弦相似度
        dot_product = 0
        norm1 = 0
        norm2 = 0
        
        for class_id in all_classes:
            prop1 = dist1.get(class_id, 0) / total1
            prop2 = dist2.get(class_id, 0) / total2
            
            dot_product += prop1 * prop2
            norm1 += prop1 * prop1
            norm2 += prop2 * prop2
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1**0.5 * norm2**0.5)
        return similarity