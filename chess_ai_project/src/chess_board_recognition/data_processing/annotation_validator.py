# -*- coding: utf-8 -*-
"""
标注验证器模块

提供YOLO格式标注文件的验证功能，确保标注数据的质量和一致性。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

from pydantic import BaseModel, ValidationError


# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    file_count: int
    annotation_count: int


@dataclass
class ClassConsistencyResult:
    """类别一致性检查结果"""
    valid_classes: Set[int]
    invalid_classes: Set[int]
    missing_classes: Set[int]
    class_distribution: Dict[int, int]


class AnnotationValidator:
    """
    标注验证器
    
    负责验证YOLO格式标注文件的格式正确性、类别一致性等。
    """
    
    def __init__(self, class_names: List[str]):
        """
        初始化标注验证器
        
        Args:
            class_names: 类别名称列表，索引对应类别ID
        """
        self.class_names = class_names
        self.valid_class_ids = set(range(len(class_names)))
        
        logger.info(f"标注验证器初始化完成，支持 {len(class_names)} 个类别")
    
    def validate_yolo_format(self, annotation_file: Union[str, Path]) -> ValidationResult:
        """
        验证单个YOLO格式标注文件
        
        Args:
            annotation_file: 标注文件路径
            
        Returns:
            验证结果
        """
        annotation_file = Path(annotation_file)
        errors = []
        warnings = []
        annotation_count = 0
        
        if not annotation_file.exists():
            errors.append(f"文件不存在: {annotation_file}")
            return ValidationResult(False, errors, warnings, 0, 0)
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 空文件是允许的
            if not lines:
                return ValidationResult(True, [], [], 1, 0)
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                
                annotation_count += 1
                line_errors = self._validate_annotation_line(line, line_num, annotation_file)
                errors.extend(line_errors)
            
            # 检查是否有重复的标注
            duplicate_warnings = self._check_duplicate_annotations(lines, annotation_file)
            warnings.extend(duplicate_warnings)
            
        except Exception as e:
            errors.append(f"读取文件失败 {annotation_file}: {e}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, 1, annotation_count)
    
    def _validate_annotation_line(self, line: str, line_num: int, 
                                annotation_file: Path) -> List[str]:
        """验证单行标注"""
        errors = []
        
        # 解析YOLO格式：class_id x_center y_center width height
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{annotation_file}:{line_num} - 格式错误，应为5个值，实际为{len(parts)}个")
            return errors
        
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 验证类别ID
            if class_id not in self.valid_class_ids:
                errors.append(f"{annotation_file}:{line_num} - 无效的类别ID: {class_id}，"
                            f"有效范围: 0-{len(self.class_names)-1}")
            
            # 验证坐标范围
            if not (0.0 <= x_center <= 1.0):
                errors.append(f"{annotation_file}:{line_num} - x_center超出范围[0,1]: {x_center}")
            
            if not (0.0 <= y_center <= 1.0):
                errors.append(f"{annotation_file}:{line_num} - y_center超出范围[0,1]: {y_center}")
            
            if not (0.0 < width <= 1.0):
                errors.append(f"{annotation_file}:{line_num} - width超出范围(0,1]: {width}")
            
            if not (0.0 < height <= 1.0):
                errors.append(f"{annotation_file}:{line_num} - height超出范围(0,1]: {height}")
            
            # 验证边界框是否超出图像边界
            left = x_center - width / 2
            right = x_center + width / 2
            top = y_center - height / 2
            bottom = y_center + height / 2
            
            if left < 0 or right > 1 or top < 0 or bottom > 1:
                errors.append(f"{annotation_file}:{line_num} - 边界框超出图像边界: "
                            f"left={left:.3f}, right={right:.3f}, top={top:.3f}, bottom={bottom:.3f}")
            
        except ValueError as e:
            errors.append(f"{annotation_file}:{line_num} - 数值解析错误: {e}")
        
        return errors
    
    def _check_duplicate_annotations(self, lines: List[str], 
                                   annotation_file: Path) -> List[str]:
        """检查重复的标注"""
        warnings = []
        annotations = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) == 5:
                try:
                    # 将坐标四舍五入到3位小数进行比较
                    annotation = (
                        int(parts[0]),
                        round(float(parts[1]), 3),
                        round(float(parts[2]), 3),
                        round(float(parts[3]), 3),
                        round(float(parts[4]), 3)
                    )
                    annotations.append(annotation)
                except ValueError:
                    continue
        
        # 检查重复
        annotation_counts = Counter(annotations)
        for annotation, count in annotation_counts.items():
            if count > 1:
                warnings.append(f"{annotation_file} - 发现重复标注 {count} 次: {annotation}")
        
        return warnings
    
    def check_class_consistency(self, annotation_dir: Union[str, Path]) -> ClassConsistencyResult:
        """
        检查类别一致性
        
        Args:
            annotation_dir: 标注文件目录
            
        Returns:
            类别一致性检查结果
        """
        annotation_dir = Path(annotation_dir)
        
        if not annotation_dir.exists():
            logger.error(f"标注目录不存在: {annotation_dir}")
            return ClassConsistencyResult(set(), set(), set(), {})
        
        found_classes = set()
        class_distribution = defaultdict(int)
        
        # 遍历所有标注文件
        annotation_files = list(annotation_dir.rglob("*.txt"))
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            class_id = int(parts[0])
                            found_classes.add(class_id)
                            class_distribution[class_id] += 1
                        except ValueError:
                            continue
                            
            except Exception as e:
                logger.warning(f"读取标注文件失败 {annotation_file}: {e}")
                continue
        
        # 分析类别一致性
        valid_classes = found_classes & self.valid_class_ids
        invalid_classes = found_classes - self.valid_class_ids
        missing_classes = self.valid_class_ids - found_classes
        
        result = ClassConsistencyResult(
            valid_classes=valid_classes,
            invalid_classes=invalid_classes,
            missing_classes=missing_classes,
            class_distribution=dict(class_distribution)
        )
        
        logger.info(f"类别一致性检查完成: 有效类别{len(valid_classes)}个, "
                   f"无效类别{len(invalid_classes)}个, 缺失类别{len(missing_classes)}个")
        
        return result
    
    def generate_validation_report(self, annotation_dir: Union[str, Path], 
                                 output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        生成验证报告
        
        Args:
            annotation_dir: 标注文件目录
            output_dir: 输出目录，默认为标注目录
            
        Returns:
            验证报告字典
        """
        annotation_dir = Path(annotation_dir)
        if output_dir is None:
            output_dir = annotation_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有标注文件
        annotation_files = list(annotation_dir.rglob("*.txt"))
        
        if not annotation_files:
            logger.warning(f"在目录 {annotation_dir} 中未找到标注文件")
            return {}
        
        # 验证所有文件
        all_errors = []
        all_warnings = []
        total_files = 0
        total_annotations = 0
        valid_files = 0
        
        logger.info(f"开始验证 {len(annotation_files)} 个标注文件")
        
        file_results = {}
        for annotation_file in annotation_files:
            result = self.validate_yolo_format(annotation_file)
            
            relative_path = str(annotation_file.relative_to(annotation_dir))
            file_results[relative_path] = {
                'is_valid': result.is_valid,
                'errors': result.errors,
                'warnings': result.warnings,
                'annotation_count': result.annotation_count
            }
            
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            total_files += result.file_count
            total_annotations += result.annotation_count
            
            if result.is_valid:
                valid_files += 1
        
        # 检查类别一致性
        consistency_result = self.check_class_consistency(annotation_dir)
        
        # 生成报告
        report = {
            'summary': {
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': total_files - valid_files,
                'total_annotations': total_annotations,
                'total_errors': len(all_errors),
                'total_warnings': len(all_warnings),
                'validation_rate': valid_files / max(1, total_files) * 100
            },
            'class_consistency': {
                'valid_classes': sorted(list(consistency_result.valid_classes)),
                'invalid_classes': sorted(list(consistency_result.invalid_classes)),
                'missing_classes': sorted(list(consistency_result.missing_classes)),
                'class_distribution': consistency_result.class_distribution
            },
            'file_results': file_results,
            'all_errors': all_errors[:100],  # 限制错误数量
            'all_warnings': all_warnings[:100]  # 限制警告数量
        }
        
        # 保存报告
        self._save_validation_report(report, output_dir)
        
        logger.info(f"验证报告生成完成: {valid_files}/{total_files} 文件有效, "
                   f"{len(all_errors)} 个错误, {len(all_warnings)} 个警告")
        
        return report
    
    def _save_validation_report(self, report: Dict, output_dir: Path) -> None:
        """保存验证报告"""
        # 保存JSON格式报告
        json_file = output_dir / "validation_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可读的文本报告
        text_file = output_dir / "validation_report.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("标注文件验证报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体统计
            summary = report['summary']
            f.write("总体统计:\n")
            f.write("-" * 20 + "\n")
            f.write(f"总文件数: {summary['total_files']}\n")
            f.write(f"有效文件数: {summary['valid_files']}\n")
            f.write(f"无效文件数: {summary['invalid_files']}\n")
            f.write(f"总标注数: {summary['total_annotations']}\n")
            f.write(f"总错误数: {summary['total_errors']}\n")
            f.write(f"总警告数: {summary['total_warnings']}\n")
            f.write(f"验证通过率: {summary['validation_rate']:.2f}%\n\n")
            
            # 类别一致性
            consistency = report['class_consistency']
            f.write("类别一致性:\n")
            f.write("-" * 20 + "\n")
            f.write(f"有效类别: {consistency['valid_classes']}\n")
            f.write(f"无效类别: {consistency['invalid_classes']}\n")
            f.write(f"缺失类别: {consistency['missing_classes']}\n\n")
            
            # 类别分布
            if consistency['class_distribution']:
                f.write("类别分布:\n")
                f.write("-" * 20 + "\n")
                for class_id, count in sorted(consistency['class_distribution'].items()):
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"未知类别{class_id}"
                    f.write(f"  {class_id:2d} ({class_name}): {count}\n")
                f.write("\n")
            
            # 错误详情（前20个）
            if report['all_errors']:
                f.write("主要错误:\n")
                f.write("-" * 20 + "\n")
                for error in report['all_errors'][:20]:
                    f.write(f"  {error}\n")
                if len(report['all_errors']) > 20:
                    f.write(f"  ... 还有 {len(report['all_errors']) - 20} 个错误\n")
                f.write("\n")
            
            # 警告详情（前20个）
            if report['all_warnings']:
                f.write("主要警告:\n")
                f.write("-" * 20 + "\n")
                for warning in report['all_warnings'][:20]:
                    f.write(f"  {warning}\n")
                if len(report['all_warnings']) > 20:
                    f.write(f"  ... 还有 {len(report['all_warnings']) - 20} 个警告\n")
    
    def fix_common_issues(self, annotation_dir: Union[str, Path], 
                         backup: bool = True) -> Dict[str, int]:
        """
        修复常见的标注问题
        
        Args:
            annotation_dir: 标注文件目录
            backup: 是否备份原文件
            
        Returns:
            修复统计信息
        """
        annotation_dir = Path(annotation_dir)
        
        if backup:
            backup_dir = annotation_dir.parent / f"{annotation_dir.name}_backup"
            if backup_dir.exists():
                import shutil
                shutil.rmtree(backup_dir)
            shutil.copytree(annotation_dir, backup_dir)
            logger.info(f"已备份原文件到: {backup_dir}")
        
        fix_stats = {
            'files_processed': 0,
            'coordinates_fixed': 0,
            'duplicates_removed': 0,
            'invalid_classes_removed': 0
        }
        
        annotation_files = list(annotation_dir.rglob("*.txt"))
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                fixed_lines = []
                seen_annotations = set()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        continue  # 跳过格式错误的行
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 修复无效类别
                        if class_id not in self.valid_class_ids:
                            fix_stats['invalid_classes_removed'] += 1
                            continue
                        
                        # 修复坐标范围
                        original_coords = (x_center, y_center, width, height)
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        width = max(0.001, min(1.0, width))
                        height = max(0.001, min(1.0, height))
                        
                        if (x_center, y_center, width, height) != original_coords:
                            fix_stats['coordinates_fixed'] += 1
                        
                        # 检查重复
                        annotation_key = (class_id, round(x_center, 3), round(y_center, 3), 
                                        round(width, 3), round(height, 3))
                        if annotation_key in seen_annotations:
                            fix_stats['duplicates_removed'] += 1
                            continue
                        
                        seen_annotations.add(annotation_key)
                        fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        
                    except ValueError:
                        continue  # 跳过无法解析的行
                
                # 写回修复后的文件
                with open(annotation_file, 'w', encoding='utf-8') as f:
                    for line in fixed_lines:
                        f.write(line + '\n')
                
                fix_stats['files_processed'] += 1
                
            except Exception as e:
                logger.warning(f"修复文件失败 {annotation_file}: {e}")
                continue
        
        logger.info(f"修复完成: 处理{fix_stats['files_processed']}个文件, "
                   f"修复坐标{fix_stats['coordinates_fixed']}个, "
                   f"移除重复{fix_stats['duplicates_removed']}个, "
                   f"移除无效类别{fix_stats['invalid_classes_removed']}个")
        
        return fix_stats