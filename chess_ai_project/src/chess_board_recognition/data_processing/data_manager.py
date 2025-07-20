# -*- coding: utf-8 -*-
"""
数据管理器模块

提供数据集管理、验证和统计功能，支持YOLO格式的标注数据处理。
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import random

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic.types import DirectoryPath, FilePath


# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """数据集划分结果"""
    train_files: List[str]
    val_files: List[str] 
    test_files: List[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass
class ClassStatistics:
    """类别统计信息"""
    class_name: str
    class_id: int
    total_instances: int
    files_count: int
    avg_instances_per_file: float


class YOLOAnnotation(BaseModel):
    """YOLO格式标注验证模型"""
    class_id: int = Field(ge=0, description="类别ID，必须大于等于0")
    x_center: float = Field(ge=0.0, le=1.0, description="中心点x坐标，归一化值[0,1]")
    y_center: float = Field(ge=0.0, le=1.0, description="中心点y坐标，归一化值[0,1]")
    width: float = Field(gt=0.0, le=1.0, description="宽度，归一化值(0,1]")
    height: float = Field(gt=0.0, le=1.0, description="高度，归一化值(0,1]")
    
    @field_validator('x_center', 'y_center', 'width', 'height')
    @classmethod
    def validate_coordinates(cls, v: float) -> float:
        """验证坐标值的有效性"""
        if not isinstance(v, (int, float)):
            raise ValueError("坐标值必须是数字")
        return float(v)


class DataManager:
    """
    数据管理器
    
    负责管理训练数据的目录结构、验证标注文件格式、
    划分数据集以及统计数据分布。
    """
    
    # 中国象棋棋子类别定义
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
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        初始化数据管理器
        
        Args:
            data_dir: 数据根目录路径
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.splits_dir = self.data_dir / "splits"
        
        # 确保数据目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"数据管理器初始化完成，数据目录: {self.data_dir}")
    
    def create_labelimg_structure(self) -> None:
        """
        创建labelImg兼容的目录结构
        
        创建以下目录结构：
        data_dir/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── splits/
        └── classes.txt
        """
        try:
            # 创建图像目录
            for split in ['train', 'val', 'test']:
                (self.images_dir / split).mkdir(parents=True, exist_ok=True)
                (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
            
            # 创建划分目录
            self.splits_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建类别文件
            classes_file = self.data_dir / "classes.txt"
            if not classes_file.exists():
                with open(classes_file, 'w', encoding='utf-8') as f:
                    for class_id, class_name in self.CHESS_CLASSES.items():
                        f.write(f"{class_name}\n")
            
            # 创建数据集配置文件
            self._create_dataset_yaml()
            
            logger.info("labelImg兼容目录结构创建完成")
            
        except Exception as e:
            logger.error(f"创建目录结构失败: {e}")
            raise    

    def _create_dataset_yaml(self) -> None:
        """创建YOLO数据集配置文件"""
        dataset_config = {
            'path': str(self.data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': len(self.CHESS_CLASSES),
            'names': list(self.CHESS_CLASSES.values())
        }
        
        yaml_file = self.data_dir / "dataset.yaml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            # 手动写入YAML格式，避免依赖yaml库
            f.write(f"path: {dataset_config['path']}\n")
            f.write(f"train: {dataset_config['train']}\n")
            f.write(f"val: {dataset_config['val']}\n")
            f.write(f"test: {dataset_config['test']}\n")
            f.write(f"nc: {dataset_config['nc']}\n")
            f.write("names:\n")
            for name in dataset_config['names']:
                f.write(f"  - {name}\n")
    
    def validate_annotations(self, annotation_dir: Optional[Union[str, Path]] = None) -> List[str]:
        """
        验证标注文件的格式正确性
        
        Args:
            annotation_dir: 标注文件目录，默认为labels目录
            
        Returns:
            错误信息列表，空列表表示验证通过
        """
        if annotation_dir is None:
            annotation_dir = self.labels_dir
        else:
            annotation_dir = Path(annotation_dir)
        
        errors = []
        
        if not annotation_dir.exists():
            errors.append(f"标注目录不存在: {annotation_dir}")
            return errors
        
        # 递归查找所有.txt文件
        annotation_files = list(annotation_dir.rglob("*.txt"))
        
        if not annotation_files:
            errors.append(f"在目录 {annotation_dir} 中未找到标注文件")
            return errors
        
        logger.info(f"开始验证 {len(annotation_files)} 个标注文件")
        
        for annotation_file in annotation_files:
            file_errors = self._validate_single_annotation(annotation_file)
            errors.extend(file_errors)
        
        if errors:
            logger.warning(f"发现 {len(errors)} 个验证错误")
        else:
            logger.info("所有标注文件验证通过")
        
        return errors
    
    def _validate_single_annotation(self, annotation_file: Path) -> List[str]:
        """
        验证单个标注文件
        
        Args:
            annotation_file: 标注文件路径
            
        Returns:
            该文件的错误信息列表
        """
        errors = []
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 检查文件是否为空
            if not lines:
                return errors  # 空文件是允许的（表示图像中没有目标）
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                
                # 解析YOLO格式：class_id x_center y_center width height
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{annotation_file}:{line_num} - 格式错误，应为5个值，实际为{len(parts)}个")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 使用Pydantic验证
                    annotation = YOLOAnnotation(
                        class_id=class_id,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height
                    )
                    
                    # 检查类别ID是否有效
                    if class_id not in self.CHESS_CLASSES:
                        errors.append(f"{annotation_file}:{line_num} - 无效的类别ID: {class_id}")
                    
                except ValueError as e:
                    errors.append(f"{annotation_file}:{line_num} - 数值解析错误: {e}")
                except ValidationError as e:
                    for error in e.errors():
                        field = error['loc'][0] if error['loc'] else 'unknown'
                        msg = error['msg']
                        errors.append(f"{annotation_file}:{line_num} - {field}: {msg}")
                        
        except Exception as e:
            errors.append(f"{annotation_file} - 文件读取错误: {e}")
        
        return errors
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                     test_ratio: float = 0.1, seed: int = 42) -> DatasetSplit:
        """
        自动划分数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
            seed: 随机种子
            
        Returns:
            数据集划分结果
        """
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"比例总和必须为1.0，当前为{total_ratio}")
        
        # 查找所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.images_dir.rglob(f"*{ext}"))
            image_files.extend(self.images_dir.rglob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"在目录 {self.images_dir} 中未找到图像文件")
        
        # 获取文件名（不含扩展名）
        file_stems = [f.stem for f in image_files]
        file_stems = list(set(file_stems))  # 去重
        
        # 设置随机种子并打乱
        random.seed(seed)
        random.shuffle(file_stems)
        
        # 计算划分点
        total_files = len(file_stems)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        # 划分文件
        train_files = file_stems[:train_end]
        val_files = file_stems[train_end:val_end]
        test_files = file_stems[val_end:]
        
        # 创建划分结果
        split_result = DatasetSplit(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            train_ratio=len(train_files) / total_files,
            val_ratio=len(val_files) / total_files,
            test_ratio=len(test_files) / total_files
        )
        
        # 保存划分结果
        self._save_split_files(split_result)
        
        logger.info(f"数据集划分完成: 训练集{len(train_files)}个, "
                   f"验证集{len(val_files)}个, 测试集{len(test_files)}个")
        
        return split_result
    
    def _save_split_files(self, split_result: DatasetSplit) -> None:
        """保存数据集划分文件"""
        splits = {
            'train': split_result.train_files,
            'val': split_result.val_files,
            'test': split_result.test_files
        }
        
        for split_name, file_list in splits.items():
            split_file = self.splits_dir / f"{split_name}.txt"
            with open(split_file, 'w', encoding='utf-8') as f:
                for filename in file_list:
                    f.write(f"{filename}\n")
        
        # 保存划分统计信息
        stats_file = self.splits_dir / "split_stats.json"
        stats = {
            'total_files': len(split_result.train_files) + len(split_result.val_files) + len(split_result.test_files),
            'train_count': len(split_result.train_files),
            'val_count': len(split_result.val_files),
            'test_count': len(split_result.test_files),
            'train_ratio': split_result.train_ratio,
            'val_ratio': split_result.val_ratio,
            'test_ratio': split_result.test_ratio
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)  
  
    def get_class_statistics(self) -> Dict[str, ClassStatistics]:
        """
        获取类别统计信息
        
        Returns:
            类别统计信息字典，键为类别名称
        """
        class_counts = defaultdict(int)  # 每个类别的实例总数
        file_class_counts = defaultdict(set)  # 每个类别出现的文件集合
        
        # 遍历所有标注文件
        annotation_files = list(self.labels_dir.rglob("*.txt"))
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                file_classes = set()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 1:
                        try:
                            class_id = int(parts[0])
                            if class_id in self.CHESS_CLASSES:
                                class_counts[class_id] += 1
                                file_classes.add(class_id)
                        except ValueError:
                            continue
                
                # 记录该文件包含的类别
                for class_id in file_classes:
                    file_class_counts[class_id].add(annotation_file.stem)
                    
            except Exception as e:
                logger.warning(f"读取标注文件失败 {annotation_file}: {e}")
                continue
        
        # 生成统计信息
        statistics = {}
        for class_id, class_name in self.CHESS_CLASSES.items():
            total_instances = class_counts[class_id]
            files_count = len(file_class_counts[class_id])
            avg_instances = total_instances / files_count if files_count > 0 else 0.0
            
            statistics[class_name] = ClassStatistics(
                class_name=class_name,
                class_id=class_id,
                total_instances=total_instances,
                files_count=files_count,
                avg_instances_per_file=avg_instances
            )
        
        # 保存统计结果
        self._save_statistics(statistics)
        
        logger.info(f"类别统计完成，共处理 {len(annotation_files)} 个标注文件")
        
        return statistics
    
    def _save_statistics(self, statistics: Dict[str, ClassStatistics]) -> None:
        """保存统计信息到文件"""
        stats_data = {}
        total_instances = 0
        total_files = len(list(self.labels_dir.rglob("*.txt")))
        
        for class_name, stats in statistics.items():
            stats_data[class_name] = {
                'class_id': stats.class_id,
                'total_instances': stats.total_instances,
                'files_count': stats.files_count,
                'avg_instances_per_file': round(stats.avg_instances_per_file, 2),
                'percentage': round(stats.total_instances / max(1, sum(s.total_instances for s in statistics.values())) * 100, 2)
            }
            total_instances += stats.total_instances
        
        # 添加总体统计
        stats_data['_summary'] = {
            'total_classes': len(self.CHESS_CLASSES),
            'total_instances': total_instances,
            'total_files': total_files,
            'avg_instances_per_file': round(total_instances / max(1, total_files), 2),
            'classes_with_data': len([s for s in statistics.values() if s.total_instances > 0])
        }
        
        # 保存到JSON文件
        stats_file = self.data_dir / "class_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        # 生成可读的统计报告
        self._generate_statistics_report(statistics, stats_data['_summary'])
    
    def _generate_statistics_report(self, statistics: Dict[str, ClassStatistics], 
                                  summary: Dict) -> None:
        """生成可读的统计报告"""
        report_file = self.data_dir / "statistics_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("棋盘识别数据集统计报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 总体统计
            f.write("总体统计:\n")
            f.write("-" * 30 + "\n")
            f.write(f"总类别数: {summary['total_classes']}\n")
            f.write(f"有数据的类别数: {summary['classes_with_data']}\n")
            f.write(f"总实例数: {summary['total_instances']}\n")
            f.write(f"总文件数: {summary['total_files']}\n")
            f.write(f"平均每文件实例数: {summary['avg_instances_per_file']}\n\n")
            
            # 各类别详细统计
            f.write("各类别详细统计:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'类别名称':<15} {'ID':<3} {'实例数':<8} {'文件数':<8} {'平均/文件':<10} {'占比%':<8}\n")
            f.write("-" * 70 + "\n")
            
            # 按实例数排序
            sorted_stats = sorted(statistics.items(), 
                                key=lambda x: x[1].total_instances, reverse=True)
            
            for class_name, stats in sorted_stats:
                percentage = stats.total_instances / max(1, summary['total_instances']) * 100
                f.write(f"{class_name:<15} {stats.class_id:<3} {stats.total_instances:<8} "
                       f"{stats.files_count:<8} {stats.avg_instances_per_file:<10.2f} {percentage:<8.2f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
    
    def organize_files_by_split(self) -> None:
        """
        根据划分文件组织图像和标注文件到对应目录
        
        将文件从原始位置移动到train/val/test子目录中
        """
        splits = ['train', 'val', 'test']
        
        for split in splits:
            split_file = self.splits_dir / f"{split}.txt"
            if not split_file.exists():
                logger.warning(f"划分文件不存在: {split_file}")
                continue
            
            # 读取该划分的文件列表
            with open(split_file, 'r', encoding='utf-8') as f:
                file_stems = [line.strip() for line in f if line.strip()]
            
            # 移动图像文件
            self._move_files_to_split(file_stems, split, 'images', 
                                    {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'})
            
            # 移动标注文件
            self._move_files_to_split(file_stems, split, 'labels', {'.txt'})
            
            logger.info(f"已组织 {len(file_stems)} 个文件到 {split} 目录")
    
    def _move_files_to_split(self, file_stems: List[str], split: str, 
                           file_type: str, extensions: set) -> None:
        """移动文件到指定划分目录"""
        source_dir = self.data_dir / file_type
        target_dir = source_dir / split
        
        for file_stem in file_stems:
            # 查找匹配的文件
            for ext in extensions:
                source_file = source_dir / f"{file_stem}{ext}"
                if source_file.exists():
                    target_file = target_dir / f"{file_stem}{ext}"
                    
                    # 确保目标目录存在
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 移动文件
                    shutil.move(str(source_file), str(target_file))
                    break
                
                # 尝试大写扩展名
                source_file = source_dir / f"{file_stem}{ext.upper()}"
                if source_file.exists():
                    target_file = target_dir / f"{file_stem}{ext.upper()}"
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source_file), str(target_file))
                    break
    
    def validate_dataset_integrity(self) -> Dict[str, List[str]]:
        """
        验证数据集完整性
        
        检查图像文件和标注文件是否一一对应
        
        Returns:
            包含各种问题的字典
        """
        issues = {
            'missing_labels': [],      # 缺少标注文件的图像
            'missing_images': [],      # 缺少图像文件的标注
            'empty_labels': [],        # 空标注文件
            'invalid_labels': []       # 无效标注文件
        }
        
        # 查找所有图像和标注文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        image_files = set()
        for ext in image_extensions:
            image_files.update(f.stem for f in self.images_dir.rglob(f"*{ext}"))
            image_files.update(f.stem for f in self.images_dir.rglob(f"*{ext.upper()}"))
        
        label_files = set(f.stem for f in self.labels_dir.rglob("*.txt"))
        
        # 检查缺失的标注文件
        issues['missing_labels'] = list(image_files - label_files)
        
        # 检查缺失的图像文件
        issues['missing_images'] = list(label_files - image_files)
        
        # 检查空标注文件和无效标注文件
        for label_file in self.labels_dir.rglob("*.txt"):
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    issues['empty_labels'].append(str(label_file.relative_to(self.data_dir)))
                else:
                    # 验证标注格式
                    errors = self._validate_single_annotation(label_file)
                    if errors:
                        issues['invalid_labels'].append(str(label_file.relative_to(self.data_dir)))
                        
            except Exception as e:
                issues['invalid_labels'].append(str(label_file.relative_to(self.data_dir)))
        
        # 保存完整性检查报告
        self._save_integrity_report(issues)
        
        return issues
    
    def _save_integrity_report(self, issues: Dict[str, List[str]]) -> None:
        """保存完整性检查报告"""
        report_file = self.data_dir / "integrity_report.json"
        
        # 添加统计信息
        report_data = dict(issues)
        report_data['summary'] = {
            'total_issues': sum(len(issue_list) for issue_list in issues.values()),
            'missing_labels_count': len(issues['missing_labels']),
            'missing_images_count': len(issues['missing_images']),
            'empty_labels_count': len(issues['empty_labels']),
            'invalid_labels_count': len(issues['invalid_labels'])
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 生成可读报告
        text_report_file = self.data_dir / "integrity_report.txt"
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write("数据集完整性检查报告\n")
            f.write("=" * 40 + "\n\n")
            
            summary = report_data['summary']
            f.write(f"总问题数: {summary['total_issues']}\n")
            f.write(f"缺少标注文件: {summary['missing_labels_count']}\n")
            f.write(f"缺少图像文件: {summary['missing_images_count']}\n")
            f.write(f"空标注文件: {summary['empty_labels_count']}\n")
            f.write(f"无效标注文件: {summary['invalid_labels_count']}\n\n")
            
            for issue_type, file_list in issues.items():
                if file_list:
                    f.write(f"{issue_type}:\n")
                    f.write("-" * 20 + "\n")
                    for file_path in file_list[:10]:  # 只显示前10个
                        f.write(f"  {file_path}\n")
                    if len(file_list) > 10:
                        f.write(f"  ... 还有 {len(file_list) - 10} 个文件\n")
                    f.write("\n")
        
        logger.info(f"完整性检查完成，发现 {report_data['summary']['total_issues']} 个问题")