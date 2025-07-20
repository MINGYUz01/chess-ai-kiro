# -*- coding: utf-8 -*-
"""
数据处理模块测试

测试DataManager和AnnotationValidator的功能。
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from chess_ai_project.src.chess_board_recognition.data_processing import (
    DataManager, AnnotationValidator, DatasetSplit, ClassStatistics, 
    ValidationResult, ClassConsistencyResult, YOLOAnnotation
)


class TestDataManager(unittest.TestCase):
    """DataManager测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_manager = DataManager(self.test_dir)
        
    def tearDown(self):
        """测试后清理"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """测试初始化"""
        self.assertTrue(self.test_dir.exists())
        self.assertEqual(self.data_manager.data_dir, self.test_dir)
        self.assertEqual(self.data_manager.images_dir, self.test_dir / "images")
        self.assertEqual(self.data_manager.labels_dir, self.test_dir / "labels")
        self.assertEqual(self.data_manager.splits_dir, self.test_dir / "splits")
    
    def test_create_labelimg_structure(self):
        """测试创建labelImg目录结构"""
        self.data_manager.create_labelimg_structure()
        
        # 检查目录是否创建
        expected_dirs = [
            self.test_dir / "images" / "train",
            self.test_dir / "images" / "val",
            self.test_dir / "images" / "test",
            self.test_dir / "labels" / "train",
            self.test_dir / "labels" / "val",
            self.test_dir / "labels" / "test",
            self.test_dir / "splits"
        ]
        
        for dir_path in expected_dirs:
            self.assertTrue(dir_path.exists(), f"目录不存在: {dir_path}")
        
        # 检查文件是否创建
        classes_file = self.test_dir / "classes.txt"
        dataset_file = self.test_dir / "dataset.yaml"
        
        self.assertTrue(classes_file.exists())
        self.assertTrue(dataset_file.exists())
        
        # 检查classes.txt内容
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        
        self.assertEqual(len(classes), len(self.data_manager.CHESS_CLASSES))
        self.assertIn("board", classes)
        self.assertIn("red_king", classes)
        self.assertIn("black_king", classes)
    
    def test_validate_annotations_empty_dir(self):
        """测试验证空标注目录"""
        self.data_manager.create_labelimg_structure()
        errors = self.data_manager.validate_annotations()
        
        # 空目录应该返回"未找到标注文件"错误
        self.assertEqual(len(errors), 1)
        self.assertIn("未找到标注文件", errors[0])
    
    def test_validate_annotations_valid_files(self):
        """测试验证有效标注文件"""
        self.data_manager.create_labelimg_structure()
        
        # 创建有效的标注文件
        labels_dir = self.data_manager.labels_dir
        test_annotation = labels_dir / "test.txt"
        
        with open(test_annotation, 'w', encoding='utf-8') as f:
            f.write("0 0.5 0.5 0.8 0.9\n")  # board
            f.write("2 0.1 0.1 0.05 0.08\n")  # red_king
        
        errors = self.data_manager.validate_annotations()
        self.assertEqual(len(errors), 0, f"不应该有验证错误: {errors}")
    
    def test_validate_annotations_invalid_files(self):
        """测试验证无效标注文件"""
        self.data_manager.create_labelimg_structure()
        
        # 创建无效的标注文件
        labels_dir = self.data_manager.labels_dir
        test_annotation = labels_dir / "invalid.txt"
        
        with open(test_annotation, 'w', encoding='utf-8') as f:
            f.write("invalid line\n")  # 格式错误
            f.write("99 0.5 0.5 0.1 0.1\n")  # 无效类别ID
            f.write("0 1.5 0.5 0.1 0.1\n")  # 坐标超出范围
        
        errors = self.data_manager.validate_annotations()
        self.assertGreater(len(errors), 0)
        
        # 检查是否包含预期的错误类型
        error_text = " ".join(errors)
        self.assertIn("格式错误", error_text)
        self.assertIn("无效的类别ID", error_text)
        # 检查坐标验证错误（可能是Pydantic的错误信息）
        self.assertTrue(any("x_center" in error or "超出范围" in error for error in errors))
    
    def test_split_dataset(self):
        """测试数据集划分"""
        self.data_manager.create_labelimg_structure()
        
        # 创建测试图像文件
        images_dir = self.data_manager.images_dir
        test_images = ["img1.jpg", "img2.png", "img3.jpg", "img4.png", "img5.jpg"]
        
        for img_name in test_images:
            (images_dir / img_name).touch()
        
        # 测试划分
        split_result = self.data_manager.split_dataset(
            train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
        )
        
        # 检查划分结果
        self.assertIsInstance(split_result, DatasetSplit)
        self.assertEqual(
            len(split_result.train_files) + len(split_result.val_files) + len(split_result.test_files),
            len(test_images)
        )
        
        # 检查比例
        total_files = len(test_images)
        self.assertAlmostEqual(len(split_result.train_files) / total_files, 0.6, delta=0.2)
        self.assertAlmostEqual(len(split_result.val_files) / total_files, 0.2, delta=0.2)
        self.assertAlmostEqual(len(split_result.test_files) / total_files, 0.2, delta=0.2)
        
        # 检查划分文件是否创建
        splits_dir = self.data_manager.splits_dir
        self.assertTrue((splits_dir / "train.txt").exists())
        self.assertTrue((splits_dir / "val.txt").exists())
        self.assertTrue((splits_dir / "test.txt").exists())
        self.assertTrue((splits_dir / "split_stats.json").exists())
    
    def test_split_dataset_invalid_ratios(self):
        """测试无效的划分比例"""
        self.data_manager.create_labelimg_structure()
        
        with self.assertRaises(ValueError):
            self.data_manager.split_dataset(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
    
    def test_get_class_statistics(self):
        """测试类别统计"""
        self.data_manager.create_labelimg_structure()
        
        # 创建测试标注文件
        labels_dir = self.data_manager.labels_dir
        
        # 文件1: 包含board和red_king
        with open(labels_dir / "test1.txt", 'w', encoding='utf-8') as f:
            f.write("0 0.5 0.5 0.8 0.9\n")  # board
            f.write("2 0.1 0.1 0.05 0.08\n")  # red_king
            f.write("2 0.9 0.1 0.05 0.08\n")  # red_king again
        
        # 文件2: 包含board和black_king
        with open(labels_dir / "test2.txt", 'w', encoding='utf-8') as f:
            f.write("0 0.5 0.5 0.8 0.9\n")  # board
            f.write("9 0.9 0.9 0.05 0.08\n")  # black_king
        
        # 获取统计信息
        statistics = self.data_manager.get_class_statistics()
        
        # 检查统计结果
        self.assertIsInstance(statistics, dict)
        self.assertEqual(len(statistics), len(self.data_manager.CHESS_CLASSES))
        
        # 检查board统计
        board_stats = statistics["board"]
        self.assertEqual(board_stats.total_instances, 2)  # 两个文件各有一个
        self.assertEqual(board_stats.files_count, 2)
        self.assertEqual(board_stats.avg_instances_per_file, 1.0)
        
        # 检查red_king统计
        red_king_stats = statistics["red_king"]
        self.assertEqual(red_king_stats.total_instances, 2)  # 一个文件有两个
        self.assertEqual(red_king_stats.files_count, 1)
        self.assertEqual(red_king_stats.avg_instances_per_file, 2.0)
        
        # 检查统计文件是否创建
        self.assertTrue((self.test_dir / "class_statistics.json").exists())
        self.assertTrue((self.test_dir / "statistics_report.txt").exists())
    
    def test_validate_dataset_integrity(self):
        """测试数据集完整性验证"""
        self.data_manager.create_labelimg_structure()
        
        # 创建测试文件
        images_dir = self.data_manager.images_dir
        labels_dir = self.data_manager.labels_dir
        
        # 有对应标注的图像
        (images_dir / "img1.jpg").touch()
        with open(labels_dir / "img1.txt", 'w') as f:
            f.write("0 0.5 0.5 0.8 0.9\n")
        
        # 缺少标注的图像
        (images_dir / "img2.jpg").touch()
        
        # 缺少图像的标注
        with open(labels_dir / "img3.txt", 'w') as f:
            f.write("0 0.5 0.5 0.8 0.9\n")
        
        # 空标注文件
        (images_dir / "img4.jpg").touch()
        (labels_dir / "img4.txt").touch()
        
        # 验证完整性
        issues = self.data_manager.validate_dataset_integrity()
        
        # 检查问题类型
        self.assertIn("missing_labels", issues)
        self.assertIn("missing_images", issues)
        self.assertIn("empty_labels", issues)
        
        self.assertIn("img2", issues["missing_labels"])
        self.assertIn("img3", issues["missing_images"])
        # 检查空标注文件（路径可能包含目录分隔符）
        self.assertTrue(any("img4.txt" in path for path in issues["empty_labels"]))


class TestAnnotationValidator(unittest.TestCase):
    """AnnotationValidator测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.class_names = [
            "board", "grid_lines", "red_king", "red_advisor", "red_bishop",
            "red_knight", "red_rook", "red_cannon", "red_pawn",
            "black_king", "black_advisor", "black_bishop", "black_knight", 
            "black_rook", "black_cannon", "black_pawn", "selected_piece"
        ]
        self.validator = AnnotationValidator(self.class_names)
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """测试后清理"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(len(self.validator.class_names), 17)
        self.assertEqual(self.validator.valid_class_ids, set(range(17)))
    
    def test_validate_yolo_format_valid_file(self):
        """测试验证有效的YOLO格式文件"""
        test_file = self.test_dir / "valid.txt"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("0 0.5 0.5 0.8 0.9\n")
            f.write("2 0.1 0.1 0.05 0.08\n")
        
        result = self.validator.validate_yolo_format(test_file)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.annotation_count, 2)
    
    def test_validate_yolo_format_invalid_file(self):
        """测试验证无效的YOLO格式文件"""
        test_file = self.test_dir / "invalid.txt"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("invalid format\n")
            f.write("99 0.5 0.5 0.1 0.1\n")  # 无效类别
            f.write("0 1.5 0.5 0.1 0.1\n")   # 坐标超范围
        
        result = self.validator.validate_yolo_format(test_file)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_validate_yolo_format_empty_file(self):
        """测试验证空文件"""
        test_file = self.test_dir / "empty.txt"
        test_file.touch()
        
        result = self.validator.validate_yolo_format(test_file)
        
        self.assertTrue(result.is_valid)  # 空文件是允许的
        self.assertEqual(result.annotation_count, 0)
    
    def test_validate_yolo_format_nonexistent_file(self):
        """测试验证不存在的文件"""
        test_file = self.test_dir / "nonexistent.txt"
        
        result = self.validator.validate_yolo_format(test_file)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        self.assertIn("文件不存在", result.errors[0])
    
    def test_check_class_consistency(self):
        """测试类别一致性检查"""
        # 创建测试标注文件
        (self.test_dir / "test1.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.1 0.1 0.05 0.05\n")
        (self.test_dir / "test2.txt").write_text("0 0.5 0.5 0.1 0.1\n99 0.9 0.9 0.05 0.05\n")  # 无效类别
        
        result = self.validator.check_class_consistency(self.test_dir)
        
        self.assertIsInstance(result, ClassConsistencyResult)
        self.assertIn(0, result.valid_classes)  # board
        self.assertIn(2, result.valid_classes)  # red_king
        self.assertIn(99, result.invalid_classes)  # 无效类别
        
        # 检查类别分布
        self.assertEqual(result.class_distribution[0], 2)  # board出现2次
        self.assertEqual(result.class_distribution[2], 1)  # red_king出现1次
    
    def test_generate_validation_report(self):
        """测试生成验证报告"""
        # 创建测试文件
        (self.test_dir / "valid.txt").write_text("0 0.5 0.5 0.8 0.9\n")
        (self.test_dir / "invalid.txt").write_text("invalid format\n")
        
        report = self.validator.generate_validation_report(self.test_dir)
        
        # 检查报告结构
        self.assertIn("summary", report)
        self.assertIn("class_consistency", report)
        self.assertIn("file_results", report)
        
        # 检查总结信息
        summary = report["summary"]
        self.assertEqual(summary["total_files"], 2)
        self.assertEqual(summary["valid_files"], 1)
        self.assertEqual(summary["invalid_files"], 1)
        
        # 检查报告文件是否创建
        self.assertTrue((self.test_dir / "validation_report.json").exists())
        self.assertTrue((self.test_dir / "validation_report.txt").exists())


class TestYOLOAnnotation(unittest.TestCase):
    """YOLOAnnotation测试类"""
    
    def test_valid_annotation(self):
        """测试有效标注"""
        annotation = YOLOAnnotation(
            class_id=0,
            x_center=0.5,
            y_center=0.5,
            width=0.8,
            height=0.9
        )
        
        self.assertEqual(annotation.class_id, 0)
        self.assertEqual(annotation.x_center, 0.5)
        self.assertEqual(annotation.y_center, 0.5)
        self.assertEqual(annotation.width, 0.8)
        self.assertEqual(annotation.height, 0.9)
    
    def test_invalid_class_id(self):
        """测试无效类别ID"""
        from pydantic import ValidationError
        
        with self.assertRaises(ValidationError):
            YOLOAnnotation(
                class_id=-1,  # 负数无效
                x_center=0.5,
                y_center=0.5,
                width=0.1,
                height=0.1
            )
    
    def test_invalid_coordinates(self):
        """测试无效坐标"""
        from pydantic import ValidationError
        
        # 测试超出范围的坐标
        with self.assertRaises(ValidationError):
            YOLOAnnotation(
                class_id=0,
                x_center=1.5,  # 超出[0,1]范围
                y_center=0.5,
                width=0.1,
                height=0.1
            )
        
        # 测试零宽度
        with self.assertRaises(ValidationError):
            YOLOAnnotation(
                class_id=0,
                x_center=0.5,
                y_center=0.5,
                width=0.0,  # 宽度必须大于0
                height=0.1
            )


if __name__ == '__main__':
    unittest.main()