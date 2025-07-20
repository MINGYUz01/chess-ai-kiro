# -*- coding: utf-8 -*-
"""
数据增强模块测试

测试DataAugmentor和QualityController的功能。
"""

import unittest
import tempfile
import shutil
import json
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

from chess_ai_project.src.chess_board_recognition.data_processing import (
    DataAugmentor, AugmentationConfig, YOLOBbox, create_default_augmentation_config,
    QualityController, QualityMetrics, QualityIssue, QualityIssueType, LabelImgConfigGenerator
)


class TestYOLOBbox(unittest.TestCase):
    """YOLOBbox测试类"""
    
    def test_valid_bbox(self):
        """测试有效边界框"""
        bbox = YOLOBbox(
            class_id=0,
            x_center=0.5,
            y_center=0.5,
            width=0.2,
            height=0.3
        )
        
        self.assertEqual(bbox.class_id, 0)
        self.assertEqual(bbox.x_center, 0.5)
        self.assertEqual(bbox.y_center, 0.5)
        self.assertEqual(bbox.width, 0.2)
        self.assertEqual(bbox.height, 0.3)
    
    def test_to_pascal_voc(self):
        """测试转换为Pascal VOC格式"""
        bbox = YOLOBbox(
            class_id=0,
            x_center=0.5,
            y_center=0.5,
            width=0.2,
            height=0.3
        )
        
        x_min, y_min, x_max, y_max = bbox.to_pascal_voc(100, 100)
        
        self.assertEqual(x_min, 40.0)  # (0.5 - 0.2/2) * 100
        self.assertEqual(y_min, 35.0)  # (0.5 - 0.3/2) * 100
        self.assertEqual(x_max, 60.0)  # (0.5 + 0.2/2) * 100
        self.assertEqual(y_max, 65.0)  # (0.5 + 0.3/2) * 100
    
    def test_from_pascal_voc(self):
        """测试从Pascal VOC格式创建"""
        bbox = YOLOBbox.from_pascal_voc(40, 35, 60, 65, 100, 100, 0)
        
        self.assertEqual(bbox.class_id, 0)
        self.assertAlmostEqual(bbox.x_center, 0.5, places=6)
        self.assertAlmostEqual(bbox.y_center, 0.5, places=6)
        self.assertAlmostEqual(bbox.width, 0.2, places=6)
        self.assertAlmostEqual(bbox.height, 0.3, places=6)


class TestAugmentationConfig(unittest.TestCase):
    """AugmentationConfig测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AugmentationConfig()
        
        self.assertEqual(config.rotation_range, (-15.0, 15.0))
        self.assertEqual(config.scale_range, (0.8, 1.2))
        self.assertEqual(config.horizontal_flip_probability, 0.5)
        self.assertEqual(config.vertical_flip_probability, 0.0)
    
    def test_create_default_config(self):
        """测试创建棋盘识别专用配置"""
        config = create_default_augmentation_config()
        
        self.assertEqual(config.rotation_range, (-10.0, 10.0))
        self.assertEqual(config.horizontal_flip_probability, 0.3)
        self.assertEqual(config.vertical_flip_probability, 0.0)


class TestDataAugmentor(unittest.TestCase):
    """DataAugmentor测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = AugmentationConfig(
            rotation_range=(-5.0, 5.0),
            horizontal_flip_probability=0.5,
            brightness_range=(-0.1, 0.1)
        )
        self.augmentor = DataAugmentor(self.config)
        
        # 创建测试图像
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 创建测试边界框
        self.test_bboxes = [
            YOLOBbox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.3),
            YOLOBbox(class_id=1, x_center=0.3, y_center=0.7, width=0.1, height=0.1)
        ]
    
    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.augmentor.config, AugmentationConfig)
        self.assertEqual(self.augmentor.config.rotation_range, (-5.0, 5.0))
    
    def test_augment_image_and_bboxes(self):
        """测试图像和边界框增强"""
        aug_image, aug_bboxes = self.augmentor.augment_image_and_bboxes(
            self.test_image, self.test_bboxes
        )
        
        # 检查输出类型和形状
        self.assertIsInstance(aug_image, np.ndarray)
        self.assertEqual(aug_image.shape, self.test_image.shape)
        self.assertIsInstance(aug_bboxes, list)
        self.assertEqual(len(aug_bboxes), len(self.test_bboxes))
        
        # 检查边界框类型
        for bbox in aug_bboxes:
            self.assertIsInstance(bbox, YOLOBbox)
    
    def test_augment_empty_bboxes(self):
        """测试无边界框的图像增强"""
        aug_image, aug_bboxes = self.augmentor.augment_image_and_bboxes(
            self.test_image, []
        )
        
        self.assertIsInstance(aug_image, np.ndarray)
        self.assertEqual(aug_image.shape, self.test_image.shape)
        self.assertEqual(len(aug_bboxes), 0)
    
    def test_preview_augmentation(self):
        """测试增强预览"""
        samples = self.augmentor.preview_augmentation(
            self.test_image, self.test_bboxes, num_samples=3
        )
        
        self.assertEqual(len(samples), 3)
        
        for aug_image, aug_bboxes in samples:
            self.assertIsInstance(aug_image, np.ndarray)
            self.assertEqual(aug_image.shape, self.test_image.shape)
            self.assertIsInstance(aug_bboxes, list)
    
    def test_get_augmentation_suggestions(self):
        """测试增强建议"""
        dataset_stats = {
            'total_files': 500,
            'class_distribution': {0: 100, 1: 50, 2: 300}
        }
        
        suggestions = self.augmentor.get_augmentation_suggestions(dataset_stats)
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # 检查是否包含预期的建议
        suggestions_text = " ".join(suggestions)
        self.assertIn("数据量", suggestions_text)
        self.assertIn("类别分布", suggestions_text)


class TestQualityController(unittest.TestCase):
    """QualityController测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.class_names = [
            "board", "red_king", "black_king", "red_pawn", "black_pawn"
        ]
        self.quality_controller = QualityController(self.class_names)
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """测试后清理"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(len(self.quality_controller.class_names), 5)
        self.assertIn('min_validation_rate', self.quality_controller.quality_thresholds)
    
    def test_check_annotation_quality_empty_dir(self):
        """测试检查空目录的质量"""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()
        
        with self.assertRaises(ValueError):
            self.quality_controller.check_annotation_quality(empty_dir)
    
    def test_check_annotation_quality_with_data(self):
        """测试检查有数据的质量"""
        # 创建测试标注文件
        annotation_dir = self.test_dir / "annotations"
        annotation_dir.mkdir()
        
        # 创建有效标注文件
        with open(annotation_dir / "test1.txt", 'w') as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
            f.write("1 0.3 0.7 0.1 0.1\n")
        
        # 创建部分无效标注文件
        with open(annotation_dir / "test2.txt", 'w') as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
            f.write("99 0.3 0.7 0.1 0.1\n")  # 无效类别
        
        quality_metrics = self.quality_controller.check_annotation_quality(annotation_dir)
        
        self.assertIsInstance(quality_metrics, QualityMetrics)
        self.assertEqual(quality_metrics.total_annotations, 4)
        self.assertEqual(quality_metrics.valid_annotations, 3)
        self.assertEqual(quality_metrics.invalid_annotations, 1)
        self.assertAlmostEqual(quality_metrics.validation_rate, 0.75, places=2)
    
    def test_identify_quality_issues(self):
        """测试识别质量问题"""
        # 创建低质量指标
        quality_metrics = QualityMetrics(
            total_annotations=100,
            valid_annotations=80,
            invalid_annotations=20,
            validation_rate=0.8,  # 低于默认阈值0.95
            class_balance_score=0.2,  # 低于默认阈值0.3
            annotation_density=0.5,  # 低于默认阈值1.0
            avg_bbox_size=0.01,
            bbox_size_variance=0.1,
            duplicate_rate=0.1,  # 高于默认阈值0.05
            overlap_rate=0.05,
            overall_quality_score=60.0
        )
        
        issues = self.quality_controller.identify_quality_issues(quality_metrics)
        
        self.assertIsInstance(issues, list)
        self.assertGreater(len(issues), 0)
        
        # 检查是否识别出预期的问题
        issue_types = {issue.issue_type for issue in issues}
        self.assertIn("low_validation_rate", issue_types)
        self.assertIn("class_imbalance", issue_types)
        self.assertIn("high_duplicate_rate", issue_types)
    
    def test_generate_quality_report(self):
        """测试生成质量报告"""
        # 创建测试数据
        annotation_dir = self.test_dir / "annotations"
        annotation_dir.mkdir()
        
        with open(annotation_dir / "test.txt", 'w') as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
            f.write("1 0.3 0.7 0.1 0.1\n")
        
        output_dir = self.test_dir / "reports"
        
        report = self.quality_controller.generate_quality_report(
            annotation_dir, output_dir
        )
        
        # 检查报告结构
        self.assertIn('summary', report)
        self.assertIn('metrics', report)
        self.assertIn('issues', report)
        self.assertIn('recommendations', report)
        
        # 检查文件是否生成
        self.assertTrue((output_dir / "quality_report.json").exists())
        self.assertTrue((output_dir / "quality_report.txt").exists())


class TestLabelImgConfigGenerator(unittest.TestCase):
    """LabelImgConfigGenerator测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.class_names = ["board", "red_king", "black_king"]
        self.generator = LabelImgConfigGenerator(self.class_names)
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """测试后清理"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_generate_predefined_classes(self):
        """测试生成预定义类别文件"""
        output_file = self.test_dir / "classes.txt"
        
        self.generator.generate_predefined_classes(output_file)
        
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.assertEqual(lines, self.class_names)
    
    def test_generate_config_file(self):
        """测试生成配置文件"""
        output_dir = self.test_dir / "config"
        image_dir = self.test_dir / "images"
        label_dir = self.test_dir / "labels"
        
        # 创建目录
        image_dir.mkdir()
        label_dir.mkdir()
        
        self.generator.generate_config_file(output_dir, image_dir, label_dir)
        
        # 检查文件是否生成
        self.assertTrue((output_dir / "predefined_classes.txt").exists())
        self.assertTrue((output_dir / "labelimg_config.json").exists())
        
        # 检查配置文件内容
        with open(output_dir / "labelimg_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.assertIn('image_dir', config)
        self.assertIn('label_dir', config)
        self.assertIn('classes', config)
        self.assertEqual(config['classes'], self.class_names)
        self.assertEqual(config['class_count'], len(self.class_names))


class TestDataAugmentorIntegration(unittest.TestCase):
    """DataAugmentor集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.augmentor = DataAugmentor(create_default_augmentation_config())
        
    def tearDown(self):
        """测试后清理"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_augment_dataset_integration(self):
        """测试数据集增强集成功能"""
        # 创建输入数据结构
        input_dir = self.test_dir / "input"
        input_images_dir = input_dir / "images"
        input_labels_dir = input_dir / "labels"
        input_images_dir.mkdir(parents=True)
        input_labels_dir.mkdir(parents=True)
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(input_images_dir / "test.jpg"), test_image)
        
        # 创建对应标注
        with open(input_labels_dir / "test.txt", 'w') as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
        
        # 执行增强
        output_dir = self.test_dir / "output"
        
        stats = self.augmentor.augment_dataset(
            input_dir, output_dir, 
            augmentation_factor=2, 
            preserve_original=True
        )
        
        # 检查统计信息
        self.assertEqual(stats['original_images'], 1)
        self.assertEqual(stats['augmented_images'], 2)
        self.assertEqual(stats['total_images'], 3)  # 1原始 + 2增强
        
        # 检查输出文件
        output_images_dir = output_dir / "images"
        output_labels_dir = output_dir / "labels"
        
        self.assertTrue(output_images_dir.exists())
        self.assertTrue(output_labels_dir.exists())
        
        # 检查生成的文件数量
        image_files = list(output_images_dir.glob("*.jpg"))
        label_files = list(output_labels_dir.glob("*.txt"))
        
        self.assertEqual(len(image_files), 3)
        self.assertEqual(len(label_files), 3)
        
        # 检查统计文件
        self.assertTrue((output_dir / "augmentation_stats.json").exists())
        self.assertTrue((output_dir / "augmentation_report.txt").exists())


if __name__ == '__main__':
    unittest.main()