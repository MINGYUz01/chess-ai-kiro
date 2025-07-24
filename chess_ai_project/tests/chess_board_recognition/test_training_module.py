"""
训练模块单元测试

该模块包含了对训练模块各个组件的单元测试。
"""

import os
import unittest
import tempfile
import shutil
from pathlib import Path

import yaml
import numpy as np

from chess_ai_project.src.chess_board_recognition.training.config_validator import TrainingConfigValidator, DataConfigGenerator
from chess_ai_project.src.chess_board_recognition.training.trainer import YOLO11Trainer
from chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer import HyperparameterOptimizer
from chess_ai_project.src.chess_board_recognition.training.evaluator import ModelEvaluator
from chess_ai_project.src.chess_board_recognition.training.model_exporter import ModelExporter

class TestTrainingConfigValidator(unittest.TestCase):
    """
    训练配置验证器测试类
    """
    
    def setUp(self):
        """
        测试前准备
        """
        self.validator = TrainingConfigValidator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """
        测试后清理
        """
        shutil.rmtree(self.temp_dir)
    
    def test_validate_config(self):
        """
        测试配置验证功能
        """
        # 测试有效配置
        valid_config = {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'image_size': 640,
            'device': 'auto',
        }
        
        valid, corrected_config, errors = self.validator.validate_config(valid_config)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        
        # 测试无效配置
        invalid_config = {
            'epochs': -10,  # 无效值
            'batch_size': 'invalid',  # 无效类型
            'learning_rate': 2.0,  # 超出范围
            'unknown_param': 'value',  # 未知参数
        }
        
        valid, corrected_config, errors = self.validator.validate_config(invalid_config)
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
        
        # 检查是否自动修正了无效值
        self.assertGreaterEqual(corrected_config['epochs'], 1)
        self.assertIsInstance(corrected_config['batch_size'], int)
        self.assertLessEqual(corrected_config['learning_rate'], 1.0)
    
    def test_generate_default_config(self):
        """
        测试生成默认配置功能
        """
        default_config = self.validator.generate_default_config()
        
        # 检查是否包含所有必需参数
        for param in self.validator.REQUIRED_PARAMS:
            self.assertIn(param, default_config)
    
    def test_save_load_config(self):
        """
        测试保存和加载配置功能
        """
        config = {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'image_size': 640,
        }
        
        # 测试YAML格式
        yaml_path = os.path.join(self.temp_dir, 'config.yaml')
        self.validator.save_config(config, yaml_path)
        self.assertTrue(os.path.exists(yaml_path))
        
        loaded_config = self.validator.load_config(yaml_path)
        self.assertEqual(loaded_config['epochs'], config['epochs'])
        
        # 测试JSON格式
        json_path = os.path.join(self.temp_dir, 'config.json')
        self.validator.save_config(config, json_path)
        self.assertTrue(os.path.exists(json_path))
        
        loaded_config = self.validator.load_config(json_path)
        self.assertEqual(loaded_config['epochs'], config['epochs'])

class TestDataConfigGenerator(unittest.TestCase):
    """
    数据配置生成器测试类
    """
    
    def setUp(self):
        """
        测试前准备
        """
        self.generator = DataConfigGenerator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """
        测试后清理
        """
        shutil.rmtree(self.temp_dir)
    
    def test_generate_data_yaml(self):
        """
        测试生成数据配置文件功能
        """
        train_path = '/path/to/train'
        val_path = '/path/to/val'
        test_path = '/path/to/test'
        class_names = ['class1', 'class2', 'class3']
        
        output_path = os.path.join(self.temp_dir, 'data.yaml')
        
        # 生成配置文件
        self.generator.generate_data_yaml(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            class_names=class_names,
            output_path=output_path
        )
        
        # 检查文件是否存在
        self.assertTrue(os.path.exists(output_path))
        
        # 检查文件内容
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.assertEqual(data['train'], train_path)
        self.assertEqual(data['val'], val_path)
        self.assertEqual(data['test'], test_path)
        self.assertEqual(data['names'], class_names)
        self.assertEqual(data['nc'], len(class_names))

# 以下测试需要实际的模型和数据，所以我们只定义测试方法，但不实际运行
class TestYOLO11Trainer(unittest.TestCase):
    """
    YOLO11训练器测试类
    """
    
    @unittest.skip("需要实际的模型和数据")
    def test_load_model(self):
        """
        测试加载模型功能
        """
        trainer = YOLO11Trainer()
        trainer.load_model("yolo11n.pt")
        self.assertIsNotNone(trainer.model)
    
    @unittest.skip("需要实际的模型和数据")
    def test_train(self):
        """
        测试训练功能
        """
        trainer = YOLO11Trainer()
        results = trainer.train("coco8.yaml", epochs=1)
        self.assertIsNotNone(results)

class TestHyperparameterOptimizer(unittest.TestCase):
    """
    超参数优化器测试类
    """
    
    @unittest.skip("需要实际的模型和数据")
    def test_random_search(self):
        """
        测试随机搜索功能
        """
        trainer = YOLO11Trainer()
        optimizer = HyperparameterOptimizer(trainer, "coco8.yaml")
        
        param_space = {
            'batch': (8, 16),
            'lr0': (0.0001, 0.001),
        }
        
        best_params = optimizer.random_search(param_space, num_trials=2, epochs=1)
        self.assertIsNotNone(best_params)

class TestModelEvaluator(unittest.TestCase):
    """
    模型评估器测试类
    """
    
    @unittest.skip("需要实际的模型和数据")
    def test_evaluate_on_dataset(self):
        """
        测试在数据集上评估模型功能
        """
        evaluator = ModelEvaluator("yolo11n.pt")
        results = evaluator.evaluate_on_dataset("coco8.yaml")
        self.assertIsNotNone(results)

class TestModelExporter(unittest.TestCase):
    """
    模型导出器测试类
    """
    
    @unittest.skip("需要实际的模型和数据")
    def test_export_model(self):
        """
        测试导出模型功能
        """
        exporter = ModelExporter("yolo11n.pt")
        path = exporter.export_model(format="onnx")
        self.assertTrue(os.path.exists(path))

if __name__ == '__main__':
    unittest.main()