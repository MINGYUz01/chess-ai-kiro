"""
配置验证器测试

该模块包含了对配置验证器的详细测试。
"""

import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path

import yaml

from chess_ai_project.src.chess_board_recognition.training.config_validator import TrainingConfigValidator, DataConfigGenerator

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
    
    def test_validate_config_valid(self):
        """
        测试验证有效配置
        """
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
        self.assertEqual(corrected_config['epochs'], 100)
        self.assertEqual(corrected_config['batch_size'], 16)
        self.assertEqual(corrected_config['learning_rate'], 0.001)
    
    def test_validate_config_invalid_values(self):
        """
        测试验证无效值
        """
        invalid_config = {
            'epochs': -10,  # 无效值
            'batch_size': 0,  # 无效值
            'learning_rate': 2.0,  # 超出范围
        }
        
        valid, corrected_config, errors = self.validator.validate_config(invalid_config)
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
        
        # 检查是否自动修正了无效值
        self.assertGreaterEqual(corrected_config['epochs'], 1)
        self.assertGreaterEqual(corrected_config['batch_size'], 1)
        self.assertLessEqual(corrected_config['learning_rate'], 1.0)
    
    def test_validate_config_invalid_types(self):
        """
        测试验证无效类型
        """
        invalid_config = {
            'epochs': '100',  # 字符串而非整数
            'batch_size': 'invalid',  # 无效字符串
            'learning_rate': '0.001',  # 字符串而非浮点数
        }
        
        valid, corrected_config, errors = self.validator.validate_config(invalid_config)
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
        
        # 检查是否自动修正了无效类型
        self.assertIsInstance(corrected_config['epochs'], int)
        self.assertIsInstance(corrected_config['batch_size'], int)
        self.assertIsInstance(corrected_config['learning_rate'], float)
    
    def test_validate_config_unknown_params(self):
        """
        测试验证未知参数
        """
        config_with_unknown = {
            'epochs': 100,
            'unknown_param1': 'value1',
            'unknown_param2': 'value2',
        }
        
        valid, corrected_config, errors = self.validator.validate_config(config_with_unknown)
        self.assertFalse(valid)  # 缺少必需参数
        
        # 检查未知参数是否保留
        self.assertEqual(corrected_config['unknown_param1'], 'value1')
        self.assertEqual(corrected_config['unknown_param2'], 'value2')
    
    def test_generate_default_config(self):
        """
        测试生成默认配置功能
        """
        default_config = self.validator.generate_default_config()
        
        # 检查是否包含所有必需参数
        for param in self.validator.REQUIRED_PARAMS:
            self.assertIn(param, default_config)
        
        # 检查默认值是否正确
        for param, constraints in self.validator.PARAM_CONSTRAINTS.items():
            self.assertEqual(default_config[param], constraints['default'])
    
    def test_save_load_config_yaml(self):
        """
        测试保存和加载YAML配置
        """
        config = {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'image_size': 640,
        }
        
        yaml_path = os.path.join(self.temp_dir, 'config.yaml')
        self.validator.save_config(config, yaml_path)
        self.assertTrue(os.path.exists(yaml_path))
        
        loaded_config = self.validator.load_config(yaml_path)
        self.assertEqual(loaded_config['epochs'], config['epochs'])
        self.assertEqual(loaded_config['batch_size'], config['batch_size'])
        self.assertEqual(loaded_config['learning_rate'], config['learning_rate'])
        self.assertEqual(loaded_config['image_size'], config['image_size'])
    
    def test_save_load_config_json(self):
        """
        测试保存和加载JSON配置
        """
        config = {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'image_size': 640,
        }
        
        json_path = os.path.join(self.temp_dir, 'config.json')
        self.validator.save_config(config, json_path)
        self.assertTrue(os.path.exists(json_path))
        
        loaded_config = self.validator.load_config(json_path)
        self.assertEqual(loaded_config['epochs'], config['epochs'])
        self.assertEqual(loaded_config['batch_size'], config['batch_size'])
        self.assertEqual(loaded_config['learning_rate'], config['learning_rate'])
        self.assertEqual(loaded_config['image_size'], config['image_size'])
    
    def test_load_nonexistent_config(self):
        """
        测试加载不存在的配置文件
        """
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.yaml')
        loaded_config = self.validator.load_config(nonexistent_path)
        
        # 应该返回默认配置
        default_config = self.validator.generate_default_config()
        for param in self.validator.REQUIRED_PARAMS:
            self.assertEqual(loaded_config[param], default_config[param])
    
    def test_merge_configs(self):
        """
        测试合并配置
        """
        base_config = {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.001,
            'image_size': 640,
        }
        
        override_config = {
            'epochs': 200,
            'batch_size': 32,
            'new_param': 'value',
        }
        
        merged_config = self.validator.merge_configs(base_config, override_config)
        
        # 检查覆盖的参数
        self.assertEqual(merged_config['epochs'], 200)
        self.assertEqual(merged_config['batch_size'], 32)
        
        # 检查未覆盖的参数
        self.assertEqual(merged_config['learning_rate'], 0.001)
        self.assertEqual(merged_config['image_size'], 640)
        
        # 检查新参数
        self.assertEqual(merged_config['new_param'], 'value')

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
    
    def test_generate_data_yaml_with_all_params(self):
        """
        测试生成数据配置文件（所有参数）
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
    
    def test_generate_data_yaml_without_test(self):
        """
        测试生成数据配置文件（无测试集）
        """
        train_path = '/path/to/train'
        val_path = '/path/to/val'
        class_names = ['class1', 'class2', 'class3']
        
        output_path = os.path.join(self.temp_dir, 'data.yaml')
        
        # 生成配置文件
        self.generator.generate_data_yaml(
            train_path=train_path,
            val_path=val_path,
            class_names=class_names,
            output_path=output_path
        )
        
        # 检查文件内容
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.assertEqual(data['train'], train_path)
        self.assertEqual(data['val'], val_path)
        self.assertNotIn('test', data)
        self.assertEqual(data['names'], class_names)
        self.assertEqual(data['nc'], len(class_names))
    
    def test_generate_data_yaml_without_class_names(self):
        """
        测试生成数据配置文件（无类别名称）
        """
        train_path = '/path/to/train'
        val_path = '/path/to/val'
        
        output_path = os.path.join(self.temp_dir, 'data.yaml')
        
        # 生成配置文件
        self.generator.generate_data_yaml(
            train_path=train_path,
            val_path=val_path,
            output_path=output_path
        )
        
        # 检查文件内容
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.assertEqual(data['train'], train_path)
        self.assertEqual(data['val'], val_path)
        self.assertNotIn('names', data)
        self.assertNotIn('nc', data)
    
    def test_validate_data_paths_valid(self):
        """
        测试验证有效的数据路径
        """
        # 创建测试目录
        train_path = os.path.join(self.temp_dir, 'train')
        val_path = os.path.join(self.temp_dir, 'val')
        test_path = os.path.join(self.temp_dir, 'test')
        
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(test_path)
        
        # 验证路径
        valid = self.generator.validate_data_paths(train_path, val_path, test_path)
        self.assertTrue(valid)
    
    def test_validate_data_paths_invalid(self):
        """
        测试验证无效的数据路径
        """
        # 创建测试目录
        train_path = os.path.join(self.temp_dir, 'train')
        val_path = os.path.join(self.temp_dir, 'val')
        test_path = os.path.join(self.temp_dir, 'nonexistent')
        
        os.makedirs(train_path)
        os.makedirs(val_path)
        # 不创建test_path
        
        # 验证路径
        valid = self.generator.validate_data_paths(train_path, val_path, test_path)
        self.assertFalse(valid)

if __name__ == '__main__':
    unittest.main()