"""
超参数优化器测试

该模块包含了对超参数优化器的详细测试。
"""

import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np

from chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer import HyperparameterOptimizer

class TestHyperparameterOptimizer(unittest.TestCase):
    """
    超参数优化器测试类
    """
    
    def setUp(self):
        """
        测试前准备
        """
        self.temp_dir = tempfile.mkdtemp()
        self.data_yaml_path = os.path.join(self.temp_dir, 'data.yaml')
        
        # 创建模拟数据配置文件
        with open(self.data_yaml_path, 'w') as f:
            f.write("train: ./train\nval: ./val\nnc: 3\nnames: ['class1', 'class2', 'class3']")
    
    def tearDown(self):
        """
        测试后清理
        """
        shutil.rmtree(self.temp_dir)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer.YOLO11Trainer')
    def test_evaluate_params(self, mock_trainer_class):
        """
        测试评估参数功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(results_dict={'metrics/mAP50-95(B)': 0.8})
        mock_trainer_class.return_value = mock_trainer
        
        # 创建优化器
        optimizer = HyperparameterOptimizer(
            trainer=mock_trainer,
            data_yaml_path=self.data_yaml_path,
            output_dir=self.temp_dir
        )
        
        # 评估参数
        params = {'batch': 16, 'lr0': 0.001}
        metric = optimizer._evaluate_params(params, epochs=1)
        
        # 验证训练方法被正确调用
        mock_trainer.train.assert_called_once()
        self.assertEqual(metric, 0.8)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer.YOLO11Trainer')
    def test_grid_search(self, mock_trainer_class):
        """
        测试网格搜索功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(results_dict={'metrics/mAP50-95(B)': 0.8})
        mock_trainer_class.return_value = mock_trainer
        
        # 创建优化器
        optimizer = HyperparameterOptimizer(
            trainer=mock_trainer,
            data_yaml_path=self.data_yaml_path,
            output_dir=self.temp_dir
        )
        
        # 执行网格搜索
        param_grid = {
            'batch': [8, 16],
            'lr0': [0.001, 0.01],
        }
        best_params = optimizer.grid_search(param_grid, epochs=1)
        
        # 验证训练方法被调用了正确的次数（2x2=4次）
        self.assertEqual(mock_trainer.train.call_count, 4)
        self.assertIsNotNone(best_params)
        self.assertIn('batch', best_params)
        self.assertIn('lr0', best_params)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer.YOLO11Trainer')
    def test_random_search(self, mock_trainer_class):
        """
        测试随机搜索功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(results_dict={'metrics/mAP50-95(B)': 0.8})
        mock_trainer_class.return_value = mock_trainer
        
        # 创建优化器
        optimizer = HyperparameterOptimizer(
            trainer=mock_trainer,
            data_yaml_path=self.data_yaml_path,
            output_dir=self.temp_dir
        )
        
        # 执行随机搜索
        param_space = {
            'batch': (8, 32),
            'lr0': (0.0001, 0.01),
        }
        best_params = optimizer.random_search(param_space, num_trials=3, epochs=1)
        
        # 验证训练方法被调用了正确的次数
        self.assertEqual(mock_trainer.train.call_count, 3)
        self.assertIsNotNone(best_params)
        self.assertIn('batch', best_params)
        self.assertIn('lr0', best_params)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer.YOLO11Trainer')
    def test_save_search_history(self, mock_trainer_class):
        """
        测试保存搜索历史功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # 创建优化器
        optimizer = HyperparameterOptimizer(
            trainer=mock_trainer,
            data_yaml_path=self.data_yaml_path,
            output_dir=self.temp_dir
        )
        
        # 添加搜索历史
        optimizer.search_history = [
            {'params': {'batch': 16, 'lr0': 0.001}, 'metric': 0.8, 'timestamp': '2023-01-01 12:00:00'},
            {'params': {'batch': 32, 'lr0': 0.01}, 'metric': 0.7, 'timestamp': '2023-01-01 12:30:00'},
        ]
        optimizer.best_params = {'batch': 16, 'lr0': 0.001}
        optimizer.best_metric = 0.8
        
        # 保存搜索历史
        optimizer._save_search_history()
        
        # 验证文件是否创建
        history_path = os.path.join(self.temp_dir, 'search_history.json')
        self.assertTrue(os.path.exists(history_path))
        
        # 验证文件内容
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['metric'], 0.8)
        self.assertEqual(history[1]['metric'], 0.7)
        
        # 验证最佳参数文件是否创建
        best_params_path = os.path.join(self.temp_dir, 'best_params.yaml')
        self.assertTrue(os.path.exists(best_params_path))
    
    @patch('chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer.YOLO11Trainer')
    @patch('matplotlib.pyplot')
    def test_plot_search_history(self, mock_plt, mock_trainer_class):
        """
        测试绘制搜索历史功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # 创建优化器
        optimizer = HyperparameterOptimizer(
            trainer=mock_trainer,
            data_yaml_path=self.data_yaml_path,
            output_dir=self.temp_dir
        )
        
        # 添加搜索历史
        optimizer.search_history = [
            {'params': {'batch': 16, 'lr0': 0.001}, 'metric': 0.8, 'timestamp': '2023-01-01 12:00:00'},
            {'params': {'batch': 32, 'lr0': 0.01}, 'metric': 0.7, 'timestamp': '2023-01-01 12:30:00'},
        ]
        
        # 绘制搜索历史
        save_path = os.path.join(self.temp_dir, 'search_history.png')
        
        # 模拟plot_search_history方法
        with patch('chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer.plt', mock_plt):
            optimizer.plot_search_history(save_path=save_path)
        
        # 验证plt方法被调用
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()
        mock_plt.savefig.assert_called_once_with(save_path)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.hyperparameter_optimizer.YOLO11Trainer')
    def test_generate_optimization_report(self, mock_trainer_class):
        """
        测试生成优化报告功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # 创建优化器
        optimizer = HyperparameterOptimizer(
            trainer=mock_trainer,
            data_yaml_path=self.data_yaml_path,
            output_dir=self.temp_dir
        )
        
        # 添加搜索历史
        optimizer.search_history = [
            {'params': {'batch': 16, 'lr0': 0.001}, 'metric': 0.8, 'timestamp': '2023-01-01 12:00:00'},
            {'params': {'batch': 32, 'lr0': 0.01}, 'metric': 0.7, 'timestamp': '2023-01-01 12:30:00'},
        ]
        optimizer.best_params = {'batch': 16, 'lr0': 0.001}
        optimizer.best_metric = 0.8
        
        # 生成优化报告
        report_path = os.path.join(self.temp_dir, 'optimization_report.json')
        report = optimizer.generate_optimization_report(save_path=report_path)
        
        # 验证报告内容
        self.assertEqual(report['搜索试验次数'], 2)
        self.assertEqual(report['最佳指标'], 0.8)
        self.assertEqual(report['最差指标'], 0.7)
        self.assertEqual(report['平均指标'], 0.75)
        
        # 验证报告文件是否创建
        self.assertTrue(os.path.exists(report_path))
        
        # 验证文件内容
        with open(report_path, 'r') as f:
            saved_report = json.load(f)
        
        self.assertEqual(saved_report['最佳指标'], report['最佳指标'])

if __name__ == '__main__':
    unittest.main()