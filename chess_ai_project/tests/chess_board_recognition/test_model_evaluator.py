"""
模型评估器测试

该模块包含了对模型评估器的详细测试。
"""

import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np

from chess_ai_project.src.chess_board_recognition.training.evaluator import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    """
    模型评估器测试类
    """
    
    def setUp(self):
        """
        测试前准备
        """
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'model.pt')
        self.data_yaml_path = os.path.join(self.temp_dir, 'data.yaml')
        
        # 创建模拟文件
        with open(self.model_path, 'w') as f:
            f.write("mock model file")
        
        with open(self.data_yaml_path, 'w') as f:
            f.write("train: ./train\nval: ./val\nnc: 3\nnames: ['class1', 'class2', 'class3']")
    
    def tearDown(self):
        """
        测试后清理
        """
        shutil.rmtree(self.temp_dir)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.evaluator.YOLO11Trainer')
    def test_load_model(self, mock_trainer_class):
        """
        测试加载模型功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # 创建评估器
        evaluator = ModelEvaluator(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        
        # 加载模型
        evaluator.load_model()
        
        # 验证训练器被正确创建和调用
        mock_trainer_class.assert_called_once()
        mock_trainer.load_model.assert_called_once_with(self.model_path)
        self.assertEqual(evaluator.trainer, mock_trainer)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.evaluator.YOLO11Trainer')
    def test_evaluate_on_dataset(self, mock_trainer_class):
        """
        测试在数据集上评估模型功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_metrics = {
            'metrics/mAP50-95(B)': 0.8,
            'metrics/mAP50(B)': 0.9,
            'metrics/precision(B)': 0.85,
            'metrics/recall(B)': 0.83,
        }
        mock_trainer.validate.return_value = MagicMock(results_dict=mock_metrics)
        mock_trainer_class.return_value = mock_trainer
        
        # 创建评估器
        evaluator = ModelEvaluator(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        evaluator.trainer = mock_trainer  # 直接设置模拟训练器
        
        # 在数据集上评估模型
        results = evaluator.evaluate_on_dataset(self.data_yaml_path)
        
        # 验证验证方法被正确调用
        mock_trainer.validate.assert_called_once()
        self.assertIn('metrics', results)
        self.assertEqual(results['metrics'], mock_metrics)
        
        # 验证评估结果文件是否创建
        result_files = [f for f in os.listdir(self.temp_dir) if f.startswith('evaluation_') and f.endswith('.json')]
        self.assertEqual(len(result_files), 1)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.evaluator.YOLO11Trainer')
    def test_evaluate_on_images(self, mock_trainer_class):
        """
        测试在图像目录上评估模型功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_results = [
            MagicMock(boxes=MagicMock(conf=np.array([0.9, 0.8]))),
            MagicMock(boxes=MagicMock(conf=np.array([0.7]))),
        ]
        mock_model.predict.return_value = mock_results
        mock_trainer.model = mock_model
        mock_trainer_class.return_value = mock_trainer
        
        # 创建评估器
        evaluator = ModelEvaluator(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        evaluator.trainer = mock_trainer  # 直接设置模拟训练器
        
        # 创建测试目录
        image_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(image_dir)
        
        # 在图像目录上评估模型
        results = evaluator.evaluate_on_images(image_dir)
        
        # 验证预测方法被正确调用
        mock_model.predict.assert_called_once()
        self.assertIn('statistics', results)
        self.assertEqual(results['statistics']['total_images'], 2)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.evaluator.YOLO11Trainer')
    def test_generate_evaluation_report(self, mock_trainer_class):
        """
        测试生成评估报告功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # 创建评估器
        evaluator = ModelEvaluator(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        
        # 设置评估结果
        evaluator.evaluation_results = {
            'timestamp': '2023-01-01 12:00:00',
            'model_path': self.model_path,
            'data_path': self.data_yaml_path,
            'metrics': {
                'metrics/mAP50-95(B)': 0.8,
                'metrics/mAP50(B)': 0.9,
                'metrics/precision(B)': 0.85,
                'metrics/recall(B)': 0.83,
            },
            'parameters': {
                'batch': 16,
                'imgsz': 640,
            },
        }
        
        # 生成评估报告
        report_path = os.path.join(self.temp_dir, 'evaluation_report.json')
        report = evaluator.generate_evaluation_report(save_path=report_path)
        
        # 验证报告内容
        self.assertEqual(report['模型路径'], self.model_path)
        self.assertEqual(report['评估数据'], self.data_yaml_path)
        self.assertEqual(report['评估指标']['metrics/mAP50-95(B)'], 0.8)
        
        # 验证报告文件是否创建
        self.assertTrue(os.path.exists(report_path))
    
    @patch('chess_ai_project.src.chess_board_recognition.training.evaluator.YOLO11Trainer')
    @patch('chess_ai_project.src.chess_board_recognition.training.evaluator.plt')
    def test_plot_evaluation_results(self, mock_plt, mock_trainer_class):
        """
        测试绘制评估结果功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # 创建评估器
        evaluator = ModelEvaluator(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        
        # 设置评估结果
        evaluator.evaluation_results = {
            'metrics': {
                'metrics/precision(A)': 0.85,
                'metrics/precision(B)': 0.82,
                'metrics/precision(C)': 0.78,
                'metrics/recall(A)': 0.83,
                'metrics/recall(B)': 0.80,
                'metrics/recall(C)': 0.75,
            }
        }
        
        # 绘制评估结果
        save_path = os.path.join(self.temp_dir, 'evaluation_plot.png')
        evaluator.plot_evaluation_results(save_path=save_path)
        
        # 验证plt方法被调用
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once_with(save_path)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.evaluator.YOLO11Trainer')
    def test_compare_models(self, mock_trainer_class):
        """
        测试比较模型功能
        """
        # 设置模拟对象
        mock_trainer = MagicMock()
        mock_metrics1 = {'metrics/mAP50-95(B)': 0.8}
        mock_metrics2 = {'metrics/mAP50-95(B)': 0.7}
        
        # 设置不同的返回值
        mock_trainer.validate.side_effect = [
            MagicMock(results_dict=mock_metrics1),
            MagicMock(results_dict=mock_metrics2),
        ]
        mock_trainer_class.return_value = mock_trainer
        
        # 创建评估器
        evaluator = ModelEvaluator(output_dir=self.temp_dir)
        evaluator.trainer = mock_trainer  # 直接设置模拟训练器
        
        # 创建测试模型路径
        model_path1 = os.path.join(self.temp_dir, 'model1.pt')
        model_path2 = os.path.join(self.temp_dir, 'model2.pt')
        
        with open(model_path1, 'w') as f:
            f.write("mock model 1")
        
        with open(model_path2, 'w') as f:
            f.write("mock model 2")
        
        # 比较模型
        comparison = evaluator.compare_models(
            model_paths=[model_path1, model_path2],
            data_yaml_path=self.data_yaml_path
        )
        
        # 验证验证方法被调用了两次
        self.assertEqual(mock_trainer.validate.call_count, 2)
        self.assertEqual(len(comparison['models']), 2)
        self.assertEqual(comparison['models'][0]['metrics'], mock_metrics1)
        self.assertEqual(comparison['models'][1]['metrics'], mock_metrics2)
        
        # 验证比较结果文件是否创建
        comparison_files = [f for f in os.listdir(self.temp_dir) if f.startswith('model_comparison_') and f.endswith('.json')]
        self.assertEqual(len(comparison_files), 1)

if __name__ == '__main__':
    unittest.main()