"""
模型导出器测试

该模块包含了对模型导出器的详细测试。
"""

import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

from chess_ai_project.src.chess_board_recognition.training.model_exporter import ModelExporter

class TestModelExporter(unittest.TestCase):
    """
    模型导出器测试类
    """
    
    def setUp(self):
        """
        测试前准备
        """
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'model.pt')
        
        # 创建模拟模型文件
        with open(self.model_path, 'w') as f:
            f.write("mock model file")
    
    def tearDown(self):
        """
        测试后清理
        """
        shutil.rmtree(self.temp_dir)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.YOLO')
    def test_load_model(self, mock_yolo):
        """
        测试加载模型功能
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # 创建导出器
        exporter = ModelExporter(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        
        # 加载模型
        exporter.load_model()
        
        # 验证YOLO被正确调用
        mock_yolo.assert_called_once_with(self.model_path)
        self.assertEqual(exporter.model, mock_model)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.YOLO')
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.shutil')
    def test_export_model_onnx(self, mock_shutil, mock_yolo):
        """
        测试导出ONNX模型功能
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model.export.return_value = os.path.join(self.temp_dir, 'exported.onnx')
        mock_yolo.return_value = mock_model
        
        # 创建导出器
        exporter = ModelExporter(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        exporter.model = mock_model  # 直接设置模拟模型
        
        # 导出模型
        exported_path = exporter.export_model(format='onnx')
        
        # 验证导出方法被正确调用
        mock_model.export.assert_called_once()
        self.assertIn('format', mock_model.export.call_args[1])
        self.assertEqual(mock_model.export.call_args[1]['format'], 'onnx')
        
        # 验证复制操作
        mock_shutil.copy.assert_called_once()
        
        # 验证导出信息文件是否创建
        info_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_info.json')]
        self.assertEqual(len(info_files), 1)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.YOLO')
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.shutil')
    def test_export_model_torchscript(self, mock_shutil, mock_yolo):
        """
        测试导出TorchScript模型功能
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model.export.return_value = os.path.join(self.temp_dir, 'exported.torchscript')
        mock_yolo.return_value = mock_model
        
        # 创建导出器
        exporter = ModelExporter(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        exporter.model = mock_model  # 直接设置模拟模型
        
        # 导出模型
        exported_path = exporter.export_model(format='torchscript')
        
        # 验证导出方法被正确调用
        mock_model.export.assert_called_once()
        self.assertIn('format', mock_model.export.call_args[1])
        self.assertEqual(mock_model.export.call_args[1]['format'], 'torchscript')
        
        # 验证复制操作
        mock_shutil.copy.assert_called_once()
    
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.YOLO')
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.shutil')
    def test_export_model_openvino(self, mock_shutil, mock_yolo):
        """
        测试导出OpenVINO模型功能
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model.export.return_value = os.path.join(self.temp_dir, 'exported_openvino')
        mock_yolo.return_value = mock_model
        
        # 创建导出器
        exporter = ModelExporter(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        exporter.model = mock_model  # 直接设置模拟模型
        
        # 导出模型
        exported_path = exporter.export_model(format='openvino')
        
        # 验证导出方法被正确调用
        mock_model.export.assert_called_once()
        self.assertIn('format', mock_model.export.call_args[1])
        self.assertEqual(mock_model.export.call_args[1]['format'], 'openvino')
        
        # 验证复制操作（目录）
        mock_shutil.copytree.assert_called_once()
    
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.YOLO')
    def test_export_model_invalid_format(self, mock_yolo):
        """
        测试导出无效格式模型功能
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # 创建导出器
        exporter = ModelExporter(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        exporter.model = mock_model  # 直接设置模拟模型
        
        # 导出无效格式模型
        with self.assertRaises(ValueError):
            exporter.export_model(format='invalid_format')
    
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.YOLO')
    def test_export_to_multiple_formats(self, mock_yolo):
        """
        测试导出多种格式模型功能
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model.export.side_effect = [
            os.path.join(self.temp_dir, 'exported.onnx'),
            os.path.join(self.temp_dir, 'exported.torchscript'),
        ]
        mock_yolo.return_value = mock_model
        
        # 创建导出器
        exporter = ModelExporter(
            model_path=self.model_path,
            output_dir=self.temp_dir
        )
        exporter.model = mock_model  # 直接设置模拟模型
        
        # 模拟export_model方法
        exporter.export_model = MagicMock()
        exporter.export_model.side_effect = [
            os.path.join(self.temp_dir, 'model_onnx.onnx'),
            os.path.join(self.temp_dir, 'model_torchscript.torchscript'),
        ]
        
        # 导出多种格式模型
        formats = ['onnx', 'torchscript']
        results = exporter.export_to_multiple_formats(formats)
        
        # 验证export_model方法被调用了正确的次数
        self.assertEqual(exporter.export_model.call_count, 2)
        
        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertIn('onnx', results)
        self.assertIn('torchscript', results)
    
    @patch('chess_ai_project.src.chess_board_recognition.training.model_exporter.YOLO')
    def test_get_model_info(self, mock_yolo):
        """
        测试获取模型信息功能
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_model.type = 'detect'
        mock_model.task = 'detect'
        mock_model.names = {0: 'class1', 1: 'class2', 2: 'class3'}
        mock_model.model = MagicMock()
        mock_model.model.args = {'imgsz': [640, 640]}
        mock_yolo.return_value = mock_model
        
        # 创建导出器
        exporter = ModelExporter(output_dir=self.temp_dir)
        
        # 获取模型信息
        info = exporter.get_model_info(self.model_path)
        
        # 验证YOLO被正确调用
        mock_yolo.assert_called_once_with(self.model_path)
        
        # 验证信息内容
        self.assertEqual(info['model_path'], self.model_path)
        self.assertEqual(info['model_type'], 'detect')
        self.assertEqual(info['task'], 'detect')
        self.assertEqual(info['num_classes'], 3)
        self.assertEqual(info['class_names'], {0: 'class1', 1: 'class2', 2: 'class3'})
        self.assertEqual(info['input_size'], [640, 640])
    
    def test_compare_model_sizes(self):
        """
        测试比较模型大小功能
        """
        # 创建测试文件
        model1_path = os.path.join(self.temp_dir, 'model1.pt')
        model2_path = os.path.join(self.temp_dir, 'model2.onnx')
        model3_dir = os.path.join(self.temp_dir, 'model3_openvino')
        
        with open(model1_path, 'w') as f:
            f.write("a" * 1000)  # 1000字节
        
        with open(model2_path, 'w') as f:
            f.write("b" * 500)  # 500字节
        
        os.makedirs(model3_dir)
        with open(os.path.join(model3_dir, 'file1.bin'), 'w') as f:
            f.write("c" * 300)  # 300字节
        
        with open(os.path.join(model3_dir, 'file2.xml'), 'w') as f:
            f.write("d" * 200)  # 200字节
        
        # 创建导出器
        exporter = ModelExporter(output_dir=self.temp_dir)
        
        # 比较模型大小
        model_paths = {
            'pt': model1_path,
            'onnx': model2_path,
            'openvino': model3_dir,
        }
        results = exporter.compare_model_sizes(model_paths)
        
        # 验证结果
        self.assertEqual(len(results), 3)
        self.assertAlmostEqual(results['pt']['size_mb'], 1000 / (1024 * 1024), places=6)
        self.assertAlmostEqual(results['onnx']['size_mb'], 500 / (1024 * 1024), places=6)
        self.assertAlmostEqual(results['openvino']['size_mb'], 500 / (1024 * 1024), places=6)
        
        # 验证比率
        self.assertEqual(results['pt']['size_ratio'], 1.0)  # 基准格式
        self.assertEqual(results['onnx']['size_ratio'], 0.5)  # 是基准的一半大小
        self.assertEqual(results['openvino']['size_ratio'], 0.5)  # 是基准的一半大小

if __name__ == '__main__':
    unittest.main()