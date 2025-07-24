"""
模型导出模块

该模块提供了用于导出YOLO11模型为不同格式的类和函数。
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

from ultralytics import YOLO

from ..system_management.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

class ModelExporter:
    """
    模型导出器类
    
    该类用于将YOLO11模型导出为不同格式，如ONNX、TorchScript等。
    """
    
    # 支持的导出格式
    SUPPORTED_FORMATS = [
        'onnx',        # ONNX格式
        'torchscript', # TorchScript格式
        'openvino',    # OpenVINO格式
        'engine',      # TensorRT格式
        'coreml',      # CoreML格式
        'saved_model', # TensorFlow SavedModel格式
        'pb',          # TensorFlow GraphDef格式
        'tflite',      # TensorFlow Lite格式
        'paddle',      # PaddlePaddle格式
        'ncnn',        # NCNN格式
    ]
    
    def __init__(self, model_path: str = None, output_dir: str = './exported_models'):
        """
        初始化模型导出器
        
        参数:
            model_path: 模型路径，如果为None则使用默认模型
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.model = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"模型导出器初始化完成，输出目录: {output_dir}")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        加载模型
        
        参数:
            model_path: 模型路径，如果为None则使用初始化时的路径
        """
        try:
            path = model_path or self.model_path
            if not path:
                logger.error("未指定模型路径")
                raise ValueError("未指定模型路径")
            
            logger.info(f"正在加载模型: {path}")
            
            # 加载模型
            self.model = YOLO(path)
            
            logger.info(f"模型加载成功: {path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def export_model(self, format: str = 'onnx', **kwargs) -> str:
        """
        导出模型
        
        参数:
            format: 导出格式，支持的格式见SUPPORTED_FORMATS
            **kwargs: 其他导出参数
            
        返回:
            导出的模型路径
        """
        try:
            if format not in self.SUPPORTED_FORMATS:
                logger.error(f"不支持的导出格式: {format}")
                raise ValueError(f"不支持的导出格式: {format}，支持的格式: {self.SUPPORTED_FORMATS}")
            
            if not self.model:
                self.load_model()
            
            logger.info(f"开始导出模型为{format}格式")
            
            # 设置导出参数
            export_args = {
                'format': format,
                'imgsz': kwargs.get('image_size', 640),
                'half': kwargs.get('half_precision', False),
                'simplify': kwargs.get('simplify', True),
                'opset': kwargs.get('opset', 12) if format == 'onnx' else None,
                'workspace': kwargs.get('workspace', 4) if format == 'engine' else None,
                'batch': kwargs.get('batch_size', 1),
                'device': kwargs.get('device', 'cpu'),
                'nms': kwargs.get('nms', True),
                'dynamic': kwargs.get('dynamic', True) if format in ['onnx', 'openvino'] else None,
            }
            
            # 过滤None值
            export_args = {k: v for k, v in export_args.items() if v is not None}
            
            # 执行导出
            exported_path = self.model.export(**export_args)
            
            # 复制到输出目录
            import shutil
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(self.model_path).split('.')[0] if self.model_path else 'model'
            
            # 确定目标路径
            if format == 'onnx':
                target_path = os.path.join(self.output_dir, f"{model_name}_{timestamp}.onnx")
                shutil.copy(exported_path, target_path)
            elif format == 'torchscript':
                target_path = os.path.join(self.output_dir, f"{model_name}_{timestamp}.torchscript")
                shutil.copy(exported_path, target_path)
            elif format in ['openvino', 'saved_model']:
                # 这些格式导出的是目录
                target_path = os.path.join(self.output_dir, f"{model_name}_{format}_{timestamp}")
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.copytree(exported_path, target_path)
            else:
                # 其他格式
                extension = format if format != 'engine' else 'trt'
                target_path = os.path.join(self.output_dir, f"{model_name}_{timestamp}.{extension}")
                shutil.copy(exported_path, target_path)
            
            logger.info(f"模型导出完成: {target_path}")
            
            # 保存导出信息
            export_info = {
                'timestamp': timestamp,
                'source_model': self.model_path,
                'format': format,
                'exported_path': target_path,
                'parameters': export_args,
            }
            
            info_path = os.path.join(self.output_dir, f"{model_name}_{format}_{timestamp}_info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(export_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"导出信息已保存至: {info_path}")
            
            return target_path
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            raise
    
    def export_to_multiple_formats(self, formats: List[str], **kwargs) -> Dict[str, str]:
        """
        将模型导出为多种格式
        
        参数:
            formats: 导出格式列表
            **kwargs: 其他导出参数
            
        返回:
            格式到导出路径的映射字典
        """
        try:
            logger.info(f"开始导出模型为多种格式: {formats}")
            
            results = {}
            for format in formats:
                try:
                    path = self.export_model(format, **kwargs)
                    results[format] = path
                except Exception as e:
                    logger.error(f"导出为{format}格式失败: {e}")
                    results[format] = f"导出失败: {e}"
            
            return results
        except Exception as e:
            logger.error(f"导出多种格式失败: {e}")
            raise
    
    def optimize_model(self, model_path: str, format: str = 'onnx', **kwargs) -> str:
        """
        优化已导出的模型
        
        参数:
            model_path: 模型路径
            format: 模型格式
            **kwargs: 其他优化参数
            
        返回:
            优化后的模型路径
        """
        try:
            logger.info(f"开始优化{format}格式模型: {model_path}")
            
            if format == 'onnx':
                return self._optimize_onnx(model_path, **kwargs)
            elif format == 'tflite':
                return self._optimize_tflite(model_path, **kwargs)
            else:
                logger.warning(f"不支持优化{format}格式模型")
                return model_path
        except Exception as e:
            logger.error(f"模型优化失败: {e}")
            raise
    
    def _optimize_onnx(self, model_path: str, **kwargs) -> str:
        """
        优化ONNX模型
        
        参数:
            model_path: 模型路径
            **kwargs: 其他优化参数
            
        返回:
            优化后的模型路径
        """
        try:
            # 尝试导入onnxruntime
            try:
                import onnx
                import onnxruntime as ort
                from onnxruntime.quantization import quantize_dynamic, QuantType
            except ImportError:
                logger.error("优化ONNX模型需要onnx和onnxruntime库")
                return model_path
            
            # 加载模型
            onnx_model = onnx.load(model_path)
            
            # 检查和优化模型
            try:
                # 检查模型
                onnx.checker.check_model(onnx_model)
                
                # 优化模型
                from onnxruntime.transformers import optimizer
                optimized_model = optimizer.optimize_model(
                    model_path,
                    model_type='yolo',
                    num_heads=8,
                    hidden_size=768
                )
                
                # 保存优化后的模型
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                model_name = os.path.basename(model_path).split('.')[0]
                optimized_path = os.path.join(self.output_dir, f"{model_name}_optimized_{timestamp}.onnx")
                optimized_model.save_model_to_file(optimized_path)
                
                logger.info(f"ONNX模型优化完成: {optimized_path}")
                
                # 量化模型（如果需要）
                if kwargs.get('quantize', False):
                    quantized_path = os.path.join(self.output_dir, f"{model_name}_quantized_{timestamp}.onnx")
                    quantize_dynamic(optimized_path, quantized_path, weight_type=QuantType.QInt8)
                    logger.info(f"ONNX模型量化完成: {quantized_path}")
                    return quantized_path
                
                return optimized_path
            except Exception as e:
                logger.error(f"ONNX模型优化失败: {e}")
                return model_path
        except Exception as e:
            logger.error(f"ONNX模型优化失败: {e}")
            return model_path
    
    def _optimize_tflite(self, model_path: str, **kwargs) -> str:
        """
        优化TFLite模型
        
        参数:
            model_path: 模型路径
            **kwargs: 其他优化参数
            
        返回:
            优化后的模型路径
        """
        try:
            # 尝试导入tensorflow
            try:
                import tensorflow as tf
            except ImportError:
                logger.error("优化TFLite模型需要tensorflow库")
                return model_path
            
            # 加载模型
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
            
            # 设置优化选项
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # 量化（如果需要）
            if kwargs.get('quantize', False):
                converter.target_spec.supported_types = [tf.int8]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            
            # 转换模型
            tflite_model = converter.convert()
            
            # 保存优化后的模型
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(model_path).split('.')[0]
            optimized_path = os.path.join(self.output_dir, f"{model_name}_optimized_{timestamp}.tflite")
            
            with open(optimized_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite模型优化完成: {optimized_path}")
            
            return optimized_path
        except Exception as e:
            logger.error(f"TFLite模型优化失败: {e}")
            return model_path
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        参数:
            model_path: 模型路径
            
        返回:
            模型信息字典
        """
        try:
            logger.info(f"获取模型信息: {model_path}")
            
            # 加载模型
            model = YOLO(model_path)
            
            # 获取模型信息
            info = {
                'model_path': model_path,
                'model_name': os.path.basename(model_path),
                'model_type': model.type if hasattr(model, 'type') else 'unknown',
                'task': model.task if hasattr(model, 'task') else 'unknown',
                'num_classes': len(model.names) if hasattr(model, 'names') else 0,
                'class_names': model.names if hasattr(model, 'names') else {},
                'input_size': model.model.args['imgsz'] if hasattr(model, 'model') and hasattr(model.model, 'args') else [640, 640],
                'file_size': os.path.getsize(model_path) / (1024 * 1024),  # MB
                'file_date': time.ctime(os.path.getmtime(model_path)),
            }
            
            logger.info(f"模型信息获取成功: {model_path}")
            
            return info
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {'error': str(e)}
    
    def compare_model_sizes(self, model_paths: Dict[str, str]) -> Dict[str, Any]:
        """
        比较不同格式模型的大小
        
        参数:
            model_paths: 格式到模型路径的映射字典
            
        返回:
            比较结果字典
        """
        try:
            logger.info(f"比较模型大小: {list(model_paths.keys())}")
            
            results = {}
            for format, path in model_paths.items():
                if os.path.exists(path):
                    if os.path.isdir(path):
                        # 计算目录大小
                        total_size = 0
                        for dirpath, dirnames, filenames in os.walk(path):
                            for f in filenames:
                                fp = os.path.join(dirpath, f)
                                total_size += os.path.getsize(fp)
                        size_mb = total_size / (1024 * 1024)
                    else:
                        # 计算文件大小
                        size_mb = os.path.getsize(path) / (1024 * 1024)
                    
                    results[format] = {
                        'path': path,
                        'size_mb': size_mb,
                    }
                else:
                    results[format] = {
                        'path': path,
                        'error': '文件不存在',
                    }
            
            # 添加比较结果
            if results:
                # 找出基准格式（通常是原始的PT格式）
                base_format = 'pt' if 'pt' in results else list(results.keys())[0]
                base_size = results[base_format].get('size_mb', 0)
                
                for format, info in results.items():
                    if 'size_mb' in info and base_size > 0:
                        info['size_ratio'] = info['size_mb'] / base_size
                        info['size_reduction'] = (1 - info['size_mb'] / base_size) * 100 if info['size_mb'] < base_size else 0
            
            logger.info(f"模型大小比较完成")
            
            return results
        except Exception as e:
            logger.error(f"比较模型大小失败: {e}")
            return {'error': str(e)}