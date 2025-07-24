"""
模型评估模块

该模块提供了用于评估YOLO11模型性能的类和函数。
"""

import os
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .trainer import YOLO11Trainer
from ..system_management.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

class ModelEvaluator:
    """
    模型评估器类
    
    该类用于评估YOLO11模型的性能和准确性。
    """
    
    def __init__(self, model_path: str = None, output_dir: str = './evaluation_results'):
        """
        初始化模型评估器
        
        参数:
            model_path: 模型路径，如果为None则使用默认模型
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.trainer = None
        self.evaluation_results = {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"模型评估器初始化完成，输出目录: {output_dir}")
    
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
            
            # 初始化训练器
            self.trainer = YOLO11Trainer()
            self.trainer.load_model(path)
            
            logger.info(f"模型加载成功: {path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def evaluate_on_dataset(self, data_yaml_path: str, **kwargs) -> Dict[str, Any]:
        """
        在数据集上评估模型
        
        参数:
            data_yaml_path: 数据配置文件路径
            **kwargs: 其他评估参数
            
        返回:
            评估结果字典
        """
        try:
            if not self.trainer:
                self.load_model()
            
            logger.info(f"开始在数据集上评估模型: {data_yaml_path}")
            
            # 设置评估参数
            val_args = {
                'data': data_yaml_path,
                'batch': kwargs.get('batch_size', 16),
                'imgsz': kwargs.get('image_size', 640),
                'device': kwargs.get('device', 'auto'),
                'verbose': kwargs.get('verbose', True),
                'save_json': kwargs.get('save_json', True),
                'save_hybrid': kwargs.get('save_hybrid', False),
                'conf': kwargs.get('conf_threshold', 0.25),
                'iou': kwargs.get('iou_threshold', 0.7),
                'max_det': kwargs.get('max_detections', 300),
                'half': kwargs.get('half_precision', False),
            }
            
            # 执行评估
            results = self.trainer.validate(**val_args)
            
            # 保存评估结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(self.output_dir, f'evaluation_{timestamp}.json')
            
            # 提取关键指标
            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            elif isinstance(results, dict):
                metrics = results
            
            # 保存结果
            self.evaluation_results = {
                'timestamp': timestamp,
                'model_path': self.model_path,
                'data_path': data_yaml_path,
                'metrics': metrics,
                'parameters': val_args,
            }
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"评估完成，结果已保存至: {result_path}")
            
            # 打印关键指标
            if 'metrics/mAP50-95(B)' in metrics:
                logger.info(f"mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
            if 'metrics/mAP50(B)' in metrics:
                logger.info(f"mAP50: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/precision(B)' in metrics:
                logger.info(f"精确率: {metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics:
                logger.info(f"召回率: {metrics['metrics/recall(B)']:.4f}")
            
            return self.evaluation_results
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            raise
    
    def evaluate_on_images(self, image_dir: str, ground_truth_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        在图像目录上评估模型
        
        参数:
            image_dir: 图像目录路径
            ground_truth_dir: 真实标注目录路径，如果为None则只进行推理
            **kwargs: 其他评估参数
            
        返回:
            评估结果字典
        """
        try:
            if not self.trainer:
                self.load_model()
            
            logger.info(f"开始在图像目录上评估模型: {image_dir}")
            
            # 设置推理参数
            pred_args = {
                'source': image_dir,
                'conf': kwargs.get('conf_threshold', 0.25),
                'iou': kwargs.get('iou_threshold', 0.7),
                'imgsz': kwargs.get('image_size', 640),
                'device': kwargs.get('device', 'auto'),
                'save': kwargs.get('save_results', True),
                'save_txt': kwargs.get('save_labels', True),
                'save_conf': kwargs.get('save_confidence', True),
                'save_crop': kwargs.get('save_crops', False),
                'project': self.output_dir,
                'name': f'predict_{time.strftime("%Y%m%d_%H%M%S")}',
                'exist_ok': True,
            }
            
            # 执行推理
            results = self.trainer.model.predict(**pred_args)
            
            # 如果有真实标注，计算评估指标
            metrics = {}
            if ground_truth_dir and os.path.exists(ground_truth_dir):
                # 这里需要实现自定义的评估逻辑，比较预测结果和真实标注
                # 由于这需要更复杂的实现，这里只是一个占位符
                logger.info(f"正在计算评估指标，真实标注目录: {ground_truth_dir}")
                metrics = self._calculate_metrics(results, ground_truth_dir)
            
            # 保存评估结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(self.output_dir, f'image_evaluation_{timestamp}.json')
            
            # 提取结果统计
            stats = {
                'total_images': len(results),
                'detected_objects': sum(len(r.boxes) for r in results if hasattr(r, 'boxes')),
                'average_confidence': np.mean([box.conf.mean().item() for r in results if hasattr(r, 'boxes') for box in [r.boxes] if len(box) > 0]) if results else 0,
            }
            
            # 保存结果
            self.evaluation_results = {
                'timestamp': timestamp,
                'model_path': self.model_path,
                'image_dir': image_dir,
                'ground_truth_dir': ground_truth_dir,
                'statistics': stats,
                'metrics': metrics,
                'parameters': pred_args,
            }
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"图像评估完成，结果已保存至: {result_path}")
            
            return self.evaluation_results
        except Exception as e:
            logger.error(f"图像评估失败: {e}")
            raise
    
    def _calculate_metrics(self, results, ground_truth_dir: str) -> Dict[str, float]:
        """
        计算评估指标
        
        参数:
            results: 预测结果
            ground_truth_dir: 真实标注目录路径
            
        返回:
            评估指标字典
        """
        # 这里需要实现自定义的评估逻辑，比较预测结果和真实标注
        # 由于这需要更复杂的实现，这里只是一个简单的示例
        try:
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'mAP50': 0.0,
                'mAP50-95': 0.0,
            }
            
            # TODO: 实现真实的指标计算逻辑
            
            return metrics
        except Exception as e:
            logger.error(f"计算评估指标失败: {e}")
            return {}
    
    def generate_evaluation_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成评估报告
        
        参数:
            save_path: 报告保存路径，如果为None则使用默认路径
            
        返回:
            评估报告字典
        """
        try:
            if not self.evaluation_results:
                logger.warning("没有评估结果可供生成报告")
                return {"error": "没有评估结果"}
            
            # 获取指标，可能在不同的键下
            metrics = self.evaluation_results.get('metrics', {})
            if not metrics and isinstance(self.evaluation_results, dict):
                # 如果metrics为空，尝试直接使用evaluation_results
                metrics = self.evaluation_results
            
            # 生成报告
            report = {
                "模型路径": self.model_path,
                "评估时间": self.evaluation_results.get('timestamp', time.strftime("%Y-%m-%d %H:%M:%S")),
                "评估数据": self.evaluation_results.get('data_path', self.evaluation_results.get('image_dir', 'N/A')),
                "评估指标": metrics,
                "统计信息": self.evaluation_results.get('statistics', {}),
                "评估参数": self.evaluation_results.get('parameters', {}),
            }
            
            # 保存报告
            if save_path:
                report_path = save_path
            else:
                report_path = os.path.join(self.output_dir, f'evaluation_report_{time.strftime("%Y%m%d_%H%M%S")}.json')
            
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"评估报告已保存至: {report_path}")
            
            return report
            
            return report
        except Exception as e:
            logger.error(f"生成评估报告失败: {e}")
            return {"error": str(e)}
    
    def plot_evaluation_results(self, save_path: Optional[str] = None) -> Optional[Figure]:
        """
        绘制评估结果图表
        
        参数:
            save_path: 图表保存路径，如果为None则返回Figure对象
            
        返回:
            如果save_path为None，则返回Figure对象，否则返回None
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.evaluation_results:
                logger.warning("没有评估指标可供绘制")
                return None
            
            # 获取指标，可能在不同的键下
            metrics = self.evaluation_results.get('metrics', {})
            if not metrics and isinstance(self.evaluation_results, dict):
                # 如果metrics为空，尝试直接使用evaluation_results
                metrics = self.evaluation_results
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 提取类别指标
            class_metrics = {}
            for key, value in metrics.items():
                if isinstance(key, str) and key.startswith('metrics/') and '(' in key and key.endswith(')'):
                    metric_name = key.split('/')[1].split('(')[0]
                    class_name = key.split('(')[1].split(')')[0]
                    
                    if class_name not in class_metrics:
                        class_metrics[class_name] = {}
                    
                    class_metrics[class_name][metric_name] = value
            
            # 绘制精确率和召回率
            if class_metrics:
                class_names = list(class_metrics.keys())
                precisions = [class_metrics[cls].get('precision', 0) for cls in class_names]
                recalls = [class_metrics[cls].get('recall', 0) for cls in class_names]
                
                # 绘制精确率
                axes[0, 0].bar(class_names, precisions)
                axes[0, 0].set_title('类别精确率')
                axes[0, 0].set_xlabel('类别')
                axes[0, 0].set_ylabel('精确率')
                axes[0, 0].set_ylim(0, 1)
                axes[0, 0].grid(True)
                plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
                
                # 绘制召回率
                axes[0, 1].bar(class_names, recalls)
                axes[0, 1].set_title('类别召回率')
                axes[0, 1].set_xlabel('类别')
                axes[0, 1].set_ylabel('召回率')
                axes[0, 1].set_ylim(0, 1)
                axes[0, 1].grid(True)
                plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
            else:
                # 如果没有类别指标，绘制简单的条形图
                metric_names = ['mAP50-95', 'mAP50', 'precision', 'recall']
                metric_values = [metrics.get(name, 0) for name in metric_names]
                
                axes[0, 0].bar(metric_names, metric_values)
                axes[0, 0].set_title('评估指标')
                axes[0, 0].set_xlabel('指标')
                axes[0, 0].set_ylabel('值')
                axes[0, 0].set_ylim(0, 1)
                axes[0, 0].grid(True)
                plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
            
            # 绘制混淆矩阵
            confusion_matrix = self.evaluation_results.get('confusion_matrix')
            if confusion_matrix:
                confusion_matrix = np.array(confusion_matrix)
                axes[1, 0].imshow(confusion_matrix, cmap='Blues')
                axes[1, 0].set_title('混淆矩阵')
                axes[1, 0].set_xlabel('预测类别')
                axes[1, 0].set_ylabel('真实类别')
                
                # 添加数值标签
                for i in range(confusion_matrix.shape[0]):
                    for j in range(confusion_matrix.shape[1]):
                        axes[1, 0].text(j, i, f'{confusion_matrix[i, j]:.0f}',
                                        ha='center', va='center', color='white' if confusion_matrix[i, j] > confusion_matrix.max() / 2 else 'black')
            
            # 绘制PR曲线
            pr_curve = self.evaluation_results.get('pr_curve')
            if pr_curve:
                for cls, curve in pr_curve.items():
                    precision = curve['precision']
                    recall = curve['recall']
                    axes[1, 1].plot(recall, precision, label=f'类别 {cls}')
                
                axes[1, 1].set_title('PR曲线')
                axes[1, 1].set_xlabel('召回率')
                axes[1, 1].set_ylabel('精确率')
                axes[1, 1].set_xlim(0, 1)
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].grid(True)
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # 保存或返回图表
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"评估结果图表已保存至: {save_path}")
                plt.close(fig)
                return None
            else:
                return fig
                
        except ImportError:
            logger.error("绘制评估结果需要matplotlib库")
            return None
        except Exception as e:
            logger.error(f"绘制评估结果失败: {e}")
            return None
    
    def compare_models(self, model_paths: List[str], data_yaml_path: str, **kwargs) -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        参数:
            model_paths: 模型路径列表
            data_yaml_path: 数据配置文件路径
            **kwargs: 其他评估参数
            
        返回:
            比较结果字典
        """
        try:
            logger.info(f"开始比较模型，模型数量: {len(model_paths)}")
            
            comparison_results = []
            
            # 评估每个模型
            for model_path in model_paths:
                logger.info(f"正在评估模型: {model_path}")
                
                # 加载模型
                self.model_path = model_path
                self.load_model()
                
                # 评估模型
                result = self.evaluate_on_dataset(data_yaml_path, **kwargs)
                
                # 提取关键指标
                model_result = {
                    'model_path': model_path,
                    'model_name': os.path.basename(model_path),
                    'metrics': result.get('metrics', {}),
                }
                
                comparison_results.append(model_result)
            
            # 保存比较结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            comparison_path = os.path.join(self.output_dir, f'model_comparison_{timestamp}.json')
            
            comparison = {
                'timestamp': timestamp,
                'data_path': data_yaml_path,
                'models': comparison_results,
                'parameters': kwargs,
            }
            
            with open(comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            
            logger.info(f"模型比较完成，结果已保存至: {comparison_path}")
            
            return comparison
        except Exception as e:
            logger.error(f"模型比较失败: {e}")
            raise
    
    def plot_model_comparison(self, comparison_results: Dict[str, Any], metrics: List[str] = None, save_path: Optional[str] = None) -> Optional[Figure]:
        """
        绘制模型比较图表
        
        参数:
            comparison_results: 比较结果字典
            metrics: 要比较的指标列表，如果为None则使用默认指标
            save_path: 图表保存路径，如果为None则返回Figure对象
            
        返回:
            如果save_path为None，则返回Figure对象，否则返回None
        """
        try:
            if not comparison_results or 'models' not in comparison_results:
                logger.warning("没有比较结果可供绘制")
                return None
            
            models = comparison_results['models']
            if not models:
                logger.warning("没有模型结果可供绘制")
                return None
            
            # 如果未指定指标，使用默认指标
            if not metrics:
                metrics = ['metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'metrics/precision(B)', 'metrics/recall(B)']
            
            # 提取模型名称和指标
            model_names = [model['model_name'] for model in models]
            metric_values = {metric: [] for metric in metrics}
            
            for model in models:
                model_metrics = model.get('metrics', {})
                for metric in metrics:
                    metric_values[metric].append(model_metrics.get(metric, 0))
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 设置柱状图的宽度和位置
            bar_width = 0.8 / len(metrics)
            index = np.arange(len(model_names))
            
            # 绘制每个指标的柱状图
            for i, metric in enumerate(metrics):
                metric_name = metric.split('/')[1].split('(')[0] if '/' in metric else metric
                ax.bar(index + i * bar_width, metric_values[metric], bar_width, label=metric_name)
            
            # 设置图表属性
            ax.set_xlabel('模型')
            ax.set_ylabel('指标值')
            ax.set_title('模型性能比较')
            ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(model_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend()
            ax.grid(True, axis='y')
            
            plt.tight_layout()
            
            # 保存或返回图表
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"模型比较图表已保存至: {save_path}")
                plt.close(fig)
                return None
            else:
                return fig
                
        except ImportError:
            logger.error("绘制模型比较需要matplotlib库")
            return None
        except Exception as e:
            logger.error(f"绘制模型比较失败: {e}")
            return None