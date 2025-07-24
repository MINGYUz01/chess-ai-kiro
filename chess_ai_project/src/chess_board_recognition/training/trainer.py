"""
YOLO11训练器模块

该模块提供了用于训练YOLO11模型的类和函数，包括训练参数配置、训练进度监控和模型权重保存。
"""

import os
import yaml
import json
import logging
import time
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

import torch
import numpy as np
from ultralytics import YOLO

from ..system_management.config_manager import ConfigManager
from ..system_management.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

class TrainingCallback:
    """
    训练回调类，用于监控训练进度和指标
    """
    
    def __init__(self, trainer):
        """
        初始化训练回调
        
        参数:
            trainer: YOLO11Trainer实例
        """
        self.trainer = trainer
        self.metrics_file = self.trainer.metrics_file
        
    def on_train_start(self, trainer):
        """
        训练开始时的回调
        """
        logger.info("训练开始")
        self.trainer.is_training = True
        self.trainer.start_time = time.time()
        self.trainer.current_epoch = 0
        self.trainer.metrics_history = []
        
        # 创建指标文件目录
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        
        # 记录初始训练信息
        self._save_metrics({
            'status': 'started',
            'timestamp': datetime.datetime.now().isoformat(),
            'config': {
                'epochs': self.trainer.epochs,
                'batch_size': self.trainer.batch_size,
                'learning_rate': self.trainer.learning_rate,
                'image_size': self.trainer.image_size,
                'device': self.trainer.device,
            }
        })
    
    def on_train_epoch_end(self, trainer):
        """
        每个训练轮次结束时的回调
        """
        # 更新当前轮次
        self.trainer.current_epoch += 1
        
        # 获取当前指标
        if hasattr(trainer, 'metrics') and trainer.metrics:
            metrics = trainer.metrics.copy()
            
            # 添加时间戳和轮次信息
            metrics['epoch'] = self.trainer.current_epoch
            metrics['timestamp'] = datetime.datetime.now().isoformat()
            metrics['elapsed_time'] = time.time() - self.trainer.start_time
            
            # 更新最佳指标
            if 'fitness' in metrics and metrics['fitness'] < self.trainer.best_metric:
                self.trainer.best_metric = metrics['fitness']
                metrics['is_best'] = True
            else:
                metrics['is_best'] = False
            
            # 保存到历史记录
            self.trainer.metrics_history.append(metrics)
            
            # 保存到文件
            self._save_metrics(metrics)
            
            # 记录日志
            logger.info(f"轮次 {self.trainer.current_epoch}/{self.trainer.epochs} 完成，"
                        f"损失: {metrics.get('loss', 'N/A'):.4f}, "
                        f"mAP: {metrics.get('map', 'N/A'):.4f}")
    
    def on_train_end(self, trainer):
        """
        训练结束时的回调
        """
        self.trainer.is_training = False
        training_time = time.time() - self.trainer.start_time
        
        # 记录训练结束信息
        final_metrics = {
            'status': 'completed',
            'timestamp': datetime.datetime.now().isoformat(),
            'total_epochs': self.trainer.current_epoch,
            'total_time': training_time,
            'best_metric': self.trainer.best_metric,
        }
        
        # 保存到文件
        self._save_metrics(final_metrics)
        
        logger.info(f"训练结束，总轮次: {self.trainer.current_epoch}，"
                    f"总时间: {training_time:.2f}秒，"
                    f"最佳指标: {self.trainer.best_metric:.4f}")
    
    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        保存指标到文件
        
        参数:
            metrics: 指标字典
        """
        try:
            # 读取现有指标
            existing_metrics = []
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
            
            # 添加新指标
            existing_metrics.append(metrics)
            
            # 保存指标
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metrics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存指标失败: {e}")


class YOLO11Trainer:
    """
    YOLO11模型训练器类
    
    该类封装了YOLO11模型的训练功能，包括参数配置、训练过程监控和模型权重保存。
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化YOLO11训练器
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.training_config = self.config.get('training', {})
        
        # 设置训练参数
        self.epochs = self.training_config.get('epochs', 100)
        self.batch_size = self.training_config.get('batch_size', 16)
        self.learning_rate = self.training_config.get('learning_rate', 0.001)
        self.image_size = self.training_config.get('image_size', 640)
        self.device = self.training_config.get('device', 'auto')
        self.workers = self.training_config.get('workers', 4)
        self.patience = self.training_config.get('patience', 50)
        self.save_period = self.training_config.get('save_period', 10)
        
        # 模型和结果
        self.model = None
        self.results = None
        self.model_path = self.config.get('model', {}).get('path', './models/best.pt')
        
        # 训练状态
        self.is_training = False
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.start_time = None
        self.metrics_history = []
        
        # 指标记录
        monitoring_config = self.config.get('monitoring', {})
        self.metrics_file = monitoring_config.get('metrics_file', './logs/training_metrics.json')
        
        # 回调
        self.callbacks = None
        
        logger.info(f"YOLO11训练器初始化完成，配置: {self.training_config}")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        加载YOLO11模型
        
        参数:
            model_path: 模型路径，如果为None则使用配置中的路径
        """
        try:
            path = model_path or self.model_path
            logger.info(f"正在加载模型: {path}")
            
            if not os.path.exists(path) and not path.startswith('yolo11'):
                logger.warning(f"模型文件不存在: {path}，将使用预训练模型")
                path = "yolo11n.pt"
                
            self.model = YOLO(path)
            logger.info(f"模型加载成功: {path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def prepare_data_config(self, data_yaml_path: str) -> None:
        """
        准备数据配置文件
        
        参数:
            data_yaml_path: 数据配置文件路径
        """
        try:
            if not os.path.exists(data_yaml_path):
                logger.error(f"数据配置文件不存在: {data_yaml_path}")
                raise FileNotFoundError(f"数据配置文件不存在: {data_yaml_path}")
            
            # 验证数据配置文件格式
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            required_keys = ['train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in data_config:
                    logger.error(f"数据配置文件缺少必要的键: {key}")
                    raise ValueError(f"数据配置文件缺少必要的键: {key}")
            
            logger.info(f"数据配置文件验证成功: {data_yaml_path}")
        except Exception as e:
            logger.error(f"准备数据配置文件失败: {e}")
            raise
    
    def _setup_callbacks(self) -> None:
        """
        设置训练回调
        """
        # 创建回调实例
        self.callbacks = TrainingCallback(self)
        
        # 注册回调
        try:
            # 尝试使用新版本的回调API
            if hasattr(self.model, 'add_callback'):
                self.model.add_callback("on_train_start", self.callbacks.on_train_start)
                self.model.add_callback("on_train_epoch_end", self.callbacks.on_train_epoch_end)
                self.model.add_callback("on_train_end", self.callbacks.on_train_end)
            # 尝试使用旧版本的回调API
            elif hasattr(self.model, 'callbacks') and hasattr(self.model.callbacks, 'register_action'):
                self.model.callbacks.register_action('on_train_start', self.callbacks.on_train_start)
                self.model.callbacks.register_action('on_train_epoch_end', self.callbacks.on_train_epoch_end)
                self.model.callbacks.register_action('on_train_end', self.callbacks.on_train_end)
            # 如果都不支持，使用自定义的监控方式
            else:
                logger.warning("模型不支持回调，将使用自定义监控方式")
                # 在这里可以实现自定义的监控方式，例如定期检查训练状态
        except Exception as e:
            logger.warning(f"设置回调失败: {e}，将无法监控训练进度")
    
    def train(self, data_yaml_path: str, **kwargs) -> Dict[str, Any]:
        """
        训练YOLO11模型
        
        参数:
            data_yaml_path: 数据配置文件路径
            **kwargs: 其他训练参数，将覆盖配置文件中的参数
            
        返回:
            训练结果字典
        """
        try:
            if self.model is None:
                self.load_model()
            
            self.prepare_data_config(data_yaml_path)
            
            # 设置回调
            self._setup_callbacks()
            
            # 合并训练参数
            train_args = {
                'data': data_yaml_path,
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.image_size,
                'device': self.device,
                'workers': self.workers,
                'patience': self.patience,
                'save_period': self.save_period,
                'lr0': self.learning_rate,
                'project': os.path.dirname(self.model_path),
                'name': f'train_{time.strftime("%Y%m%d_%H%M%S")}',
                'exist_ok': True,
            }
            
            # 更新用户提供的参数
            train_args.update(kwargs)
            
            logger.info(f"开始训练模型，参数: {train_args}")
            self.is_training = True
            self.start_time = time.time()
            
            # 执行训练
            self.results = self.model.train(**train_args)
            
            self.is_training = False
            training_time = time.time() - self.start_time
            logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
            
            # 保存最佳模型
            self._save_best_model(train_args)
            
            return self.results
        except Exception as e:
            self.is_training = False
            logger.error(f"模型训练失败: {e}")
            raise
    
    def _save_best_model(self, train_args: Dict[str, Any]) -> None:
        """
        保存最佳模型
        
        参数:
            train_args: 训练参数
        """
        try:
            # 获取最佳模型路径
            best_model_path = os.path.join(train_args['project'], train_args['name'], 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                # 创建模型目录
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                
                # 复制最佳模型
                import shutil
                shutil.copy(best_model_path, self.model_path)
                logger.info(f"最佳模型已保存至: {self.model_path}")
                
                # 创建模型版本备份
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(os.path.dirname(self.model_path), f"best_{timestamp}.pt")
                shutil.copy(best_model_path, backup_path)
                logger.info(f"模型版本备份已保存至: {backup_path}")
                
                # 保存模型元数据
                metadata = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'training_args': train_args,
                    'metrics': self.metrics_history[-1] if self.metrics_history else {},
                    'model_path': self.model_path,
                    'backup_path': backup_path,
                }
                
                metadata_path = os.path.join(os.path.dirname(self.model_path), f"metadata_{timestamp}.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                logger.info(f"模型元数据已保存至: {metadata_path}")
            else:
                logger.warning(f"未找到最佳模型: {best_model_path}")
        except Exception as e:
            logger.error(f"保存最佳模型失败: {e}")
    
    def resume_training(self, checkpoint_path: str, **kwargs) -> Dict[str, Any]:
        """
        从检查点恢复训练
        
        参数:
            checkpoint_path: 检查点路径
            **kwargs: 其他训练参数
            
        返回:
            训练结果字典
        """
        try:
            logger.info(f"从检查点恢复训练: {checkpoint_path}")
            
            # 加载检查点模型
            self.model = YOLO(checkpoint_path)
            
            # 设置回调
            self._setup_callbacks()
            
            # 设置恢复训练参数
            train_args = {
                'resume': True,
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.image_size,
                'device': self.device,
                'workers': self.workers,
                'patience': self.patience,
                'save_period': self.save_period,
            }
            
            # 更新用户提供的参数
            train_args.update(kwargs)
            
            logger.info(f"恢复训练，参数: {train_args}")
            self.is_training = True
            self.start_time = time.time()
            
            # 执行训练
            self.results = self.model.train(**train_args)
            
            self.is_training = False
            training_time = time.time() - self.start_time
            logger.info(f"恢复训练完成，耗时: {training_time:.2f}秒")
            
            # 保存最佳模型
            self._save_best_model(train_args)
            
            return self.results
        except Exception as e:
            self.is_training = False
            logger.error(f"恢复训练失败: {e}")
            raise
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        获取训练状态
        
        返回:
            训练状态字典
        """
        status = {
            'is_training': self.is_training,
            'current_epoch': self.current_epoch,
            'total_epochs': self.epochs,
            'best_metric': self.best_metric,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'metrics_history': self.metrics_history,
            'progress_percentage': (self.current_epoch / self.epochs) * 100 if self.epochs > 0 else 0,
        }
        return status
    
    def stop_training(self) -> None:
        """
        停止训练
        """
        if self.is_training:
            logger.info("正在停止训练...")
            # 在实际实现中，需要一种机制来安全地停止训练
            # 这里只是设置标志位，实际上需要更复杂的实现
            self.is_training = False
            logger.info("训练已停止")
        else:
            logger.warning("没有正在进行的训练")
    
    def validate(self, data_yaml_path: str = None, **kwargs) -> Dict[str, Any]:
        """
        验证模型
        
        参数:
            data_yaml_path: 数据配置文件路径，如果为None则使用训练时的数据
            **kwargs: 其他验证参数
            
        返回:
            验证结果字典
        """
        try:
            if self.model is None:
                self.load_model()
            
            val_args = {
                'data': data_yaml_path,
                'batch': self.batch_size,
                'imgsz': self.image_size,
                'device': self.device,
            }
            
            # 更新用户提供的参数
            val_args.update(kwargs)
            
            logger.info(f"开始验证模型，参数: {val_args}")
            
            # 执行验证
            results = self.model.val(**val_args)
            
            logger.info(f"模型验证完成")
            return results
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            raise
    
    def export_model(self, format: str = 'onnx', **kwargs) -> str:
        """
        导出模型为不同格式
        
        参数:
            format: 导出格式，支持'onnx', 'torchscript', 'openvino'等
            **kwargs: 其他导出参数
            
        返回:
            导出的模型路径
        """
        try:
            if self.model is None:
                self.load_model()
            
            logger.info(f"开始导出模型为{format}格式")
            
            # 设置导出参数
            export_args = {
                'format': format,
                'imgsz': self.image_size,
            }
            
            # 更新用户提供的参数
            export_args.update(kwargs)
            
            # 执行导出
            exported_path = self.model.export(**export_args)
            
            logger.info(f"模型导出完成: {exported_path}")
            return exported_path
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            raise
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """
        获取训练指标
        
        返回:
            训练指标字典
        """
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                return metrics
            else:
                logger.warning(f"指标文件不存在: {self.metrics_file}")
                return {'metrics': self.metrics_history}
        except Exception as e:
            logger.error(f"获取训练指标失败: {e}")
            return {'error': str(e), 'metrics': self.metrics_history}
    
    def plot_metrics(self, save_path: str = None) -> None:
        """
        绘制训练指标图表
        
        参数:
            save_path: 图表保存路径，如果为None则显示图表
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.metrics_history:
                logger.warning("没有训练指标可供绘制")
                return
            
            # 提取指标
            epochs = [m.get('epoch', i+1) for i, m in enumerate(self.metrics_history) if isinstance(m, dict)]
            losses = [m.get('loss', None) for m in self.metrics_history if isinstance(m, dict)]
            maps = [m.get('map', None) for m in self.metrics_history if isinstance(m, dict)]
            
            # 过滤None值
            epochs_loss = [e for e, l in zip(epochs, losses) if l is not None]
            losses = [l for l in losses if l is not None]
            
            epochs_map = [e for e, m in zip(epochs, maps) if m is not None]
            maps = [m for m in maps if m is not None]
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            
            # 绘制损失曲线
            if losses:
                plt.subplot(1, 2, 1)
                plt.plot(epochs_loss, losses, 'b-', label='训练损失')
                plt.title('训练损失曲线')
                plt.xlabel('轮次')
                plt.ylabel('损失')
                plt.grid(True)
                plt.legend()
            
            # 绘制mAP曲线
            if maps:
                plt.subplot(1, 2, 2)
                plt.plot(epochs_map, maps, 'r-', label='mAP')
                plt.title('mAP曲线')
                plt.xlabel('轮次')
                plt.ylabel('mAP')
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            
            # 保存或显示图表
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"指标图表已保存至: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.error("绘制指标图表需要matplotlib库")
        except Exception as e:
            logger.error(f"绘制指标图表失败: {e}")