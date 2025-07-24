"""
训练监控模块

该模块提供了用于监控YOLO11训练进度和指标的类和函数。
"""

import os
import json
import time
import datetime
import threading
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..system_management.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

class TrainingMonitor:
    """
    训练监控器类
    
    该类用于监控和记录YOLO11训练的进度和指标。
    """
    
    def __init__(self, metrics_file: str, report_interval: int = 60):
        """
        初始化训练监控器
        
        参数:
            metrics_file: 指标文件路径
            report_interval: 报告间隔（秒）
        """
        self.metrics_file = metrics_file
        self.report_interval = report_interval
        self.metrics_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
        # 创建指标文件目录
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        logger.info(f"训练监控器初始化，指标文件: {metrics_file}")
    
    def start_monitoring(self, trainer) -> None:
        """
        开始监控训练
        
        参数:
            trainer: YOLO11Trainer实例
        """
        if self.is_monitoring:
            logger.warning("监控已经在运行")
            return
        
        self.trainer = trainer
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("训练监控已启动")
    
    def stop_monitoring(self) -> None:
        """
        停止监控训练
        """
        if not self.is_monitoring:
            logger.warning("监控未在运行")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("训练监控已停止")
    
    def _monitoring_loop(self) -> None:
        """
        监控循环
        """
        last_report_time = 0
        
        while self.is_monitoring:
            try:
                # 获取当前训练状态
                status = self.trainer.get_training_status()
                
                # 记录指标
                if status['is_training'] and status['metrics_history']:
                    latest_metrics = status['metrics_history'][-1]
                    self.metrics_history.append(latest_metrics)
                    
                    # 保存指标到文件
                    self._save_metrics(latest_metrics)
                    
                    # 定期报告
                    current_time = time.time()
                    if current_time - last_report_time >= self.report_interval:
                        self._report_progress(status)
                        last_report_time = current_time
                
                # 调用回调函数
                for callback in self.callbacks:
                    try:
                        callback(status)
                    except Exception as e:
                        logger.error(f"回调函数执行失败: {e}")
                
                # 如果训练已结束，停止监控
                if not status['is_training'] and status['current_epoch'] > 0:
                    logger.info("训练已完成，停止监控")
                    self.is_monitoring = False
                    break
                
                # 休眠一段时间
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(5.0)
    
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
    
    def _report_progress(self, status: Dict[str, Any]) -> None:
        """
        报告训练进度
        
        参数:
            status: 训练状态字典
        """
        if not status['is_training']:
            return
        
        current_epoch = status['current_epoch']
        total_epochs = status['total_epochs']
        progress = (current_epoch / total_epochs) * 100 if total_epochs > 0 else 0
        elapsed_time = status['elapsed_time']
        
        # 估算剩余时间
        if current_epoch > 0:
            time_per_epoch = elapsed_time / current_epoch
            remaining_epochs = total_epochs - current_epoch
            eta = time_per_epoch * remaining_epochs
            eta_str = str(datetime.timedelta(seconds=int(eta)))
        else:
            eta_str = "未知"
        
        # 获取最新指标
        if status['metrics_history']:
            latest_metrics = status['metrics_history'][-1]
            loss = latest_metrics.get('loss', 'N/A')
            map = latest_metrics.get('map', 'N/A')
            
            logger.info(f"训练进度: {current_epoch}/{total_epochs} ({progress:.1f}%), "
                        f"损失: {loss}, mAP: {map}, "
                        f"已用时间: {datetime.timedelta(seconds=int(elapsed_time))}, "
                        f"预计剩余时间: {eta_str}")
        else:
            logger.info(f"训练进度: {current_epoch}/{total_epochs} ({progress:.1f}%), "
                        f"已用时间: {datetime.timedelta(seconds=int(elapsed_time))}, "
                        f"预计剩余时间: {eta_str}")
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        添加监控回调函数
        
        参数:
            callback: 回调函数，接收训练状态字典作为参数
        """
        self.callbacks.append(callback)
        logger.info(f"已添加监控回调函数: {callback.__name__}")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        获取指标历史记录
        
        返回:
            指标历史记录列表
        """
        return self.metrics_history
    
    def plot_metrics(self, save_path: Optional[str] = None) -> Optional[Figure]:
        """
        绘制训练指标图表
        
        参数:
            save_path: 图表保存路径，如果为None则返回Figure对象
            
        返回:
            如果save_path为None，则返回Figure对象，否则返回None
        """
        try:
            if not self.metrics_history:
                logger.warning("没有训练指标可供绘制")
                return None
            
            # 提取指标
            epochs = []
            losses = []
            maps = []
            
            for metrics in self.metrics_history:
                if isinstance(metrics, dict):
                    epoch = metrics.get('epoch')
                    loss = metrics.get('loss')
                    map_val = metrics.get('map')
                    
                    if epoch is not None:
                        epochs.append(epoch)
                        if loss is not None:
                            losses.append(loss)
                        if map_val is not None:
                            maps.append(map_val)
            
            if not epochs:
                logger.warning("没有有效的轮次数据")
                return None
            
            # 创建图表
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 绘制损失曲线
            if losses:
                axes[0].plot(epochs[:len(losses)], losses, 'b-', label='训练损失')
                axes[0].set_title('训练损失曲线')
                axes[0].set_xlabel('轮次')
                axes[0].set_ylabel('损失')
                axes[0].grid(True)
                axes[0].legend()
            
            # 绘制mAP曲线
            if maps:
                axes[1].plot(epochs[:len(maps)], maps, 'r-', label='mAP')
                axes[1].set_title('mAP曲线')
                axes[1].set_xlabel('轮次')
                axes[1].set_ylabel('mAP')
                axes[1].grid(True)
                axes[1].legend()
            
            plt.tight_layout()
            
            # 保存或返回图表
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"指标图表已保存至: {save_path}")
                plt.close(fig)
                return None
            else:
                return fig
                
        except ImportError:
            logger.error("绘制指标图表需要matplotlib库")
            return None
        except Exception as e:
            logger.error(f"绘制指标图表失败: {e}")
            return None
    
    def generate_training_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成训练报告
        
        参数:
            save_path: 报告保存路径，如果为None则不保存
            
        返回:
            训练报告字典
        """
        try:
            if not self.metrics_history:
                logger.warning("没有训练指标可供生成报告")
                return {"error": "没有训练指标"}
            
            # 提取指标
            epochs = []
            losses = []
            maps = []
            
            for metrics in self.metrics_history:
                if isinstance(metrics, dict):
                    epoch = metrics.get('epoch')
                    loss = metrics.get('loss')
                    map_val = metrics.get('map')
                    
                    if epoch is not None:
                        epochs.append(epoch)
                        if loss is not None:
                            losses.append(loss)
                        if map_val is not None:
                            maps.append(map_val)
            
            # 计算统计信息
            report = {
                "总轮次": max(epochs) if epochs else 0,
                "开始时间": self.metrics_history[0].get('timestamp') if self.metrics_history else None,
                "结束时间": self.metrics_history[-1].get('timestamp') if self.metrics_history else None,
                "训练时长（秒）": self.metrics_history[-1].get('elapsed_time') if self.metrics_history else None,
            }
            
            if losses:
                report.update({
                    "初始损失": losses[0],
                    "最终损失": losses[-1],
                    "最小损失": min(losses),
                    "平均损失": sum(losses) / len(losses),
                })
            
            if maps:
                report.update({
                    "初始mAP": maps[0],
                    "最终mAP": maps[-1],
                    "最大mAP": max(maps),
                    "平均mAP": sum(maps) / len(maps),
                })
            
            # 保存报告
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                logger.info(f"训练报告已保存至: {save_path}")
            
            return report
        except Exception as e:
            logger.error(f"生成训练报告失败: {e}")
            return {"error": str(e)}