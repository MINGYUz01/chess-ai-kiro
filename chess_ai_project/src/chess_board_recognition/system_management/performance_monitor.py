"""
性能监控模块

该模块提供了用于监控系统性能的类和函数。
"""

import os
import time
import json
import logging
import psutil
import threading
from typing import Dict, Any, List, Optional

from .logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

class PerformanceMonitorImpl:
    """
    性能监控器实现类
    
    该类用于监控系统性能，包括CPU使用率、内存使用量、GPU使用率等。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化性能监控器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.metrics_file = self.config.get('metrics_file', './logs/performance_metrics.json')
        self.report_interval = self.config.get('report_interval', 60)  # 秒
        self.max_inference_time = self.config.get('max_inference_time', 100)  # 毫秒
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        
        # 创建指标文件目录
        if self.enabled:
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        
        logger.info(f"性能监控器初始化完成，启用状态: {self.enabled}")
    
    def start_monitoring(self) -> None:
        """
        开始监控系统性能
        """
        if not self.enabled:
            logger.warning("性能监控已禁用")
            return
        
        if self.is_monitoring:
            logger.warning("性能监控已经在运行")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("性能监控已启动")
    
    def stop_monitoring(self) -> None:
        """
        停止监控系统性能
        """
        if not self.is_monitoring:
            logger.warning("性能监控未在运行")
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("性能监控已停止")
    
    def _monitoring_loop(self) -> None:
        """
        监控循环
        """
        last_report_time = 0
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # 定期报告
                if current_time - last_report_time >= self.report_interval:
                    metrics = self._collect_metrics()
                    self._save_metrics(metrics)
                    self._report_metrics(metrics)
                    last_report_time = current_time
                
                # 休眠一段时间
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(5.0)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """
        收集性能指标
        
        返回:
            性能指标字典
        """
        metrics = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
            },
            'memory': {
                'total': psutil.virtual_memory().total / (1024 * 1024),  # MB
                'available': psutil.virtual_memory().available / (1024 * 1024),  # MB
                'used': psutil.virtual_memory().used / (1024 * 1024),  # MB
                'percent': psutil.virtual_memory().percent,
            },
            'disk': {
                'total': psutil.disk_usage('/').total / (1024 * 1024 * 1024),  # GB
                'used': psutil.disk_usage('/').used / (1024 * 1024 * 1024),  # GB
                'free': psutil.disk_usage('/').free / (1024 * 1024 * 1024),  # GB
                'percent': psutil.disk_usage('/').percent,
            },
        }
        
        # 尝试获取GPU信息
        try:
            import torch
            if torch.cuda.is_available():
                metrics['gpu'] = {
                    'count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(0),
                }
                
                # 尝试获取GPU使用率
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    metrics['gpu']['memory_total'] = info.total / (1024 * 1024)  # MB
                    metrics['gpu']['memory_used'] = info.used / (1024 * 1024)  # MB
                    metrics['gpu']['memory_free'] = info.free / (1024 * 1024)  # MB
                    metrics['gpu']['memory_percent'] = (info.used / info.total) * 100
                    pynvml.nvmlShutdown()
                except (ImportError, Exception) as e:
                    logger.debug(f"无法获取GPU使用率: {e}")
        except ImportError:
            logger.debug("无法导入torch，跳过GPU信息收集")
        
        return metrics
    
    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        保存性能指标到文件
        
        参数:
            metrics: 性能指标字典
        """
        try:
            # 添加到历史记录
            self.metrics_history.append(metrics)
            
            # 限制历史记录长度
            max_history = 1000
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
            
            # 保存到文件
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存性能指标失败: {e}")
    
    def _report_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        报告性能指标
        
        参数:
            metrics: 性能指标字典
        """
        cpu_usage = metrics['cpu']['usage_percent']
        memory_percent = metrics['memory']['percent']
        disk_percent = metrics['disk']['percent']
        
        logger.info(f"系统性能: CPU使用率: {cpu_usage:.1f}%, "
                    f"内存使用率: {memory_percent:.1f}%, "
                    f"磁盘使用率: {disk_percent:.1f}%")
        
        # 检查是否超过阈值
        if cpu_usage > 90:
            logger.warning(f"CPU使用率过高: {cpu_usage:.1f}%")
        
        if memory_percent > 90:
            logger.warning(f"内存使用率过高: {memory_percent:.1f}%")
        
        if disk_percent > 90:
            logger.warning(f"磁盘使用率过高: {disk_percent:.1f}%")
    
    def log_inference_time(self, time_ms: float) -> None:
        """
        记录推理时间
        
        参数:
            time_ms: 推理时间（毫秒）
        """
        if not self.enabled:
            return
        
        # 检查是否超过最大推理时间
        if time_ms > self.max_inference_time:
            logger.warning(f"推理时间过长: {time_ms:.2f}ms > {self.max_inference_time}ms")
        
        # 记录推理时间
        metrics = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'inference_time_ms': time_ms,
        }
        
        self._save_metrics(metrics)
    
    def log_accuracy_metrics(self, metrics: Dict[str, float]) -> None:
        """
        记录准确率指标
        
        参数:
            metrics: 准确率指标字典
        """
        if not self.enabled:
            return
        
        # 记录准确率指标
        metrics_with_timestamp = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'accuracy_metrics': metrics,
        }
        
        self._save_metrics(metrics_with_timestamp)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        生成性能报告
        
        返回:
            性能报告字典
        """
        if not self.metrics_history:
            return {'error': '没有性能指标数据'}
        
        # 提取CPU使用率
        cpu_usages = [m['cpu']['usage_percent'] for m in self.metrics_history if 'cpu' in m]
        
        # 提取内存使用率
        memory_percents = [m['memory']['percent'] for m in self.metrics_history if 'memory' in m]
        
        # 提取推理时间
        inference_times = [m['inference_time_ms'] for m in self.metrics_history if 'inference_time_ms' in m]
        
        # 计算统计信息
        report = {
            '开始时间': self.metrics_history[0]['timestamp'],
            '结束时间': self.metrics_history[-1]['timestamp'],
            '样本数': len(self.metrics_history),
            'CPU使用率': {
                '平均值': sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0,
                '最大值': max(cpu_usages) if cpu_usages else 0,
                '最小值': min(cpu_usages) if cpu_usages else 0,
            },
            '内存使用率': {
                '平均值': sum(memory_percents) / len(memory_percents) if memory_percents else 0,
                '最大值': max(memory_percents) if memory_percents else 0,
                '最小值': min(memory_percents) if memory_percents else 0,
            },
        }
        
        # 添加推理时间统计（如果有）
        if inference_times:
            report['推理时间(ms)'] = {
                '平均值': sum(inference_times) / len(inference_times),
                '最大值': max(inference_times),
                '最小值': min(inference_times),
                '超过阈值次数': sum(1 for t in inference_times if t > self.max_inference_time),
            }
        
        return report