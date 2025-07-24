"""
日志系统

提供统一的日志记录功能。
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'chess_ai',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    log_dir: str = 'logs/chinese_chess_ai_engine',
    max_size: int = 10,  # MB
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件名
        log_dir: 日志目录
        max_size: 日志文件最大大小(MB)
        backup_count: 备份文件数量
        console_output: 是否输出到控制台
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # 文件路径
        file_path = log_path / log_file
        
        # 使用RotatingFileHandler实现日志轮转
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=max_size * 1024 * 1024,  # 转换为字节
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'chess_ai') -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    日志记录器混入类
    
    为类提供日志记录功能。
    """
    
    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        class_name = self.__class__.__name__
        return get_logger(f'chess_ai.{class_name}')
    
    def log_info(self, message: str, *args, **kwargs):
        """记录信息日志"""
        self.logger.info(message, *args, **kwargs)
    
    def log_warning(self, message: str, *args, **kwargs):
        """记录警告日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def log_error(self, message: str, *args, **kwargs):
        """记录错误日志"""
        self.logger.error(message, *args, **kwargs)
    
    def log_debug(self, message: str, *args, **kwargs):
        """记录调试日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def log_exception(self, message: str, *args, **kwargs):
        """记录异常日志"""
        self.logger.exception(message, *args, **kwargs)


class PerformanceLogger:
    """
    性能日志记录器
    
    用于记录性能相关的指标。
    """
    
    def __init__(self, name: str = 'performance'):
        self.logger = get_logger(f'chess_ai.{name}')
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """开始计时"""
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"开始计时: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """结束计时并返回耗时"""
        if operation not in self.start_times:
            self.logger.warning(f"未找到计时器: {operation}")
            return 0.0
        
        start_time = self.start_times.pop(operation)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"操作完成: {operation}, 耗时: {elapsed:.3f}秒")
        return elapsed
    
    def log_metric(self, metric_name: str, value: float, unit: str = ""):
        """记录性能指标"""
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"性能指标 - {metric_name}: {value:.3f}{unit_str}")
    
    def log_memory_usage(self, operation: str, memory_mb: float):
        """记录内存使用情况"""
        self.logger.info(f"内存使用 - {operation}: {memory_mb:.2f} MB")
    
    def log_search_stats(self, simulations: int, time_used: float, nodes_per_second: float):
        """记录搜索统计信息"""
        self.logger.info(
            f"搜索统计 - 模拟次数: {simulations}, "
            f"耗时: {time_used:.3f}秒, "
            f"速度: {nodes_per_second:.0f} nodes/sec"
        )
    
    def log_training_stats(self, epoch: int, loss: float, accuracy: float, lr: float):
        """记录训练统计信息"""
        self.logger.info(
            f"训练统计 - Epoch: {epoch}, "
            f"Loss: {loss:.6f}, "
            f"Accuracy: {accuracy:.4f}, "
            f"LR: {lr:.6f}"
        )


# 全局性能日志记录器实例
performance_logger = PerformanceLogger()