"""
日志系统模块

提供统一的日志记录功能，支持文件和控制台输出。
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str = "chess_board_recognition",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size: str = "10MB",
    backup_count: int = 5,
    format_string: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则不写入文件
        max_size: 日志文件最大大小，支持 KB, MB, GB 单位
        backup_count: 备份文件数量
        format_string: 自定义日志格式
        console_output: 是否输出到控制台
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有的处理器，避免重复添加
    logger.handlers.clear()
    
    # 设置日志格式
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # 添加控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        try:
            # 创建日志目录
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 解析文件大小
            max_bytes = _parse_size(max_size)
            
            # 创建旋转文件处理器
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.warning(f"无法设置文件日志处理器: {e}")
    
    return logger


def _parse_size(size_str: str) -> int:
    """
    解析大小字符串为字节数
    
    Args:
        size_str: 大小字符串，如 "10MB", "1GB"
        
    Returns:
        字节数
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # 假设是字节数
        return int(size_str)


class LoggerMixin:
    """日志记录器混入类，为其他类提供日志功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
    
    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        if self._logger is None:
            class_name = self.__class__.__name__
            self._logger = logging.getLogger(f"chess_board_recognition.{class_name}")
        return self._logger
    
    def log_info(self, message: str, **kwargs) -> None:
        """记录信息日志"""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """记录警告日志"""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs) -> None:
        """记录错误日志"""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs) -> None:
        """记录调试日志"""
        self.logger.debug(message, **kwargs)
    
    def log_exception(self, message: str, **kwargs) -> None:
        """记录异常日志"""
        self.logger.exception(message, **kwargs)


def configure_logging_from_config(config: Dict[str, Any]) -> logging.Logger:
    """
    从配置字典设置日志
    
    Args:
        config: 包含日志配置的字典
        
    Returns:
        配置好的日志记录器
    """
    logging_config = config.get('logging', {})
    
    return setup_logger(
        level=logging_config.get('level', 'INFO'),
        log_file=logging_config.get('file'),
        max_size=logging_config.get('max_size', '10MB'),
        backup_count=logging_config.get('backup_count', 5),
        console_output=logging_config.get('console_output', True)
    )


# 创建默认日志记录器
default_logger = setup_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称，如果为None则返回默认记录器
        
    Returns:
        日志记录器
    """
    if name is None:
        return default_logger
    return logging.getLogger(name)