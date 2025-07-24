"""
日志模块

该模块提供了用于设置和管理日志的函数。
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any

def setup_logger(name: str, config: Dict[str, Any] = None) -> logging.Logger:
    """
    设置日志记录器
    
    参数:
        name: 日志记录器名称
        config: 日志配置字典
        
    返回:
        日志记录器
    """
    # 默认配置
    default_config = {
        'level': 'INFO',
        'file': './logs/chess_board_recognition.log',
        'max_size': '10MB',
        'backup_count': 5,
        'console_output': True,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    }
    
    # 合并配置
    config = config or {}
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 如果已经设置了处理器，则不重复设置
    if logger.handlers:
        return logger
    
    # 设置日志级别
    level = getattr(logging, config['level'].upper(), logging.INFO)
    logger.setLevel(level)
    
    # 创建格式化器
    formatter = logging.Formatter(config['format'])
    
    # 添加控制台处理器
    if config['console_output']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件处理器
    if config['file']:
        # 创建日志目录
        log_dir = os.path.dirname(config['file'])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 解析最大大小
        max_size_str = config['max_size']
        if isinstance(max_size_str, str):
            if max_size_str.endswith('KB'):
                max_size = int(max_size_str[:-2]) * 1024
            elif max_size_str.endswith('MB'):
                max_size = int(max_size_str[:-2]) * 1024 * 1024
            elif max_size_str.endswith('GB'):
                max_size = int(max_size_str[:-2]) * 1024 * 1024 * 1024
            else:
                max_size = int(max_size_str)
        else:
            max_size = max_size_str
        
        # 创建文件处理器
        file_handler = RotatingFileHandler(
            config['file'],
            maxBytes=max_size,
            backupCount=config['backup_count'],
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger