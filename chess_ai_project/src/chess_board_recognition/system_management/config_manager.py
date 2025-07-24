"""
配置管理模块

该模块提供了用于管理配置的类和函数。
"""

import os
import yaml
import json
from typing import Dict, Any, Optional

from .logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

class ConfigManager:
    """
    配置管理器类
    
    该类用于加载、验证和保存配置。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path or 'chess_ai_project/configs/chess_board_recognition.yaml'
        self.default_config_path = 'chess_ai_project/configs/default.yaml'
        self.config = {}
        
        # 加载配置
        self.load_config()
        
        logger.info(f"配置管理器初始化完成，配置文件: {self.config_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置
        
        返回:
            配置字典
        """
        try:
            # 加载默认配置
            default_config = {}
            if os.path.exists(self.default_config_path):
                with open(self.default_config_path, 'r', encoding='utf-8') as f:
                    default_config = yaml.safe_load(f)
                logger.info(f"已加载默认配置: {self.default_config_path}")
            
            # 加载用户配置
            user_config = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                logger.info(f"已加载用户配置: {self.config_path}")
            
            # 合并配置
            self.config = self._merge_configs(default_config, user_config)
            
            return self.config
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return {}
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取配置
        
        返回:
            配置字典
        """
        return self.config
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        保存配置
        
        参数:
            config: 配置字典
            
        返回:
            是否保存成功
        """
        try:
            # 验证配置
            if not self.validate_config(config):
                logger.error("配置验证失败，无法保存")
                return False
            
            # 保存配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置已保存: {self.config_path}")
            
            # 更新当前配置
            self.config = config
            
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置
        
        参数:
            config: 配置字典
            
        返回:
            是否有效
        """
        # 这里可以添加配置验证逻辑
        # 例如检查必需的键、值的类型和范围等
        
        # 简单的验证示例
        required_sections = ['model', 'training', 'data', 'classes']
        for section in required_sections:
            if section not in config:
                logger.error(f"配置缺少必需的部分: {section}")
                return False
        
        return True
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        返回:
            默认配置字典
        """
        try:
            if os.path.exists(self.default_config_path):
                with open(self.default_config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"默认配置文件不存在: {self.default_config_path}")
                return {}
        except Exception as e:
            logger.error(f"获取默认配置失败: {e}")
            return {}
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置
        
        参数:
            base_config: 基础配置
            override_config: 覆盖配置
            
        返回:
            合并后的配置
        """
        result = base_config.copy()
        
        for key, value in override_config.items():
            # 如果值是字典，则递归合并
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result