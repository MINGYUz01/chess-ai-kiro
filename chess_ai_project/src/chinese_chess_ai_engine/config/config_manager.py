"""
配置管理器

负责加载、保存和管理各种配置。
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar
from dataclasses import asdict, fields
import logging

from .model_config import (
    MCTSConfig, TrainingConfig, ModelConfig, AIConfig, 
    SystemConfig, GameConfig,
    DEFAULT_MCTS_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_MODEL_CONFIG,
    DEFAULT_AI_CONFIG, DEFAULT_SYSTEM_CONFIG, DEFAULT_GAME_CONFIG
)

T = TypeVar('T')

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    配置管理器
    
    负责加载、保存和管理系统的各种配置。
    """
    
    def __init__(self, config_dir: str = "chess_ai_project/configs/chinese_chess_ai_engine"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 配置文件路径
        self.config_files = {
            'mcts': self.config_dir / 'mcts_config.yaml',
            'training': self.config_dir / 'training_config.yaml',
            'model': self.config_dir / 'model_config.yaml',
            'ai': self.config_dir / 'ai_config.yaml',
            'system': self.config_dir / 'system_config.yaml',
            'game': self.config_dir / 'game_config.yaml'
        }
        
        # 默认配置
        self.default_configs = {
            'mcts': DEFAULT_MCTS_CONFIG,
            'training': DEFAULT_TRAINING_CONFIG,
            'model': DEFAULT_MODEL_CONFIG,
            'ai': DEFAULT_AI_CONFIG,
            'system': DEFAULT_SYSTEM_CONFIG,
            'game': DEFAULT_GAME_CONFIG
        }
        
        # 配置类型映射
        self.config_types = {
            'mcts': MCTSConfig,
            'training': TrainingConfig,
            'model': ModelConfig,
            'ai': AIConfig,
            'system': SystemConfig,
            'game': GameConfig
        }
        
        # 初始化默认配置文件
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """初始化默认配置文件"""
        for config_name, config_obj in self.default_configs.items():
            config_file = self.config_files[config_name]
            if not config_file.exists():
                self.save_config(config_name, config_obj)
                logger.info(f"创建默认配置文件: {config_file}")
    
    def load_config(self, config_name: str, config_class: Type[T]) -> T:
        """
        加载配置
        
        Args:
            config_name: 配置名称
            config_class: 配置类
            
        Returns:
            配置对象
        """
        config_file = self.config_files.get(config_name)
        if not config_file or not config_file.exists():
            logger.warning(f"配置文件不存在: {config_file}，使用默认配置")
            return self.default_configs[config_name]
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix == '.yaml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # 创建配置对象
            config = self._dict_to_dataclass(data, config_class)
            logger.info(f"成功加载配置: {config_file}")
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {config_file}, 错误: {e}")
            return self.default_configs[config_name]
    
    def save_config(self, config_name: str, config_obj: Any):
        """
        保存配置
        
        Args:
            config_name: 配置名称
            config_obj: 配置对象
        """
        config_file = self.config_files.get(config_name)
        if not config_file:
            raise ValueError(f"未知的配置名称: {config_name}")
        
        try:
            # 转换为字典
            data = asdict(config_obj)
            
            # 保存文件
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix == '.yaml':
                    yaml.dump(data, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                else:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功保存配置: {config_file}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {config_file}, 错误: {e}")
            raise
    
    def get_mcts_config(self) -> MCTSConfig:
        """获取MCTS配置"""
        return self.load_config('mcts', MCTSConfig)
    
    def get_training_config(self) -> TrainingConfig:
        """获取训练配置"""
        return self.load_config('training', TrainingConfig)
    
    def get_model_config(self) -> ModelConfig:
        """获取模型配置"""
        return self.load_config('model', ModelConfig)
    
    def get_ai_config(self) -> AIConfig:
        """获取AI配置"""
        return self.load_config('ai', AIConfig)
    
    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        return self.load_config('system', SystemConfig)
    
    def get_game_config(self) -> GameConfig:
        """获取游戏配置"""
        return self.load_config('game', GameConfig)
    
    def update_config(self, config_name: str, **kwargs):
        """
        更新配置
        
        Args:
            config_name: 配置名称
            **kwargs: 要更新的配置项
        """
        config_class = self.config_types[config_name]
        config = self.load_config(config_name, config_class)
        
        # 更新配置项
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"配置项不存在: {key}")
        
        # 保存更新后的配置
        self.save_config(config_name, config)
    
    def reset_config(self, config_name: str):
        """
        重置配置为默认值
        
        Args:
            config_name: 配置名称
        """
        default_config = self.default_configs[config_name]
        self.save_config(config_name, default_config)
        logger.info(f"配置已重置为默认值: {config_name}")
    
    def validate_config(self, config_name: str) -> bool:
        """
        验证配置的有效性
        
        Args:
            config_name: 配置名称
            
        Returns:
            bool: 配置是否有效
        """
        try:
            config_class = self.config_types[config_name]
            config = self.load_config(config_name, config_class)
            
            # 基本验证
            if config_name == 'mcts':
                return (config.num_simulations > 0 and 
                       config.c_puct > 0 and 
                       config.temperature > 0)
            elif config_name == 'training':
                return (config.learning_rate > 0 and 
                       config.batch_size > 0 and 
                       config.num_epochs > 0)
            elif config_name == 'model':
                return (config.input_channels > 0 and 
                       config.num_blocks > 0 and 
                       config.hidden_channels > 0)
            elif config_name == 'ai':
                return (config.search_time > 0 and 
                       config.max_simulations > 0 and 
                       1 <= config.difficulty_level <= 10)
            
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {config_name}, 错误: {e}")
            return False
    
    def get_all_configs(self) -> Dict[str, Any]:
        """
        获取所有配置
        
        Returns:
            Dict[str, Any]: 所有配置的字典
        """
        configs = {}
        for config_name, config_class in self.config_types.items():
            configs[config_name] = self.load_config(config_name, config_class)
        return configs
    
    def export_configs(self, export_path: str):
        """
        导出所有配置到文件
        
        Args:
            export_path: 导出文件路径
        """
        configs = self.get_all_configs()
        export_data = {}
        
        for config_name, config_obj in configs.items():
            export_data[config_name] = asdict(config_obj)
        
        export_file = Path(export_path)
        with open(export_file, 'w', encoding='utf-8') as f:
            if export_file.suffix == '.yaml':
                yaml.dump(export_data, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            else:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"配置已导出到: {export_path}")
    
    def _dict_to_dataclass(self, data: Dict[str, Any], dataclass_type: Type[T]) -> T:
        """
        将字典转换为数据类对象
        
        Args:
            data: 字典数据
            dataclass_type: 数据类类型
            
        Returns:
            数据类对象
        """
        # 获取数据类的字段
        field_names = {f.name for f in fields(dataclass_type)}
        
        # 过滤有效字段
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        
        return dataclass_type(**filtered_data)