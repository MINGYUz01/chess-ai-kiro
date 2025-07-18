"""
配置管理模块

提供系统配置的加载、保存、验证和管理功能。
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .interfaces import DEFAULT_CONFIG, ChessboardRecognitionError


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        self.config_file = config_file
        self._config = DEFAULT_CONFIG.copy()
        
        if config_file and Path(config_file).exists():
            self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        从文件加载配置
        
        Returns:
            配置字典
            
        Raises:
            ChessboardRecognitionError: 配置文件格式错误时抛出
        """
        if not self.config_file:
            return self._config
            
        try:
            config_path = Path(self.config_file)
            
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            # 合并配置，保留默认值
            self._merge_config(loaded_config)
            
            # 验证配置
            if not self.validate_config(self._config):
                raise ValueError("配置验证失败")
                
            return self._config
            
        except Exception as e:
            raise ChessboardRecognitionError(f"加载配置文件失败: {e}")
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        保存配置到文件
        
        Args:
            config: 要保存的配置字典，如果为None则保存当前配置
            
        Raises:
            ChessboardRecognitionError: 保存失败时抛出
        """
        if not self.config_file:
            raise ChessboardRecognitionError("未指定配置文件路径")
        
        config_to_save = config if config is not None else self._config
        
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_to_save, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_to_save, f, ensure_ascii=False, indent=2)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
                    
        except Exception as e:
            raise ChessboardRecognitionError(f"保存配置文件失败: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置有效性
        
        Args:
            config: 要验证的配置字典
            
        Returns:
            验证是否通过
        """
        try:
            # 验证必需的配置节
            required_sections = ['model', 'capture', 'training', 'logging']
            for section in required_sections:
                if section not in config:
                    print(f"警告: 缺少配置节 '{section}'")
                    return False
            
            # 验证模型配置
            model_config = config.get('model', {})
            if 'confidence_threshold' in model_config:
                threshold = model_config['confidence_threshold']
                if not (0 <= threshold <= 1):
                    print(f"错误: confidence_threshold 必须在0-1之间，当前值: {threshold}")
                    return False
            
            if 'nms_threshold' in model_config:
                nms_threshold = model_config['nms_threshold']
                if not (0 <= nms_threshold <= 1):
                    print(f"错误: nms_threshold 必须在0-1之间，当前值: {nms_threshold}")
                    return False
            
            # 验证训练配置
            training_config = config.get('training', {})
            if 'epochs' in training_config:
                epochs = training_config['epochs']
                if not isinstance(epochs, int) or epochs <= 0:
                    print(f"错误: epochs 必须是正整数，当前值: {epochs}")
                    return False
            
            if 'batch_size' in training_config:
                batch_size = training_config['batch_size']
                if not isinstance(batch_size, int) or batch_size <= 0:
                    print(f"错误: batch_size 必须是正整数，当前值: {batch_size}")
                    return False
            
            if 'learning_rate' in training_config:
                lr = training_config['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0:
                    print(f"错误: learning_rate 必须是正数，当前值: {lr}")
                    return False
            
            # 验证截图配置
            capture_config = config.get('capture', {})
            if 'region' in capture_config:
                region = capture_config['region']
                if not (isinstance(region, list) and len(region) == 4):
                    print(f"错误: region 必须是包含4个元素的列表，当前值: {region}")
                    return False
                if not all(isinstance(x, (int, float)) and x >= 0 for x in region):
                    print(f"错误: region 的所有值必须是非负数，当前值: {region}")
                    return False
            
            if 'auto_interval' in capture_config:
                interval = capture_config['auto_interval']
                if not isinstance(interval, (int, float)) or interval <= 0:
                    print(f"错误: auto_interval 必须是正数，当前值: {interval}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"配置验证过程中出错: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return DEFAULT_CONFIG.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，支持 'section.key' 格式
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，支持 'section.key' 格式
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        # 导航到最后一级的父节点
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        批量更新配置
        
        Args:
            updates: 更新的配置字典
        """
        self._merge_config(updates)
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """
        合并配置，递归合并嵌套字典
        
        Args:
            new_config: 新配置字典
        """
        def merge_dict(base: Dict, update: Dict) -> Dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
            return base
        
        merge_dict(self._config, new_config)
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.set(key, value)