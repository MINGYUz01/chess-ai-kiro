"""
训练配置验证模块

该模块提供了用于验证和管理YOLO11训练配置的类和函数。
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

from ..system_management.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

class TrainingConfigValidator:
    """
    训练配置验证器类
    
    该类用于验证和管理YOLO11训练的配置参数。
    """
    
    # 定义参数约束
    PARAM_CONSTRAINTS = {
        'epochs': {'type': int, 'min': 1, 'max': 10000, 'default': 100},
        'batch_size': {'type': int, 'min': 1, 'max': 512, 'default': 16},
        'learning_rate': {'type': float, 'min': 1e-6, 'max': 1.0, 'default': 0.001},
        'image_size': {'type': int, 'min': 32, 'max': 1920, 'default': 640},
        'device': {'type': str, 'options': ['auto', 'cpu', 'cuda', '0', '1', '2', '3'], 'default': 'auto'},
        'workers': {'type': int, 'min': 0, 'max': 32, 'default': 4},
        'patience': {'type': int, 'min': 0, 'max': 1000, 'default': 50},
        'save_period': {'type': int, 'min': -1, 'max': 1000, 'default': 10},
    }
    
    # 定义必需参数
    REQUIRED_PARAMS = ['epochs', 'batch_size', 'learning_rate', 'image_size']
    
    def __init__(self):
        """
        初始化训练配置验证器
        """
        logger.info("训练配置验证器初始化")
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        验证训练配置
        
        参数:
            config: 训练配置字典
            
        返回:
            验证结果元组 (是否有效, 修正后的配置, 错误消息列表)
        """
        valid = True
        errors = []
        corrected_config = {}
        
        # 检查必需参数
        for param in self.REQUIRED_PARAMS:
            if param not in config:
                valid = False
                errors.append(f"缺少必需参数: {param}")
                # 使用默认值
                corrected_config[param] = self.PARAM_CONSTRAINTS[param]['default']
            
        # 验证所有参数
        for param, value in config.items():
            if param in self.PARAM_CONSTRAINTS:
                constraints = self.PARAM_CONSTRAINTS[param]
                
                # 类型检查
                if not isinstance(value, constraints['type']):
                    try:
                        # 尝试类型转换
                        value = constraints['type'](value)
                        logger.warning(f"参数 {param} 类型已自动转换为 {constraints['type'].__name__}")
                    except (ValueError, TypeError):
                        valid = False
                        errors.append(f"参数 {param} 类型错误，应为 {constraints['type'].__name__}")
                        value = constraints['default']
                
                # 范围检查
                if 'min' in constraints and value < constraints['min']:
                    valid = False
                    errors.append(f"参数 {param} 值 {value} 小于最小值 {constraints['min']}")
                    value = constraints['min']
                
                if 'max' in constraints and value > constraints['max']:
                    valid = False
                    errors.append(f"参数 {param} 值 {value} 大于最大值 {constraints['max']}")
                    value = constraints['max']
                
                # 选项检查
                if 'options' in constraints and value not in constraints['options']:
                    valid = False
                    errors.append(f"参数 {param} 值 {value} 不在有效选项 {constraints['options']} 中")
                    value = constraints['default']
                
                corrected_config[param] = value
            else:
                # 未知参数，保留但发出警告
                logger.warning(f"未知参数: {param}={value}")
                corrected_config[param] = value
        
        return valid, corrected_config, errors
    
    def generate_default_config(self) -> Dict[str, Any]:
        """
        生成默认训练配置
        
        返回:
            默认配置字典
        """
        default_config = {}
        for param, constraints in self.PARAM_CONSTRAINTS.items():
            default_config[param] = constraints['default']
        return default_config
    
    def save_config(self, config: Dict[str, Any], file_path: str) -> bool:
        """
        保存训练配置到文件
        
        参数:
            config: 训练配置字典
            file_path: 配置文件路径
            
        返回:
            是否保存成功
        """
        try:
            # 创建目录
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 根据文件扩展名选择保存格式
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
            else:
                logger.error(f"不支持的文件格式: {file_path}")
                return False
            
            logger.info(f"配置已保存至: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
    
    def load_config(self, file_path: str) -> Dict[str, Any]:
        """
        从文件加载训练配置
        
        参数:
            file_path: 配置文件路径
            
        返回:
            训练配置字典
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"配置文件不存在: {file_path}")
                return self.generate_default_config()
            
            # 根据文件扩展名选择加载格式
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                logger.error(f"不支持的文件格式: {file_path}")
                return self.generate_default_config()
            
            # 验证加载的配置
            valid, corrected_config, errors = self.validate_config(config)
            if not valid:
                for error in errors:
                    logger.warning(f"配置错误: {error}")
                logger.warning("已自动修正配置错误")
            
            return corrected_config
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return self.generate_default_config()
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并两个配置
        
        参数:
            base_config: 基础配置
            override_config: 覆盖配置
            
        返回:
            合并后的配置
        """
        merged_config = base_config.copy()
        merged_config.update(override_config)
        return merged_config


class DataConfigGenerator:
    """
    数据配置生成器类
    
    该类用于生成YOLO11训练所需的数据配置文件。
    """
    
    def __init__(self):
        """
        初始化数据配置生成器
        """
        logger.info("数据配置生成器初始化")
    
    def generate_data_yaml(self, 
                          train_path: str, 
                          val_path: str, 
                          test_path: Optional[str] = None,
                          class_names: List[str] = None,
                          output_path: str = 'data.yaml') -> str:
        """
        生成YOLO11数据配置文件
        
        参数:
            train_path: 训练数据路径
            val_path: 验证数据路径
            test_path: 测试数据路径，可选
            class_names: 类别名称列表
            output_path: 输出文件路径
            
        返回:
            生成的配置文件路径
        """
        try:
            # 准备配置数据
            data_config = {
                'train': train_path,
                'val': val_path,
            }
            
            if test_path:
                data_config['test'] = test_path
            
            if class_names:
                data_config['names'] = class_names
                data_config['nc'] = len(class_names)
            else:
                logger.warning("未提供类别名称，将在训练时自动检测")
            
            # 创建目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存配置
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"数据配置已生成: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"生成数据配置失败: {e}")
            raise
    
    def validate_data_paths(self, train_path: str, val_path: str, test_path: Optional[str] = None) -> bool:
        """
        验证数据路径是否有效
        
        参数:
            train_path: 训练数据路径
            val_path: 验证数据路径
            test_path: 测试数据路径，可选
            
        返回:
            是否有效
        """
        valid = True
        
        # 检查训练数据路径
        if not os.path.exists(train_path):
            logger.error(f"训练数据路径不存在: {train_path}")
            valid = False
        
        # 检查验证数据路径
        if not os.path.exists(val_path):
            logger.error(f"验证数据路径不存在: {val_path}")
            valid = False
        
        # 检查测试数据路径（如果提供）
        if test_path and not os.path.exists(test_path):
            logger.error(f"测试数据路径不存在: {test_path}")
            valid = False
        
        return valid