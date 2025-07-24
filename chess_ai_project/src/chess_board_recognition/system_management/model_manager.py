"""
模型管理模块

该模块提供了用于管理模型的类和函数。
"""

import os
import json
import shutil
import time
from typing import Dict, Any, List, Optional

from .logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

class ModelManager:
    """
    模型管理器类
    
    该类用于管理模型，包括版本控制、备份和恢复等功能。
    """
    
    def __init__(self, models_dir: str = './models'):
        """
        初始化模型管理器
        
        参数:
            models_dir: 模型目录
        """
        self.models_dir = models_dir
        
        # 创建模型目录
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info(f"模型管理器初始化完成，模型目录: {models_dir}")
    
    def save_model(self, model_path: str, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        保存模型
        
        参数:
            model_path: 模型路径
            model_name: 模型名称
            metadata: 模型元数据
            
        返回:
            保存后的模型路径
        """
        try:
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 生成版本号
            version = time.strftime("%Y%m%d%H%M%S")
            
            # 构建目标路径
            target_dir = os.path.join(self.models_dir, model_name)
            os.makedirs(target_dir, exist_ok=True)
            
            target_path = os.path.join(target_dir, f"{model_name}_v{version}.pt")
            
            # 复制模型文件
            shutil.copy2(model_path, target_path)
            
            # 保存元数据
            if metadata:
                metadata_path = os.path.join(target_dir, f"{model_name}_v{version}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 更新最新版本链接
            latest_path = os.path.join(target_dir, f"{model_name}_latest.pt")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            
            # 在Windows上，我们不能使用符号链接，所以直接复制文件
            shutil.copy2(target_path, latest_path)
            
            logger.info(f"模型已保存: {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
    
    def load_model(self, model_name: str, version: str = 'latest') -> str:
        """
        加载模型
        
        参数:
            model_name: 模型名称
            version: 模型版本，默认为最新版本
            
        返回:
            模型路径
        """
        try:
            # 构建模型路径
            model_dir = os.path.join(self.models_dir, model_name)
            
            if version == 'latest':
                model_path = os.path.join(model_dir, f"{model_name}_latest.pt")
            else:
                model_path = os.path.join(model_dir, f"{model_name}_v{version}.pt")
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            logger.info(f"模型已加载: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def list_models(self) -> Dict[str, List[str]]:
        """
        列出所有模型
        
        返回:
            模型字典，格式为 {模型名称: [版本列表]}
        """
        try:
            models = {}
            
            # 遍历模型目录
            for model_name in os.listdir(self.models_dir):
                model_dir = os.path.join(self.models_dir, model_name)
                
                # 检查是否为目录
                if not os.path.isdir(model_dir):
                    continue
                
                # 获取版本列表
                versions = []
                for filename in os.listdir(model_dir):
                    if filename.startswith(f"{model_name}_v") and filename.endswith(".pt"):
                        version = filename[len(f"{model_name}_v"):-3]
                        versions.append(version)
                
                # 按版本排序
                versions.sort(reverse=True)
                
                models[model_name] = versions
            
            return models
        except Exception as e:
            logger.error(f"列出模型失败: {e}")
            return {}
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """
        删除模型
        
        参数:
            model_name: 模型名称
            version: 模型版本
            
        返回:
            是否删除成功
        """
        try:
            # 构建模型路径
            model_dir = os.path.join(self.models_dir, model_name)
            model_path = os.path.join(model_dir, f"{model_name}_v{version}.pt")
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            # 删除模型文件
            os.remove(model_path)
            
            # 删除元数据文件（如果存在）
            metadata_path = os.path.join(model_dir, f"{model_name}_v{version}_metadata.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # 检查是否为最新版本
            latest_path = os.path.join(model_dir, f"{model_name}_latest.pt")
            if os.path.exists(latest_path):
                # 获取最新版本的实际路径
                real_path = os.path.realpath(latest_path)
                
                # 如果删除的是最新版本，则更新最新版本链接
                if os.path.basename(real_path) == f"{model_name}_v{version}.pt":
                    # 获取剩余版本
                    versions = []
                    for filename in os.listdir(model_dir):
                        if filename.startswith(f"{model_name}_v") and filename.endswith(".pt"):
                            versions.append(filename[len(f"{model_name}_v"):-3])
                    
                    # 按版本排序
                    versions.sort(reverse=True)
                    
                    # 如果还有其他版本，则更新最新版本链接
                    if versions:
                        new_latest_path = os.path.join(model_dir, f"{model_name}_v{versions[0]}.pt")
                        os.remove(latest_path)
                        shutil.copy2(new_latest_path, latest_path)
                    else:
                        # 如果没有其他版本，则删除最新版本链接
                        os.remove(latest_path)
            
            logger.info(f"模型已删除: {model_path}")
            return True
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False
    
    def get_model_metadata(self, model_name: str, version: str = 'latest') -> Dict[str, Any]:
        """
        获取模型元数据
        
        参数:
            model_name: 模型名称
            version: 模型版本，默认为最新版本
            
        返回:
            模型元数据
        """
        try:
            # 构建模型路径
            model_dir = os.path.join(self.models_dir, model_name)
            
            if version == 'latest':
                # 获取最新版本的实际路径
                latest_path = os.path.join(model_dir, f"{model_name}_latest.pt")
                if not os.path.exists(latest_path):
                    logger.error(f"最新版本模型文件不存在: {latest_path}")
                    return {}
                
                # 从文件名中提取版本号
                real_path = os.path.realpath(latest_path)
                filename = os.path.basename(real_path)
                if filename.startswith(f"{model_name}_v") and filename.endswith(".pt"):
                    version = filename[len(f"{model_name}_v"):-3]
                else:
                    logger.error(f"无法从文件名中提取版本号: {filename}")
                    return {}
            
            # 构建元数据路径
            metadata_path = os.path.join(model_dir, f"{model_name}_v{version}_metadata.json")
            
            # 检查元数据文件是否存在
            if not os.path.exists(metadata_path):
                logger.warning(f"模型元数据文件不存在: {metadata_path}")
                return {}
            
            # 读取元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return metadata
        except Exception as e:
            logger.error(f"获取模型元数据失败: {e}")
            return {}