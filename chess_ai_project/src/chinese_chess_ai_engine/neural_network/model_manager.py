"""
模型管理器

负责模型的保存、加载、版本管理和元数据存储。
"""

import os
import json
import torch
import shutil
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from .chess_net import ChessNet


class ModelManager:
    """
    模型管理器
    
    提供模型的保存、加载、版本管理和元数据存储功能。
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        初始化模型管理器
        
        Args:
            model_dir: 模型存储目录
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.versions_dir = self.model_dir / "versions"
        self.exports_dir = self.model_dir / "exports"
        self.metadata_dir = self.model_dir / "metadata"
        
        for dir_path in [self.checkpoints_dir, self.versions_dir, self.exports_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 版本信息文件
        self.version_file = self.model_dir / "versions.json"
        self._load_version_info()
    
    def _load_version_info(self):
        """加载版本信息"""
        if self.version_file.exists():
            with open(self.version_file, 'r', encoding='utf-8') as f:
                self.version_info = json.load(f)
        else:
            self.version_info = {
                'versions': {},
                'latest': None,
                'best': None
            }
    
    def _save_version_info(self):
        """保存版本信息"""
        with open(self.version_file, 'w', encoding='utf-8') as f:
            json.dump(self.version_info, f, indent=2, ensure_ascii=False)
    
    def save_model(
        self, 
        model: ChessNet, 
        version: str, 
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        保存模型
        
        Args:
            model: 要保存的模型
            version: 版本号
            metadata: 元数据信息
            is_best: 是否为最佳模型
            
        Returns:
            str: 保存的文件路径
        """
        # 创建版本目录
        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型文件
        model_path = version_dir / "model.pth"
        config_path = version_dir / "config.json"
        metadata_path = version_dir / "metadata.json"
        
        # 准备保存数据
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_channels': model.input_channels,
                'num_blocks': model.num_blocks,
                'channels': model.channels,
            },
            'version': version,
            'timestamp': datetime.now().isoformat(),
        }
        
        # 计算模型哈希
        model_hash = self._calculate_model_hash(model)
        save_data['model_hash'] = model_hash
        
        # 保存模型
        torch.save(save_data, model_path)
        
        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(save_data['model_config'], f, indent=2)
        
        # 准备元数据
        full_metadata = {
            'version': version,
            'timestamp': save_data['timestamp'],
            'model_hash': model_hash,
            'model_info': model.get_model_info(),
            'file_size': 0,  # 将在保存后更新
        }
        
        if metadata:
            full_metadata.update(metadata)
        
        # 更新文件大小
        full_metadata['file_size'] = model_path.stat().st_size
        
        # 保存元数据
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        
        # 更新版本信息
        self.version_info['versions'][version] = {
            'path': str(model_path),
            'config_path': str(config_path),
            'metadata_path': str(metadata_path),
            'timestamp': save_data['timestamp'],
            'model_hash': model_hash,
            'is_best': is_best
        }
        
        self.version_info['latest'] = version
        if is_best:
            self.version_info['best'] = version
        
        self._save_version_info()
        
        self.logger.info(f"模型已保存: {version} -> {model_path}")
        return str(model_path)
    
    def load_model(self, version: str = 'latest') -> Tuple[ChessNet, Dict[str, Any]]:
        """
        加载模型
        
        Args:
            version: 版本号，'latest'表示最新版本，'best'表示最佳版本
            
        Returns:
            Tuple[ChessNet, Dict[str, Any]]: (模型对象, 元数据)
        """
        # 解析版本号
        if version == 'latest':
            version = self.version_info.get('latest')
        elif version == 'best':
            version = self.version_info.get('best')
        
        if not version or version not in self.version_info['versions']:
            raise ValueError(f"版本 {version} 不存在")
        
        version_data = self.version_info['versions'][version]
        model_path = Path(version_data['path'])
        metadata_path = Path(version_data['metadata_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型数据
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        # 创建模型
        model = ChessNet(
            input_channels=model_config['input_channels'],
            num_blocks=model_config['num_blocks'],
            channels=model_config['channels']
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载元数据
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        self.logger.info(f"模型已加载: {version} <- {model_path}")
        return model, metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        列出所有模型版本
        
        Returns:
            List[Dict[str, Any]]: 模型版本列表
        """
        models = []
        
        for version, version_data in self.version_info['versions'].items():
            model_info = {
                'version': version,
                'timestamp': version_data['timestamp'],
                'model_hash': version_data['model_hash'],
                'is_best': version_data.get('is_best', False),
                'is_latest': version == self.version_info.get('latest'),
                'path': version_data['path']
            }
            
            # 添加文件大小信息
            model_path = Path(version_data['path'])
            if model_path.exists():
                model_info['file_size'] = model_path.stat().st_size
                model_info['file_size_mb'] = model_info['file_size'] / (1024 * 1024)
            
            # 添加元数据信息
            metadata_path = Path(version_data.get('metadata_path', ''))
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        model_info['metadata'] = metadata
                except Exception as e:
                    self.logger.warning(f"无法加载元数据 {metadata_path}: {e}")
            
            models.append(model_info)
        
        # 按时间戳排序
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models
    
    def delete_model(self, version: str) -> bool:
        """
        删除模型版本
        
        Args:
            version: 版本号
            
        Returns:
            bool: 是否删除成功
        """
        if version not in self.version_info['versions']:
            self.logger.warning(f"版本 {version} 不存在")
            return False
        
        version_data = self.version_info['versions'][version]
        version_dir = Path(version_data['path']).parent
        
        try:
            # 删除版本目录
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # 更新版本信息
            del self.version_info['versions'][version]
            
            # 更新latest和best指针
            if self.version_info.get('latest') == version:
                remaining_versions = list(self.version_info['versions'].keys())
                if remaining_versions:
                    # 选择最新的版本作为latest
                    latest_version = max(remaining_versions, 
                                       key=lambda v: self.version_info['versions'][v]['timestamp'])
                    self.version_info['latest'] = latest_version
                else:
                    self.version_info['latest'] = None
            
            if self.version_info.get('best') == version:
                # 寻找其他最佳模型
                best_candidates = [v for v, data in self.version_info['versions'].items() 
                                 if data.get('is_best', False)]
                self.version_info['best'] = best_candidates[0] if best_candidates else None
            
            self._save_version_info()
            
            self.logger.info(f"模型版本已删除: {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除模型版本失败 {version}: {e}")
            return False
    
    def export_onnx(
        self, 
        model: ChessNet, 
        version: str, 
        input_shape: Tuple[int, ...] = (1, 20, 10, 9)
    ) -> str:
        """
        导出ONNX格式模型
        
        Args:
            model: 要导出的模型
            version: 版本号
            input_shape: 输入形状
            
        Returns:
            str: 导出的ONNX文件路径
        """
        try:
            import onnx
            import onnxruntime
        except ImportError:
            raise ImportError("需要安装onnx和onnxruntime: pip install onnx onnxruntime")
        
        # 创建导出目录
        export_dir = self.exports_dir / version
        export_dir.mkdir(parents=True, exist_ok=True)
        
        onnx_path = export_dir / "model.onnx"
        
        # 设置模型为评估模式
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(input_shape)
        
        # 导出ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['board_features'],
            output_names=['value', 'policy'],
            dynamic_axes={
                'board_features': {0: 'batch_size'},
                'value': {0: 'batch_size'},
                'policy': {0: 'batch_size'}
            }
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        # 测试ONNX Runtime
        ort_session = onnxruntime.InferenceSession(str(onnx_path))
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # 保存导出信息
        export_info = {
            'version': version,
            'onnx_path': str(onnx_path),
            'input_shape': input_shape,
            'export_timestamp': datetime.now().isoformat(),
            'onnx_opset_version': 11,
            'input_names': ['board_features'],
            'output_names': ['value', 'policy'],
            'file_size': onnx_path.stat().st_size
        }
        
        info_path = export_dir / "export_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(export_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"ONNX模型已导出: {version} -> {onnx_path}")
        return str(onnx_path)
    
    def quantize_model(
        self, 
        model: ChessNet, 
        version: str,
        quantization_type: str = 'dynamic'
    ) -> str:
        """
        量化模型
        
        Args:
            model: 要量化的模型
            version: 版本号
            quantization_type: 量化类型 ('dynamic', 'static')
            
        Returns:
            str: 量化模型路径
        """
        # 创建量化目录
        quant_dir = self.exports_dir / f"{version}_quantized"
        quant_dir.mkdir(parents=True, exist_ok=True)
        
        quant_path = quant_dir / "model_quantized.pth"
        
        # 设置模型为评估模式
        model.eval()
        
        if quantization_type == 'dynamic':
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
        else:
            raise NotImplementedError(f"量化类型 {quantization_type} 暂未实现")
        
        # 保存量化模型
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'quantization_type': quantization_type,
            'original_version': version,
            'quantized_timestamp': datetime.now().isoformat()
        }, quant_path)
        
        # 保存量化信息
        quant_info = {
            'original_version': version,
            'quantization_type': quantization_type,
            'quantized_path': str(quant_path),
            'quantized_timestamp': datetime.now().isoformat(),
            'file_size': quant_path.stat().st_size
        }
        
        info_path = quant_dir / "quantization_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(quant_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"模型已量化: {version} -> {quant_path}")
        return str(quant_path)
    
    def _calculate_model_hash(self, model: ChessNet) -> str:
        """
        计算模型哈希值
        
        Args:
            model: 模型对象
            
        Returns:
            str: 模型哈希值
        """
        # 获取模型参数的字节表示
        model_bytes = b''
        for param in model.parameters():
            model_bytes += param.data.cpu().numpy().tobytes()
        
        # 计算MD5哈希
        return hashlib.md5(model_bytes).hexdigest()
    
    def backup_models(self, backup_dir: str) -> str:
        """
        备份所有模型
        
        Args:
            backup_dir: 备份目录
            
        Returns:
            str: 备份路径
        """
        backup_path = Path(backup_dir) / f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 复制整个模型目录
        shutil.copytree(self.model_dir, backup_path / "models")
        
        # 创建备份信息
        backup_info = {
            'backup_timestamp': datetime.now().isoformat(),
            'original_path': str(self.model_dir),
            'backup_path': str(backup_path),
            'models_count': len(self.version_info['versions']),
            'total_size': sum(Path(data['path']).stat().st_size 
                            for data in self.version_info['versions'].values()
                            if Path(data['path']).exists())
        }
        
        with open(backup_path / "backup_info.json", 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"模型已备份到: {backup_path}")
        return str(backup_path)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        获取存储信息
        
        Returns:
            Dict[str, Any]: 存储信息
        """
        total_size = 0
        file_count = 0
        
        # 统计所有文件
        for root, dirs, files in os.walk(self.model_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.exists():
                    total_size += file_path.stat().st_size
                    file_count += 1
        
        return {
            'model_dir': str(self.model_dir),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'versions_count': len(self.version_info['versions']),
            'latest_version': self.version_info.get('latest'),
            'best_version': self.version_info.get('best'),
            'subdirectories': {
                'checkpoints': str(self.checkpoints_dir),
                'versions': str(self.versions_dir),
                'exports': str(self.exports_dir),
                'metadata': str(self.metadata_dir)
            }
        }