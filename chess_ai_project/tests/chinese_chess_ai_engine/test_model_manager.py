"""
模型管理器测试

测试模型的保存、加载、版本管理等功能。
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from chess_ai_project.src.chinese_chess_ai_engine.neural_network import (
    ChessNet, ModelManager
)


class TestModelManager:
    """模型管理器测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelManager(self.temp_dir)
        
        # 创建测试模型
        self.test_model = ChessNet(
            input_channels=20,
            num_blocks=4,  # 使用较小的模型以加快测试
            channels=64
        )
    
    def teardown_method(self):
        """测试后的清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_model(self):
        """测试模型保存"""
        version = "test_v1.0"
        metadata = {
            "description": "测试模型",
            "training_epochs": 10,
            "accuracy": 0.85
        }
        
        # 保存模型
        saved_path = self.model_manager.save_model(
            self.test_model, 
            version, 
            metadata,
            is_best=True
        )
        
        # 验证文件存在
        assert Path(saved_path).exists()
        
        # 验证版本信息
        assert version in self.model_manager.version_info['versions']
        assert self.model_manager.version_info['latest'] == version
        assert self.model_manager.version_info['best'] == version
        
        # 验证元数据文件
        version_data = self.model_manager.version_info['versions'][version]
        metadata_path = Path(version_data['metadata_path'])
        assert metadata_path.exists()
    
    def test_load_model(self):
        """测试模型加载"""
        version = "test_v1.0"
        metadata = {"description": "测试模型"}
        
        # 先保存模型
        self.model_manager.save_model(self.test_model, version, metadata)
        
        # 加载模型
        loaded_model, loaded_metadata = self.model_manager.load_model(version)
        
        # 验证模型结构
        assert isinstance(loaded_model, ChessNet)
        assert loaded_model.input_channels == self.test_model.input_channels
        assert loaded_model.num_blocks == self.test_model.num_blocks
        assert loaded_model.channels == self.test_model.channels
        
        # 验证元数据
        assert loaded_metadata['description'] == metadata['description']
    
    def test_load_latest_model(self):
        """测试加载最新模型"""
        # 保存多个版本
        versions = ["v1.0", "v1.1", "v1.2"]
        for version in versions:
            self.model_manager.save_model(self.test_model, version)
        
        # 加载最新版本
        loaded_model, _ = self.model_manager.load_model('latest')
        
        # 验证加载的是最新版本
        assert isinstance(loaded_model, ChessNet)
        assert self.model_manager.version_info['latest'] == "v1.2"
    
    def test_load_best_model(self):
        """测试加载最佳模型"""
        # 保存多个版本，其中一个标记为最佳
        self.model_manager.save_model(self.test_model, "v1.0")
        self.model_manager.save_model(self.test_model, "v1.1", is_best=True)
        self.model_manager.save_model(self.test_model, "v1.2")
        
        # 加载最佳版本
        loaded_model, _ = self.model_manager.load_model('best')
        
        # 验证加载的是最佳版本
        assert isinstance(loaded_model, ChessNet)
        assert self.model_manager.version_info['best'] == "v1.1"
    
    def test_list_models(self):
        """测试列出模型版本"""
        # 保存多个版本
        versions = ["v1.0", "v1.1", "v1.2"]
        for i, version in enumerate(versions):
            metadata = {"epoch": i * 10}
            self.model_manager.save_model(
                self.test_model, 
                version, 
                metadata,
                is_best=(version == "v1.1")
            )
        
        # 获取模型列表
        models = self.model_manager.list_models()
        
        # 验证列表内容
        assert len(models) == 3
        
        # 验证排序（按时间戳倒序）
        timestamps = [model['timestamp'] for model in models]
        assert timestamps == sorted(timestamps, reverse=True)
        
        # 验证最新和最佳标记
        latest_model = next(m for m in models if m['is_latest'])
        best_model = next(m for m in models if m['is_best'])
        
        assert latest_model['version'] == "v1.2"
        assert best_model['version'] == "v1.1"
    
    def test_delete_model(self):
        """测试删除模型"""
        version = "test_v1.0"
        
        # 保存模型
        self.model_manager.save_model(self.test_model, version)
        
        # 验证模型存在
        assert version in self.model_manager.version_info['versions']
        
        # 删除模型
        success = self.model_manager.delete_model(version)
        
        # 验证删除成功
        assert success
        assert version not in self.model_manager.version_info['versions']
    
    def test_delete_latest_model(self):
        """测试删除最新模型时的指针更新"""
        # 保存多个版本
        versions = ["v1.0", "v1.1", "v1.2"]
        for version in versions:
            self.model_manager.save_model(self.test_model, version)
        
        # 删除最新版本
        self.model_manager.delete_model("v1.2")
        
        # 验证latest指针更新
        assert self.model_manager.version_info['latest'] == "v1.1"
    
    def test_export_onnx(self):
        """测试ONNX导出"""
        version = "test_v1.0"
        
        try:
            # 尝试导出ONNX
            onnx_path = self.model_manager.export_onnx(self.test_model, version)
            
            # 如果成功，验证导出路径
            assert Path(onnx_path).name == "model.onnx"
            assert Path(onnx_path).exists()
            
        except ImportError as e:
            # 如果ONNX相关模块不可用，跳过测试
            pytest.skip(f"ONNX相关模块不可用: {e}")
        except Exception as e:
            # 其他异常应该被重新抛出
            raise
    
    def test_quantize_model(self):
        """测试模型量化"""
        version = "test_v1.0"
        
        # 量化模型
        quant_path = self.model_manager.quantize_model(self.test_model, version)
        
        # 验证量化文件存在
        assert Path(quant_path).exists()
        assert "quantized" in quant_path
    
    def test_backup_models(self):
        """测试模型备份"""
        # 保存一些模型
        versions = ["v1.0", "v1.1"]
        for version in versions:
            self.model_manager.save_model(self.test_model, version)
        
        # 创建备份
        backup_dir = tempfile.mkdtemp()
        try:
            backup_path = self.model_manager.backup_models(backup_dir)
            
            # 验证备份存在
            assert Path(backup_path).exists()
            assert Path(backup_path, "models").exists()
            assert Path(backup_path, "backup_info.json").exists()
            
        finally:
            shutil.rmtree(backup_dir, ignore_errors=True)
    
    def test_get_storage_info(self):
        """测试获取存储信息"""
        # 保存一些模型
        versions = ["v1.0", "v1.1"]
        for version in versions:
            self.model_manager.save_model(self.test_model, version)
        
        # 获取存储信息
        storage_info = self.model_manager.get_storage_info()
        
        # 验证信息内容
        assert 'total_size_bytes' in storage_info
        assert 'total_size_mb' in storage_info
        assert 'file_count' in storage_info
        assert 'versions_count' in storage_info
        assert storage_info['versions_count'] == 2
        assert storage_info['latest_version'] == "v1.1"
    
    def test_model_hash_calculation(self):
        """测试模型哈希计算"""
        # 创建两个相同的模型
        model1 = ChessNet(input_channels=20, num_blocks=4, channels=64)
        model2 = ChessNet(input_channels=20, num_blocks=4, channels=64)
        
        # 计算哈希
        hash1 = self.model_manager._calculate_model_hash(model1)
        hash2 = self.model_manager._calculate_model_hash(model2)
        
        # 相同结构的模型应该有相同的哈希（在初始化权重相同的情况下）
        # 注意：由于权重初始化的随机性，这个测试可能需要调整
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)
        assert len(hash1) == 32  # MD5哈希长度
        assert len(hash2) == 32
    
    def test_invalid_version_load(self):
        """测试加载不存在的版本"""
        with pytest.raises(ValueError, match="版本 .* 不存在"):
            self.model_manager.load_model("nonexistent_version")
    
    def test_missing_model_file(self):
        """测试模型文件缺失的情况"""
        version = "test_v1.0"
        
        # 保存模型
        self.model_manager.save_model(self.test_model, version)
        
        # 删除模型文件但保留版本信息
        version_data = self.model_manager.version_info['versions'][version]
        model_path = Path(version_data['path'])
        model_path.unlink()
        
        # 尝试加载应该抛出异常
        with pytest.raises(FileNotFoundError):
            self.model_manager.load_model(version)


if __name__ == "__main__":
    pytest.main([__file__])