"""
核心模块测试

测试核心接口、配置管理和日志系统。
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np

from chess_ai_project.src.chess_board_recognition.core.interfaces import (
    Detection, BoardState, CHESS_CLASSES, DEFAULT_CONFIG
)
from chess_ai_project.src.chess_board_recognition.core.config import ConfigManager
from chess_ai_project.src.chess_board_recognition.core.logger import setup_logger


class TestDetection:
    """测试Detection数据结构"""
    
    def test_valid_detection(self):
        """测试有效的检测结果"""
        detection = Detection(
            class_id=2,
            class_name="red_king",
            confidence=0.95,
            bbox=(100, 200, 150, 250),
            center=(125, 225)
        )
        
        assert detection.class_id == 2
        assert detection.class_name == "red_king"
        assert detection.confidence == 0.95
        assert detection.bbox == (100, 200, 150, 250)
        assert detection.center == (125, 225)
    
    def test_invalid_confidence(self):
        """测试无效的置信度"""
        with pytest.raises(ValueError, match="置信度必须在0-1之间"):
            Detection(
                class_id=2,
                class_name="red_king",
                confidence=1.5,
                bbox=(100, 200, 150, 250),
                center=(125, 225)
            )
    
    def test_invalid_bbox(self):
        """测试无效的边界框"""
        with pytest.raises(ValueError, match="边界框必须包含4个坐标值"):
            Detection(
                class_id=2,
                class_name="red_king",
                confidence=0.95,
                bbox=(100, 200, 150),  # 缺少一个坐标
                center=(125, 225)
            )


class TestBoardState:
    """测试BoardState数据结构"""
    
    def test_valid_board_state(self):
        """测试有效的棋局状态"""
        matrix = np.zeros((10, 9), dtype=int)
        detections = [
            Detection(2, "red_king", 0.95, (100, 200, 150, 250), (125, 225))
        ]
        
        board_state = BoardState(
            matrix=matrix,
            selected_piece=(4, 5),
            confidence=0.85,
            timestamp=datetime.now(),
            detections=detections
        )
        
        assert board_state.matrix.shape == (10, 9)
        assert board_state.selected_piece == (4, 5)
        assert board_state.confidence == 0.85
        assert len(board_state.detections) == 1
    
    def test_invalid_matrix_shape(self):
        """测试无效的矩阵形状"""
        matrix = np.zeros((8, 8), dtype=int)  # 错误的形状
        
        with pytest.raises(ValueError, match="棋局矩阵必须是10x9"):
            BoardState(
                matrix=matrix,
                selected_piece=None,
                confidence=0.85,
                timestamp=datetime.now(),
                detections=[]
            )
    
    def test_invalid_selected_piece(self):
        """测试无效的选中棋子位置"""
        matrix = np.zeros((10, 9), dtype=int)
        
        with pytest.raises(ValueError, match="选中棋子位置超出棋盘范围"):
            BoardState(
                matrix=matrix,
                selected_piece=(10, 5),  # 超出范围
                confidence=0.85,
                timestamp=datetime.now(),
                detections=[]
            )


class TestConfigManager:
    """测试配置管理器"""
    
    def test_default_config(self):
        """测试默认配置"""
        config_manager = ConfigManager()
        config = config_manager.config
        
        assert "model" in config
        assert "capture" in config
        assert "training" in config
        assert "logging" in config
    
    def test_load_yaml_config(self):
        """测试加载YAML配置"""
        test_config = {
            "model": {"confidence_threshold": 0.7},
            "training": {"epochs": 200}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            config = config_manager.config
            
            # 验证配置合并
            assert config["model"]["confidence_threshold"] == 0.7
            assert config["training"]["epochs"] == 200
            # 默认值应该保留
            assert "nms_threshold" in config["model"]
            
        finally:
            Path(config_file).unlink()
    
    def test_load_json_config(self):
        """测试加载JSON配置"""
        test_config = {
            "model": {"confidence_threshold": 0.6},
            "capture": {"auto_interval": 5}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            config = config_manager.config
            
            assert config["model"]["confidence_threshold"] == 0.6
            assert config["capture"]["auto_interval"] == 5
            
        finally:
            Path(config_file).unlink()
    
    def test_get_nested_config(self):
        """测试获取嵌套配置"""
        config_manager = ConfigManager()  # 不加载任何配置文件
        
        # 测试点号分隔的键
        confidence = config_manager.get("model.confidence_threshold")
        # 使用DEFAULT_CONFIG中的实际值
        from chess_ai_project.src.chess_board_recognition.core.interfaces import DEFAULT_CONFIG
        expected_confidence = DEFAULT_CONFIG["model"]["confidence_threshold"]
        assert confidence == expected_confidence
        
        # 测试不存在的键
        non_existent = config_manager.get("model.non_existent", "default")
        assert non_existent == "default"
    
    def test_set_nested_config(self):
        """测试设置嵌套配置"""
        config_manager = ConfigManager()
        
        config_manager.set("model.confidence_threshold", 0.8)
        assert config_manager.get("model.confidence_threshold") == 0.8
        
        # 测试创建新的嵌套键
        config_manager.set("new_section.new_key", "new_value")
        assert config_manager.get("new_section.new_key") == "new_value"
    
    def test_config_validation(self):
        """测试配置验证"""
        config_manager = ConfigManager()
        
        # 测试有效配置
        valid_config = DEFAULT_CONFIG.copy()
        assert config_manager.validate_config(valid_config) is True
        
        # 测试无效的置信度阈值
        invalid_config = DEFAULT_CONFIG.copy()
        invalid_config["model"]["confidence_threshold"] = 1.5
        assert config_manager.validate_config(invalid_config) is False


class TestLogger:
    """测试日志系统"""
    
    def test_setup_logger(self):
        """测试日志设置"""
        logger = setup_logger(
            name="test_logger",
            level="DEBUG",
            console_output=True
        )
        
        assert logger.name == "test_logger"
        assert logger.level == 10  # DEBUG level
        assert len(logger.handlers) > 0
    
    def test_file_logger(self):
        """测试文件日志"""
        import tempfile
        import logging
        
        # 创建临时目录和文件
        temp_dir = tempfile.mkdtemp()
        log_file = Path(temp_dir) / "test.log"
        
        try:
            logger = setup_logger(
                name="file_test_logger",
                log_file=str(log_file),
                console_output=False
            )
            
            logger.info("测试日志消息")
            
            # 关闭所有处理器以释放文件锁
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # 验证日志文件存在且包含内容
            assert log_file.exists()
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "测试日志消息" in content
                
        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestConstants:
    """测试常量定义"""
    
    def test_chess_classes(self):
        """测试棋子类别定义"""
        assert len(CHESS_CLASSES) == 17
        assert CHESS_CLASSES[0] == "board"
        assert CHESS_CLASSES[2] == "red_king"
        assert CHESS_CLASSES[9] == "black_king"
        assert CHESS_CLASSES[16] == "selected_piece"
    
    def test_default_config_structure(self):
        """测试默认配置结构"""
        assert "model" in DEFAULT_CONFIG
        assert "capture" in DEFAULT_CONFIG
        assert "training" in DEFAULT_CONFIG
        assert "logging" in DEFAULT_CONFIG
        
        # 验证模型配置
        model_config = DEFAULT_CONFIG["model"]
        assert "confidence_threshold" in model_config
        assert "nms_threshold" in model_config
        assert "device" in model_config
        
        # 验证训练配置
        training_config = DEFAULT_CONFIG["training"]
        assert "epochs" in training_config
        assert "batch_size" in training_config
        assert "learning_rate" in training_config