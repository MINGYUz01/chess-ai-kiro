"""
屏幕截图功能测试

测试屏幕截图和区域选择功能。
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from chess_ai_project.src.chess_board_recognition.data_collection.screen_capture import ScreenCaptureImpl
from chess_ai_project.src.chess_board_recognition.data_collection.region_selector import RegionSelector


class TestScreenCapture:
    """测试屏幕截图功能"""
    
    def setup_method(self):
        """测试前设置"""
        # 创建临时配置文件
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # 创建测试配置 (使用正斜杠避免YAML转义问题)
        save_path = str(Path(self.temp_dir) / "captures").replace("\\", "/")
        config_content = f"""
capture:
  save_path: "{save_path}"
  format: "jpg"
  quality: 95
  max_storage_gb: 1
  region: [100, 100, 400, 300]
"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
    
    def test_screen_capture_init(self):
        """测试屏幕截图器初始化"""
        capture = ScreenCaptureImpl(str(self.config_file))
        
        assert capture.config_path == str(self.config_file)
        assert capture.format == "jpg"
        assert capture.quality == 95
        assert capture.max_storage_gb == 1
        assert capture.save_path.name == "captures"
    
    @patch('chess_ai_project.src.chess_board_recognition.data_collection.screen_capture.pyautogui')
    def test_manual_capture(self, mock_pyautogui):
        """测试手动截图"""
        # 模拟pyautogui.screenshot
        mock_screenshot = Mock()
        mock_pyautogui.screenshot.return_value = mock_screenshot
        
        # 模拟save方法，实际创建文件
        def mock_save(filepath, format_type=None, **kwargs):
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).touch()
        
        mock_screenshot.save.side_effect = mock_save
        
        capture = ScreenCaptureImpl(str(self.config_file))
        
        # 执行手动截图
        result = capture.manual_capture()
        
        # 验证结果
        assert result != ""
        assert Path(result).exists()
        
        # 验证pyautogui被正确调用
        mock_pyautogui.screenshot.assert_called_once()
        mock_screenshot.save.assert_called_once()
    
    def test_get_capture_stats(self):
        """测试获取截图统计"""
        capture = ScreenCaptureImpl(str(self.config_file))
        
        stats = capture.get_capture_stats()
        
        assert isinstance(stats, dict)
        assert "is_capturing" in stats
        assert "capture_count" in stats
        assert "save_path" in stats
        assert "format" in stats
        assert "quality" in stats
        assert "storage" in stats
        assert "file_count" in stats
        assert "total_size_mb" in stats
    
    @patch('chess_ai_project.src.chess_board_recognition.data_collection.screen_capture.pyautogui')
    def test_auto_capture_start_stop(self, mock_pyautogui):
        """测试自动截图的启动和停止"""
        mock_screenshot = Mock()
        mock_pyautogui.screenshot.return_value = mock_screenshot
        
        capture = ScreenCaptureImpl(str(self.config_file))
        
        # 启动自动截图
        capture.start_auto_capture(1)
        assert capture._is_capturing is True
        
        # 等待一小段时间
        time.sleep(0.1)
        
        # 停止自动截图
        capture.stop_capture()
        assert capture._is_capturing is False
    
    @patch('chess_ai_project.src.chess_board_recognition.data_collection.screen_capture.shutil')
    def test_check_storage_space(self, mock_shutil):
        """测试存储空间检查"""
        # 模拟磁盘使用情况 (总计1GB, 已用0.5GB, 剩余0.5GB)
        mock_shutil.disk_usage.return_value = (1024**3, 512*1024**2, 512*1024**2)
        
        capture = ScreenCaptureImpl(str(self.config_file))
        
        # 测试有足够空间的情况
        result = capture._check_storage_space()
        assert result is True
        
        # 模拟空间不足的情况 (剩余100MB)
        mock_shutil.disk_usage.return_value = (1024**3, 924*1024**2, 100*1024**2)
        result = capture._check_storage_space()
        assert result is False
    
    def test_generate_filename(self):
        """测试文件名生成"""
        capture = ScreenCaptureImpl(str(self.config_file))
        
        filename = capture._generate_filename()
        
        assert isinstance(filename, str)
        assert filename.endswith(".jpg")
        assert "/" in filename  # 包含日期目录
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestRegionSelector:
    """测试区域选择器"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
    def test_region_selector_init(self):
        """测试区域选择器初始化"""
        selector = RegionSelector()
        
        assert selector.config_file.name == "region_config.json"
        assert selector.selected_region is None
    
    @patch('chess_ai_project.src.chess_board_recognition.data_collection.region_selector.pyautogui')
    def test_save_and_load_region_config(self, mock_pyautogui):
        """测试保存和加载区域配置"""
        mock_pyautogui.size.return_value = (1920, 1080)
        
        selector = RegionSelector()
        # 设置临时配置文件路径
        selector.config_file = Path(self.temp_dir) / "region_config.json"
        
        # 测试保存配置
        test_region = (100, 200, 400, 300)
        selector.save_region_config(test_region)
        
        assert selector.config_file.exists()
        
        # 测试加载配置
        loaded_region = selector.load_region_config()
        assert loaded_region == test_region
    
    def test_load_nonexistent_config(self):
        """测试加载不存在的配置文件"""
        selector = RegionSelector()
        selector.config_file = Path(self.temp_dir) / "nonexistent.json"
        
        region = selector.load_region_config()
        assert region == (0, 0, 800, 600)  # 默认值
    
    @patch('chess_ai_project.src.chess_board_recognition.data_collection.region_selector.pyautogui')
    @patch('chess_ai_project.src.chess_board_recognition.data_collection.region_selector.simpledialog')
    def test_get_region_with_dialog(self, mock_dialog, mock_pyautogui):
        """测试通过对话框获取区域"""
        mock_pyautogui.size.return_value = (1920, 1080)
        
        # 模拟用户输入
        mock_dialog.askinteger.side_effect = [100, 200, 400, 300]
        
        selector = RegionSelector()
        region = selector.get_region_with_dialog()
        
        assert region == (100, 200, 400, 300)
        assert mock_dialog.askinteger.call_count == 4
    
    @patch('chess_ai_project.src.chess_board_recognition.data_collection.region_selector.pyautogui')
    @patch('chess_ai_project.src.chess_board_recognition.data_collection.region_selector.simpledialog')
    def test_get_region_with_dialog_cancel(self, mock_dialog, mock_pyautogui):
        """测试对话框取消操作"""
        mock_pyautogui.size.return_value = (1920, 1080)
        
        # 模拟用户取消（返回None）
        mock_dialog.askinteger.return_value = None
        
        selector = RegionSelector()
        region = selector.get_region_with_dialog()
        
        assert region == (0, 0, 800, 600)  # 默认值
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestIntegration:
    """集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
        # 创建测试配置 (使用正斜杠避免YAML转义问题)
        save_path = str(Path(self.temp_dir) / "captures").replace("\\", "/")
        config_content = f"""
capture:
  save_path: "{save_path}"
  format: "png"
  quality: 95
  max_storage_gb: 1
  region: [0, 0, 100, 100]
"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
    
    @patch('chess_ai_project.src.chess_board_recognition.data_collection.screen_capture.pyautogui')
    def test_capture_with_region_selection(self, mock_pyautogui):
        """测试结合区域选择的截图功能"""
        # 模拟pyautogui
        mock_screenshot = Mock()
        mock_pyautogui.screenshot.return_value = mock_screenshot
        mock_pyautogui.size.return_value = (1920, 1080)
        
        capture = ScreenCaptureImpl(str(self.config_file))
        
        # 模拟区域选择
        with patch.object(capture.region_selector, 'get_selected_region') as mock_select:
            mock_select.return_value = (100, 100, 400, 300)
            
            # 选择区域
            region = capture.select_region()
            assert region == (100, 100, 400, 300)
            
            # 执行截图
            result = capture.manual_capture()
            assert result != ""
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)