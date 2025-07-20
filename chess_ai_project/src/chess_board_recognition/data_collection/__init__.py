"""
数据收集模块

提供屏幕截图和数据收集功能，包括：
- 屏幕区域选择
- 自动和手动截图
- 截图文件管理
- 存储监控
"""

from .screen_capture import ScreenCaptureImpl
from .region_selector import RegionSelector

# GUI功能（可选导入，因为可能缺少GUI依赖）
try:
    from .capture_gui import CaptureGUI, launch_capture_gui
    GUI_AVAILABLE = True
    __all__ = [
        "ScreenCaptureImpl",
        "RegionSelector",
        "CaptureGUI",
        "launch_capture_gui",
    ]
except ImportError:
    GUI_AVAILABLE = False
    __all__ = [
        "ScreenCaptureImpl",
        "RegionSelector",
    ]