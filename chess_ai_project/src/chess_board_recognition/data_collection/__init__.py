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

__all__ = [
    "ScreenCaptureImpl",
    "RegionSelector",
]