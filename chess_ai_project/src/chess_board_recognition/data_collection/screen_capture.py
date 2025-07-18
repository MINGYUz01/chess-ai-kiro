"""
屏幕截图实现模块

提供屏幕截图的具体实现。
"""

import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import pyautogui
import shutil

from ..core.interfaces import ScreenCapture
from ..core.logger import LoggerMixin
from ..core.config import ConfigManager
from .region_selector import RegionSelector


class ScreenCaptureImpl(ScreenCapture, LoggerMixin):
    """屏幕截图实现类"""
    
    def __init__(self, config_path: str):
        """
        初始化屏幕截图器
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__()
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.region_selector = RegionSelector()
        
        # 截图状态
        self._is_capturing = False
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_count = 0
        self._start_time: Optional[datetime] = None
        
        # 获取配置
        self.save_path = Path(self.config_manager.get("capture.save_path", "./data/captures"))
        self.format = self.config_manager.get("capture.format", "jpg")
        self.quality = self.config_manager.get("capture.quality", 95)
        self.max_storage_gb = self.config_manager.get("capture.max_storage_gb", 10)
        
        # 创建保存目录
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # 设置pyautogui
        pyautogui.FAILSAFE = True  # 启用安全模式
        pyautogui.PAUSE = 0.1  # 设置操作间隔
        
        self.log_info("屏幕截图器初始化完成")
        self.log_info(f"保存路径: {self.save_path}")
        self.log_info(f"图像格式: {self.format}")
    
    def select_region(self) -> Tuple[int, int, int, int]:
        """
        选择截图区域
        
        Returns:
            区域坐标 (x, y, width, height)
        """
        self.log_info("启动区域选择器")
        
        try:
            # 使用区域选择器获取区域
            region = self.region_selector.get_selected_region()
            
            # 保存区域配置
            self.region_selector.save_region_config(region)
            self.config_manager.set("capture.region", list(region))
            
            self.log_info(f"选择的截图区域: {region}")
            return region
            
        except Exception as e:
            self.log_error(f"区域选择失败: {e}")
            # 返回默认区域
            default_region = self.config_manager.get("capture.region", [0, 0, 800, 600])
            return tuple(default_region)
    
    def start_auto_capture(self, interval: int) -> None:
        """
        开始自动截图
        
        Args:
            interval: 截图间隔（秒）
        """
        if self._is_capturing:
            self.log_warning("自动截图已在运行中")
            return
        
        self.log_info(f"开始自动截图，间隔: {interval}秒")
        
        # 检查存储空间
        if not self._check_storage_space():
            self.log_error("存储空间不足，无法开始截图")
            return
        
        self._is_capturing = True
        self._capture_count = 0
        self._start_time = datetime.now()
        
        # 启动截图线程
        self._capture_thread = threading.Thread(
            target=self._auto_capture_loop,
            args=(interval,),
            daemon=True
        )
        self._capture_thread.start()
    
    def manual_capture(self) -> str:
        """
        手动截图
        
        Returns:
            截图文件路径
        """
        self.log_info("执行手动截图")
        
        try:
            # 检查存储空间
            if not self._check_storage_space():
                self.log_error("存储空间不足，无法截图")
                return ""
            
            # 获取截图区域
            region = self.config_manager.get("capture.region", [0, 0, 800, 600])
            
            # 执行截图
            screenshot = pyautogui.screenshot(region=tuple(region))
            
            # 生成文件名
            filename = self._generate_filename()
            filepath = self.save_path / filename
            
            # 保存截图
            if self.format.lower() == 'jpg':
                screenshot.save(filepath, 'JPEG', quality=self.quality)
            else:
                screenshot.save(filepath, self.format.upper())
            
            self._capture_count += 1
            self.log_info(f"截图保存成功: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            self.log_error(f"手动截图失败: {e}")
            return ""
    
    def stop_capture(self) -> None:
        """停止截图"""
        if not self._is_capturing:
            self.log_info("没有正在运行的截图任务")
            return
        
        self.log_info("停止自动截图")
        self._is_capturing = False
        
        # 等待线程结束
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
        
        self.log_info("自动截图已停止")
    
    def get_capture_stats(self) -> Dict:
        """
        获取截图统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "is_capturing": self._is_capturing,
            "capture_count": self._capture_count,
            "save_path": str(self.save_path),
            "format": self.format,
            "quality": self.quality,
        }
        
        if self._start_time:
            stats["start_time"] = self._start_time.isoformat()
            stats["duration_seconds"] = (datetime.now() - self._start_time).total_seconds()
        
        # 获取存储信息
        try:
            total, used, free = shutil.disk_usage(self.save_path)
            stats["storage"] = {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "usage_percent": round((used / total) * 100, 2)
            }
        except Exception as e:
            self.log_warning(f"获取存储信息失败: {e}")
            stats["storage"] = {}
        
        # 获取截图文件统计
        try:
            image_files = list(self.save_path.glob(f"*.{self.format}"))
            stats["file_count"] = len(image_files)
            
            if image_files:
                total_size = sum(f.stat().st_size for f in image_files)
                stats["total_size_mb"] = round(total_size / (1024**2), 2)
            else:
                stats["total_size_mb"] = 0
                
        except Exception as e:
            self.log_warning(f"获取文件统计失败: {e}")
            stats["file_count"] = 0
            stats["total_size_mb"] = 0
        
        return stats
    
    def _auto_capture_loop(self, interval: int) -> None:
        """
        自动截图循环
        
        Args:
            interval: 截图间隔（秒）
        """
        self.log_info("自动截图循环开始")
        
        while self._is_capturing:
            try:
                # 检查存储空间
                if not self._check_storage_space():
                    self.log_error("存储空间不足，停止自动截图")
                    self._is_capturing = False
                    break
                
                # 执行截图
                filepath = self.manual_capture()
                if filepath:
                    self.log_debug(f"自动截图完成: {filepath}")
                
                # 等待间隔时间
                time.sleep(interval)
                
            except Exception as e:
                self.log_error(f"自动截图循环出错: {e}")
                time.sleep(1)  # 短暂等待后继续
        
        self.log_info("自动截图循环结束")
    
    def _generate_filename(self) -> str:
        """
        生成截图文件名
        
        Returns:
            文件名
        """
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        
        # 创建日期目录
        date_dir = self.save_path / date_str
        date_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        filename = f"screenshot_{date_str}_{time_str}.{self.format}"
        return date_str + "/" + filename
    
    def _check_storage_space(self) -> bool:
        """
        检查存储空间
        
        Returns:
            是否有足够空间
        """
        try:
            total, used, free = shutil.disk_usage(self.save_path)
            free_gb = free / (1024**3)
            
            if free_gb < 0.5:  # 少于500MB时警告
                self.log_warning(f"存储空间不足: 剩余 {free_gb:.2f}GB")
                return False
            
            # 检查是否超过最大存储限制
            image_files = list(self.save_path.glob(f"*.{self.format}"))
            if image_files:
                total_size = sum(f.stat().st_size for f in image_files)
                total_size_gb = total_size / (1024**3)
                
                if total_size_gb > self.max_storage_gb:
                    self.log_warning(f"截图文件总大小超过限制: {total_size_gb:.2f}GB > {self.max_storage_gb}GB")
                    # 可以在这里实现自动清理旧文件的逻辑
                    return False
            
            return True
            
        except Exception as e:
            self.log_error(f"检查存储空间失败: {e}")
            return True  # 检查失败时允许继续