"""
区域选择器模块

提供图形界面的区域选择功能。
"""

import json
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
from typing import Tuple, Optional
from pathlib import Path
import pyautogui

from ..core.logger import LoggerMixin


class RegionSelector(LoggerMixin):
    """区域选择器类"""
    
    def __init__(self):
        """初始化区域选择器"""
        super().__init__()
        self.config_file = Path("./data/region_config.json")
        self.selected_region: Optional[Tuple[int, int, int, int]] = None
        self.log_info("区域选择器初始化完成")
    
    def show_selection_overlay(self) -> None:
        """显示选择覆盖层"""
        self.log_info("显示区域选择界面")
        
        try:
            # 创建全屏透明窗口
            root = tk.Tk()
            root.title("区域选择器")
            root.attributes('-fullscreen', True)
            root.attributes('-alpha', 0.3)  # 设置透明度
            root.configure(bg='black')
            
            # 获取屏幕尺寸
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            
            # 创建画布
            canvas = tk.Canvas(
                root, 
                width=screen_width, 
                height=screen_height,
                bg='black',
                highlightthickness=0
            )
            canvas.pack()
            
            # 选择状态变量
            self.start_x = 0
            self.start_y = 0
            self.rect_id = None
            self.is_selecting = False
            
            def on_button_press(event):
                """鼠标按下事件"""
                self.start_x = event.x
                self.start_y = event.y
                self.is_selecting = True
                
                # 删除之前的矩形
                if self.rect_id:
                    canvas.delete(self.rect_id)
            
            def on_mouse_drag(event):
                """鼠标拖拽事件"""
                if self.is_selecting:
                    # 删除之前的矩形
                    if self.rect_id:
                        canvas.delete(self.rect_id)
                    
                    # 绘制新矩形
                    self.rect_id = canvas.create_rectangle(
                        self.start_x, self.start_y, event.x, event.y,
                        outline='red', width=2, fill='', stipple='gray50'
                    )
            
            def on_button_release(event):
                """鼠标释放事件"""
                if self.is_selecting:
                    self.is_selecting = False
                    
                    # 计算选择区域
                    x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
                    x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
                    width, height = x2 - x1, y2 - y1
                    
                    # 验证区域大小
                    if width > 10 and height > 10:
                        self.selected_region = (x1, y1, width, height)
                        self.log_info(f"选择区域: {self.selected_region}")
                        root.quit()
                    else:
                        messagebox.showwarning("警告", "选择区域太小，请重新选择")
            
            def on_key_press(event):
                """键盘事件"""
                if event.keysym == 'Escape':
                    root.quit()
                elif event.keysym == 'Return':
                    # 使用当前选择的区域
                    if self.selected_region:
                        root.quit()
            
            # 绑定事件
            canvas.bind("<Button-1>", on_button_press)
            canvas.bind("<B1-Motion>", on_mouse_drag)
            canvas.bind("<ButtonRelease-1>", on_button_release)
            root.bind("<KeyPress>", on_key_press)
            root.focus_set()
            
            # 显示说明文字
            canvas.create_text(
                screen_width // 2, 50,
                text="拖拽鼠标选择截图区域，按ESC取消，按Enter确认",
                fill='white', font=('Arial', 16)
            )
            
            # 运行界面
            root.mainloop()
            root.destroy()
            
        except Exception as e:
            self.log_error(f"显示选择界面失败: {e}")
    
    def get_selected_region(self) -> Tuple[int, int, int, int]:
        """
        获取选中区域
        
        Returns:
            区域坐标 (x, y, width, height)
        """
        self.log_info("启动区域选择")
        
        # 首先尝试加载已保存的配置
        saved_region = self.load_region_config()
        
        # 询问用户是否使用已保存的区域
        if saved_region != (0, 0, 800, 600):  # 不是默认值
            try:
                root = tk.Tk()
                root.withdraw()  # 隐藏主窗口
                
                use_saved = messagebox.askyesno(
                    "区域选择",
                    f"发现已保存的区域配置: {saved_region}\n是否使用此配置？\n\n选择'否'将重新选择区域"
                )
                
                root.destroy()
                
                if use_saved:
                    self.log_info(f"使用已保存的区域: {saved_region}")
                    return saved_region
                    
            except Exception as e:
                self.log_warning(f"显示确认对话框失败: {e}")
        
        # 显示区域选择界面
        self.show_selection_overlay()
        
        # 返回选择的区域或默认区域
        if self.selected_region:
            return self.selected_region
        else:
            self.log_warning("未选择区域，使用默认区域")
            return (0, 0, 800, 600)
    
    def save_region_config(self, region: Tuple) -> None:
        """
        保存区域配置
        
        Args:
            region: 区域坐标
        """
        try:
            # 确保配置目录存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config = {
                "region": list(region),
                "screen_size": list(pyautogui.size()),
                "saved_time": str(time.time())
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.log_info(f"区域配置已保存: {region}")
            
        except Exception as e:
            self.log_error(f"保存区域配置失败: {e}")
    
    def load_region_config(self) -> Tuple[int, int, int, int]:
        """
        加载区域配置
        
        Returns:
            区域坐标
        """
        try:
            if not self.config_file.exists():
                self.log_info("区域配置文件不存在，使用默认区域")
                return (0, 0, 800, 600)
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            region = tuple(config.get("region", [0, 0, 800, 600]))
            saved_screen_size = tuple(config.get("screen_size", [0, 0]))
            current_screen_size = pyautogui.size()
            
            # 检查屏幕分辨率是否改变
            if saved_screen_size != current_screen_size:
                self.log_warning(
                    f"屏幕分辨率已改变: {saved_screen_size} -> {current_screen_size}，"
                    "建议重新选择区域"
                )
            
            self.log_info(f"加载区域配置: {region}")
            return region
            
        except Exception as e:
            self.log_error(f"加载区域配置失败: {e}")
            return (0, 0, 800, 600)
    
    def get_region_with_dialog(self) -> Tuple[int, int, int, int]:
        """
        通过对话框获取区域坐标
        
        Returns:
            区域坐标
        """
        try:
            root = tk.Tk()
            root.withdraw()
            
            # 获取屏幕尺寸作为参考
            screen_width, screen_height = pyautogui.size()
            
            # 输入对话框
            x = simpledialog.askinteger(
                "区域选择", 
                f"请输入X坐标 (0-{screen_width}):",
                initialvalue=0,
                minvalue=0,
                maxvalue=screen_width
            )
            
            if x is None:
                root.destroy()
                return (0, 0, 800, 600)
            
            y = simpledialog.askinteger(
                "区域选择",
                f"请输入Y坐标 (0-{screen_height}):",
                initialvalue=0,
                minvalue=0,
                maxvalue=screen_height
            )
            
            if y is None:
                root.destroy()
                return (0, 0, 800, 600)
            
            width = simpledialog.askinteger(
                "区域选择",
                f"请输入宽度 (1-{screen_width-x}):",
                initialvalue=800,
                minvalue=1,
                maxvalue=screen_width-x
            )
            
            if width is None:
                root.destroy()
                return (0, 0, 800, 600)
            
            height = simpledialog.askinteger(
                "区域选择",
                f"请输入高度 (1-{screen_height-y}):",
                initialvalue=600,
                minvalue=1,
                maxvalue=screen_height-y
            )
            
            root.destroy()
            
            if height is None:
                return (0, 0, 800, 600)
            
            region = (x, y, width, height)
            self.log_info(f"通过对话框设置区域: {region}")
            return region
            
        except Exception as e:
            self.log_error(f"对话框输入失败: {e}")
            return (0, 0, 800, 600)