"""
区域选择器模块

提供图形界面的区域选择功能。
"""

import json
import time
from typing import Tuple, Optional
from pathlib import Path
import pyautogui

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("警告: tkinter不可用，将使用控制台输入模式")

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
            # 检查是否可以创建GUI界面
            if not GUI_AVAILABLE:
                raise ImportError("tkinter不可用")
                
            import tkinter as tk
            from tkinter import messagebox
            
            # 创建全屏透明窗口
            root = tk.Tk()
            root.title("区域选择器")
            
            # 设置窗口属性
            try:
                root.attributes('-fullscreen', True)
                root.attributes('-alpha', 0.3)  # 设置透明度
                root.configure(bg='black')
                root.attributes('-topmost', True)  # 置顶显示
            except tk.TclError as e:
                self.log_warning(f"设置窗口属性失败: {e}")
                # 如果全屏失败，使用普通窗口
                root.state('zoomed')  # Windows下最大化
                root.configure(bg='black')
            
            # 获取屏幕尺寸
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            
            self.log_info(f"屏幕尺寸: {screen_width}x{screen_height}")
            
            # 创建画布
            canvas = tk.Canvas(
                root, 
                width=screen_width, 
                height=screen_height,
                bg='black',
                highlightthickness=0,
                cursor='crosshair'
            )
            canvas.pack(fill='both', expand=True)
            
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
                
                self.log_debug(f"开始选择: ({self.start_x}, {self.start_y})")
            
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
                    
                    self.log_info(f"选择完成: ({x1}, {y1}, {width}, {height})")
                    
                    # 验证区域大小
                    if width > 10 and height > 10:
                        self.selected_region = (x1, y1, width, height)
                        self.log_info(f"选择区域: {self.selected_region}")
                        root.quit()
                    else:
                        self.log_warning("选择区域太小，请重新选择")
                        try:
                            messagebox.showwarning("警告", "选择区域太小，请重新选择")
                        except Exception as e:
                            self.log_warning(f"显示警告对话框失败: {e}")
            
            def on_key_press(event):
                """键盘事件"""
                self.log_debug(f"按键: {event.keysym}")
                if event.keysym == 'Escape':
                    self.log_info("用户取消选择")
                    root.quit()
                elif event.keysym == 'Return':
                    # 使用当前选择的区域
                    if self.selected_region:
                        self.log_info("用户确认选择")
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
            
            # 添加额外的说明
            canvas.create_text(
                screen_width // 2, 100,
                text="选择完成后会自动关闭此窗口",
                fill='yellow', font=('Arial', 12)
            )
            
            self.log_info("开始GUI事件循环")
            
            # 设置超时机制，避免无限等待
            def timeout_handler():
                self.log_warning("区域选择超时，使用默认区域")
                root.quit()
            
            # 30秒超时
            root.after(30000, timeout_handler)
            
            # 运行界面
            root.mainloop()
            
            self.log_info("GUI事件循环结束")
            
            try:
                root.destroy()
            except Exception as e:
                self.log_warning(f"销毁窗口失败: {e}")
            
        except Exception as e:
            self.log_error(f"显示选择界面失败: {e}")
            # 如果GUI失败，提供控制台输入选项
            print(f"GUI界面启动失败: {e}")
            print("将使用控制台输入模式")
            self.selected_region = self.get_region_with_console_input()
    
    def get_selected_region(self) -> Tuple[int, int, int, int]:
        """
        获取选中区域
        
        Returns:
            区域坐标 (x, y, width, height)
        """
        self.log_info("启动区域选择")
        
        # 首先尝试加载已保存的配置
        saved_region = self.load_region_config()
        self.log_info(f"加载区域配置: {saved_region}")
        
        # 询问用户是否使用已保存的区域
        if saved_region != (0, 0, 800, 600):  # 不是默认值
            try:
                # 使用控制台询问而不是GUI对话框，避免GUI阻塞问题
                print(f"\n发现已保存的区域配置: {saved_region}")
                print("1. 使用已保存的区域配置")
                print("2. 重新选择区域")
                print("3. 手动输入区域坐标")
                
                while True:
                    choice = input("请选择 (1-3): ").strip()
                    
                    if choice == '1':
                        self.log_info(f"使用已保存的区域: {saved_region}")
                        return saved_region
                    elif choice == '2':
                        break  # 继续到区域选择界面
                    elif choice == '3':
                        return self.get_region_with_dialog()
                    else:
                        print("无效选择，请输入 1-3")
                        
            except Exception as e:
                self.log_warning(f"用户选择失败: {e}")
                # 发生错误时直接使用已保存的区域
                self.log_info(f"使用已保存的区域: {saved_region}")
                return saved_region
        
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
    
    def get_region_with_console_input(self) -> Tuple[int, int, int, int]:
        """
        通过控制台输入获取区域坐标
        
        Returns:
            区域坐标
        """
        try:
            # 获取屏幕尺寸作为参考
            screen_width, screen_height = pyautogui.size()
            print(f"\n屏幕尺寸: {screen_width} x {screen_height}")
            print("请输入截图区域坐标:")
            
            while True:
                try:
                    x = int(input(f"X坐标 (0-{screen_width}): "))
                    if 0 <= x <= screen_width:
                        break
                    print(f"X坐标必须在 0-{screen_width} 范围内")
                except ValueError:
                    print("请输入有效的数字")
            
            while True:
                try:
                    y = int(input(f"Y坐标 (0-{screen_height}): "))
                    if 0 <= y <= screen_height:
                        break
                    print(f"Y坐标必须在 0-{screen_height} 范围内")
                except ValueError:
                    print("请输入有效的数字")
            
            while True:
                try:
                    width = int(input(f"宽度 (1-{screen_width-x}): "))
                    if 1 <= width <= screen_width-x:
                        break
                    print(f"宽度必须在 1-{screen_width-x} 范围内")
                except ValueError:
                    print("请输入有效的数字")
            
            while True:
                try:
                    height = int(input(f"高度 (1-{screen_height-y}): "))
                    if 1 <= height <= screen_height-y:
                        break
                    print(f"高度必须在 1-{screen_height-y} 范围内")
                except ValueError:
                    print("请输入有效的数字")
            
            region = (x, y, width, height)
            self.log_info(f"通过控制台设置区域: {region}")
            return region
            
        except Exception as e:
            self.log_error(f"控制台输入失败: {e}")
            return (0, 0, 800, 600)

    def get_region_with_dialog(self) -> Tuple[int, int, int, int]:
        """
        通过对话框获取区域坐标
        
        Returns:
            区域坐标
        """
        try:
            if not GUI_AVAILABLE:
                self.log_warning("GUI不可用，使用控制台输入")
                return self.get_region_with_console_input()
                
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