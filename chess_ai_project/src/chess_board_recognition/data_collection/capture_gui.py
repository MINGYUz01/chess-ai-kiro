"""
截屏GUI界面模块

提供可视化的截屏控制界面，包括区域预览和按钮操作。
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
from typing import Optional, Tuple, Callable
from pathlib import Path
import pyautogui
from PIL import Image, ImageTk

from ..core.logger import LoggerMixin
from .screen_capture import ScreenCaptureImpl


class CaptureGUI(LoggerMixin):
    """截屏GUI控制界面"""
    
    def __init__(self, capture_instance: ScreenCaptureImpl):
        """
        初始化GUI界面
        
        Args:
            capture_instance: 截屏实例
        """
        super().__init__()
        self.capture = capture_instance
        self.root: Optional[tk.Tk] = None
        self.preview_label: Optional[tk.Label] = None
        self.status_label: Optional[tk.Label] = None
        self.capture_button: Optional[tk.Button] = None
        self.auto_button: Optional[tk.Button] = None
        self.region_button: Optional[tk.Button] = None
        self.stats_text: Optional[tk.Text] = None
        
        # 自动截屏状态
        self.auto_capture_thread: Optional[threading.Thread] = None
        self.auto_capture_running = False
        self.auto_interval = 2
        
        # 预览相关
        self.preview_image: Optional[ImageTk.PhotoImage] = None
        self.current_region: Tuple[int, int, int, int] = (0, 0, 800, 600)
        
        self.log_info("截屏GUI界面初始化完成")
    
    def create_gui(self) -> None:
        """创建GUI界面"""
        self.log_info("创建GUI界面")
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("棋盘截屏工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置窗口图标（如果有的话）
        try:
            # 可以在这里设置窗口图标
            pass
        except Exception as e:
            self.log_warning(f"设置窗口图标失败: {e}")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 创建左侧控制面板
        self._create_control_panel(main_frame)
        
        # 创建右侧预览面板
        self._create_preview_panel(main_frame)
        
        # 创建底部状态栏
        self._create_status_bar(main_frame)
        
        # 加载当前区域配置
        self._load_current_region()
        
        # 更新预览
        self._update_preview()
        
        # 更新统计信息
        self._update_stats()
        
        # 设置定时更新
        self._schedule_updates()
        
        self.log_info("GUI界面创建完成")
    
    def _create_control_panel(self, parent: ttk.Frame) -> None:
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 区域选择按钮
        self.region_button = ttk.Button(
            control_frame,
            text="选择截屏区域",
            command=self._select_region,
            width=20
        )
        self.region_button.pack(pady=5, fill=tk.X)
        
        # 手动截屏按钮
        self.capture_button = ttk.Button(
            control_frame,
            text="立即截屏",
            command=self._manual_capture,
            width=20
        )
        self.capture_button.pack(pady=5, fill=tk.X)
        
        # 自动截屏控制
        auto_frame = ttk.LabelFrame(control_frame, text="自动截屏", padding="5")
        auto_frame.pack(pady=10, fill=tk.X)
        
        # 间隔设置
        interval_frame = ttk.Frame(auto_frame)
        interval_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(interval_frame, text="间隔(秒):").pack(side=tk.LEFT)
        
        self.interval_var = tk.StringVar(value=str(self.auto_interval))
        interval_spinbox = ttk.Spinbox(
            interval_frame,
            from_=1, to=10,
            textvariable=self.interval_var,
            width=5,
            command=self._update_interval
        )
        interval_spinbox.pack(side=tk.RIGHT)
        
        # 自动截屏按钮
        self.auto_button = ttk.Button(
            auto_frame,
            text="开始自动截屏",
            command=self._toggle_auto_capture,
            width=20
        )
        self.auto_button.pack(pady=5, fill=tk.X)
        
        # 统计信息
        stats_frame = ttk.LabelFrame(control_frame, text="统计信息", padding="5")
        stats_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # 创建滚动文本框
        text_frame = ttk.Frame(stats_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(
            text_frame,
            height=8,
            width=30,
            wrap=tk.WORD,
            font=('Consolas', 9)
        )
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 刷新按钮
        ttk.Button(
            stats_frame,
            text="刷新统计",
            command=self._update_stats
        ).pack(pady=5)
    
    def _create_preview_panel(self, parent: ttk.Frame) -> None:
        """创建预览面板"""
        preview_frame = ttk.LabelFrame(parent, text="截屏预览", padding="10")
        preview_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # 创建预览标签
        self.preview_label = tk.Label(
            preview_frame,
            text="点击'选择截屏区域'来设置截屏区域\n然后预览将显示在这里",
            bg='lightgray',
            relief=tk.SUNKEN,
            bd=2
        )
        self.preview_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 预览控制按钮
        preview_control_frame = ttk.Frame(preview_frame)
        preview_control_frame.grid(row=1, column=0, pady=10)
        
        ttk.Button(
            preview_control_frame,
            text="刷新预览",
            command=self._update_preview
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            preview_control_frame,
            text="保存预览",
            command=self._save_preview
        ).pack(side=tk.LEFT, padx=5)
    
    def _create_status_bar(self, parent: ttk.Frame) -> None:
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(
            status_frame,
            text="就绪 - 等待操作",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def _load_current_region(self) -> None:
        """加载当前区域配置"""
        try:
            region = self.capture.config_manager.get("capture.region", [0, 0, 800, 600])
            self.current_region = tuple(region)
            self.log_info(f"加载区域配置: {self.current_region}")
        except Exception as e:
            self.log_error(f"加载区域配置失败: {e}")
            self.current_region = (0, 0, 800, 600)
    
    def _select_region(self) -> None:
        """选择截屏区域"""
        self._update_status("正在选择截屏区域...")
        self.log_info("启动区域选择")
        
        try:
            # 最小化当前窗口
            self.root.iconify()
            
            # 等待一下让窗口完全最小化
            self.root.after(500, self._do_region_selection)
            
        except Exception as e:
            self.log_error(f"区域选择失败: {e}")
            self._update_status(f"区域选择失败: {e}")
            messagebox.showerror("错误", f"区域选择失败: {e}")
    
    def _do_region_selection(self) -> None:
        """执行区域选择"""
        try:
            # 调用原有的区域选择功能
            region = self.capture.select_region()
            self.current_region = region
            
            # 恢复窗口
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            
            # 更新预览
            self._update_preview()
            self._update_status(f"区域选择完成: {region}")
            
        except Exception as e:
            self.log_error(f"区域选择执行失败: {e}")
            self.root.deiconify()
            self._update_status(f"区域选择失败: {e}")
            messagebox.showerror("错误", f"区域选择失败: {e}")
    
    def _manual_capture(self) -> None:
        """手动截屏"""
        self._update_status("正在截屏...")
        self.log_info("执行手动截屏")
        
        try:
            # 执行截屏
            filepath = self.capture.manual_capture()
            
            if filepath:
                self._update_status(f"截屏成功: {Path(filepath).name}")
                self._update_stats()
                self._update_preview()
                
                # 显示成功消息
                messagebox.showinfo("成功", f"截屏已保存至:\n{filepath}")
            else:
                self._update_status("截屏失败")
                messagebox.showerror("错误", "截屏失败，请检查设置")
                
        except Exception as e:
            self.log_error(f"手动截屏失败: {e}")
            self._update_status(f"截屏失败: {e}")
            messagebox.showerror("错误", f"截屏失败: {e}")
    
    def _toggle_auto_capture(self) -> None:
        """切换自动截屏状态"""
        if self.auto_capture_running:
            self._stop_auto_capture()
        else:
            self._start_auto_capture()
    
    def _start_auto_capture(self) -> None:
        """开始自动截屏"""
        self.log_info(f"开始自动截屏，间隔: {self.auto_interval}秒")
        
        try:
            self.auto_capture_running = True
            self.auto_button.config(text="停止自动截屏")
            self.capture_button.config(state="disabled")
            self.region_button.config(state="disabled")
            
            # 启动自动截屏线程
            self.auto_capture_thread = threading.Thread(
                target=self._auto_capture_loop,
                daemon=True
            )
            self.auto_capture_thread.start()
            
            self._update_status(f"自动截屏已开始，间隔: {self.auto_interval}秒")
            
        except Exception as e:
            self.log_error(f"启动自动截屏失败: {e}")
            self._update_status(f"启动自动截屏失败: {e}")
            messagebox.showerror("错误", f"启动自动截屏失败: {e}")
    
    def _stop_auto_capture(self) -> None:
        """停止自动截屏"""
        self.log_info("停止自动截屏")
        
        self.auto_capture_running = False
        self.auto_button.config(text="开始自动截屏")
        self.capture_button.config(state="normal")
        self.region_button.config(state="normal")
        
        self._update_status("自动截屏已停止")
    
    def _auto_capture_loop(self) -> None:
        """自动截屏循环"""
        self.log_info("自动截屏循环开始")
        
        while self.auto_capture_running:
            try:
                # 执行截屏
                filepath = self.capture.manual_capture()
                
                if filepath:
                    # 在主线程中更新UI
                    self.root.after(0, lambda: self._update_status(
                        f"自动截屏: {Path(filepath).name} ({datetime.now().strftime('%H:%M:%S')})"
                    ))
                    self.root.after(0, self._update_stats)
                
                # 等待间隔时间
                time.sleep(self.auto_interval)
                
            except Exception as e:
                self.log_error(f"自动截屏循环出错: {e}")
                self.root.after(0, lambda: self._update_status(f"自动截屏出错: {e}"))
                time.sleep(1)
        
        self.log_info("自动截屏循环结束")
    
    def _update_interval(self) -> None:
        """更新截屏间隔"""
        try:
            self.auto_interval = int(self.interval_var.get())
            self.log_info(f"截屏间隔更新为: {self.auto_interval}秒")
        except ValueError:
            self.interval_var.set(str(self.auto_interval))
    
    def _update_preview(self) -> None:
        """更新预览图像"""
        try:
            # 截取预览图像
            screenshot = pyautogui.screenshot(region=self.current_region)
            
            # 调整图像大小以适应预览区域
            preview_size = (400, 300)
            try:
                # 新版本PIL
                screenshot.thumbnail(preview_size, Image.Resampling.LANCZOS)
            except AttributeError:
                # 旧版本PIL
                screenshot.thumbnail(preview_size, Image.LANCZOS)
            
            # 转换为Tkinter可用的格式
            self.preview_image = ImageTk.PhotoImage(screenshot)
            
            # 更新预览标签
            self.preview_label.config(
                image=self.preview_image,
                text="",
                compound=tk.CENTER
            )
            
            self.log_debug("预览图像已更新")
            
        except Exception as e:
            self.log_error(f"更新预览失败: {e}")
            self.preview_label.config(
                image="",
                text=f"预览更新失败:\n{e}",
                compound=tk.CENTER
            )
    
    def _save_preview(self) -> None:
        """保存当前预览图像"""
        try:
            if self.preview_image:
                # 重新截取原始大小的图像
                screenshot = pyautogui.screenshot(region=self.current_region)
                
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"preview_{timestamp}.png"
                filepath = Path("./data/captures") / filename
                
                # 确保目录存在
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存图像
                screenshot.save(filepath)
                
                self._update_status(f"预览已保存: {filename}")
                messagebox.showinfo("成功", f"预览图像已保存至:\n{filepath}")
                
            else:
                messagebox.showwarning("警告", "没有可保存的预览图像")
                
        except Exception as e:
            self.log_error(f"保存预览失败: {e}")
            messagebox.showerror("错误", f"保存预览失败: {e}")
    
    def _update_stats(self) -> None:
        """更新统计信息"""
        try:
            stats = self.capture.get_capture_stats()
            
            # 格式化统计信息
            stats_text = f"""截屏统计信息
{'='*20}
截屏数量: {stats['capture_count']}
文件数量: {stats['file_count']}
总大小: {stats['total_size_mb']} MB
图像格式: {stats['format']}
图像质量: {stats['quality']}
保存路径: {Path(stats['save_path']).name}

当前区域: {self.current_region}
区域大小: {self.current_region[2]}×{self.current_region[3]}

"""
            
            # 添加存储信息
            if 'storage' in stats and stats['storage']:
                storage = stats['storage']
                stats_text += f"""存储信息
{'='*20}
磁盘使用: {storage.get('usage_percent', 0):.1f}%
剩余空间: {storage.get('free_gb', 0):.2f} GB
总空间: {storage.get('total_gb', 0):.2f} GB

"""
            
            # 添加运行时间信息
            if 'start_time' in stats:
                stats_text += f"""运行信息
{'='*20}
开始时间: {stats['start_time'][:19]}
运行时长: {stats.get('duration_seconds', 0):.0f} 秒
"""
            
            # 更新文本框
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text)
            
        except Exception as e:
            self.log_error(f"更新统计信息失败: {e}")
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, f"统计信息更新失败:\n{e}")
    
    def _update_status(self, message: str) -> None:
        """更新状态栏"""
        if self.status_label:
            self.status_label.config(text=f"{datetime.now().strftime('%H:%M:%S')} - {message}")
    
    def _schedule_updates(self) -> None:
        """安排定时更新"""
        # 每5秒更新一次统计信息
        self.root.after(5000, self._schedule_updates)
        
        # 如果不在自动截屏模式，更新预览
        if not self.auto_capture_running:
            self._update_stats()
    
    def run(self) -> None:
        """运行GUI界面"""
        self.log_info("启动GUI界面")
        
        try:
            self.create_gui()
            
            # 设置关闭事件处理
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            
            # 运行主循环
            self.root.mainloop()
            
        except Exception as e:
            self.log_error(f"GUI运行失败: {e}")
            raise
    
    def _on_closing(self) -> None:
        """窗口关闭事件处理"""
        self.log_info("GUI界面关闭")
        
        # 停止自动截屏
        if self.auto_capture_running:
            self._stop_auto_capture()
        
        # 销毁窗口
        if self.root:
            self.root.destroy()


def launch_capture_gui(config_path: str = "./configs/chess_board_recognition.yaml") -> None:
    """
    启动截屏GUI界面
    
    Args:
        config_path: 配置文件路径
    """
    try:
        # 创建截屏实例
        capture = ScreenCaptureImpl(config_path)
        
        # 创建GUI界面
        gui = CaptureGUI(capture)
        
        # 运行界面
        gui.run()
        
    except Exception as e:
        print(f"启动截屏GUI失败: {e}")
        raise


if __name__ == "__main__":
    launch_capture_gui()