#!/usr/bin/env python3
"""
截屏GUI启动脚本

快速启动截屏GUI界面的便捷脚本。
"""

import sys
import os
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from chess_ai_project.src.chess_board_recognition.data_collection.capture_gui import launch_capture_gui
    
    def main():
        """主函数"""
        print("=" * 50)
        print("棋盘截屏工具 - GUI界面")
        print("=" * 50)
        print("正在启动GUI界面...")
        
        # 检查配置文件
        config_path = "chess_ai_project/configs/chess_board_recognition.yaml"
        if not Path(config_path).exists():
            print(f"警告: 配置文件不存在 {config_path}")
            print("将使用默认配置")
            config_path = None
        
        try:
            # 启动GUI
            launch_capture_gui(config_path)
            
        except KeyboardInterrupt:
            print("\n用户中断，退出程序")
        except Exception as e:
            print(f"启动失败: {e}")
            print("\n可能的解决方案:")
            print("1. 确保已安装必要依赖: pip install pillow pyautogui")
            print("2. 确保系统支持GUI界面")
            print("3. 检查配置文件是否正确")
            return 1
        
        return 0

    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已正确安装项目依赖")
    print("运行: pip install pillow pyautogui")
    sys.exit(1)