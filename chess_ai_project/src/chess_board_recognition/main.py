"""
棋盘识别系统主入口

提供命令行接口和主要功能入口。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from .core.config import ConfigManager
from .core.logger import setup_logger, configure_logging_from_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="棋盘识别系统 - 基于YOLO11的中国象棋棋局识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 启动截图工具（命令行模式）
  python -m chess_ai_project.src.chess_board_recognition.main capture

  # 启动截图工具（GUI模式）
  python -m chess_ai_project.src.chess_board_recognition.main capture-gui

  # 训练模型
  python -m chess_ai_project.src.chess_board_recognition.main train --data data.yaml --epochs 100

  # 运行推理
  python -m chess_ai_project.src.chess_board_recognition.main predict --source image.jpg

  # 验证模型
  python -m chess_ai_project.src.chess_board_recognition.main validate --model best.pt
        """
    )
    
    parser.add_argument(
        "command",
        choices=["capture", "capture-gui", "train", "predict", "validate", "export"],
        help="要执行的命令"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="./configs/chess_board_recognition.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="模型文件路径"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        help="数据配置文件路径"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="输入源（图像、视频或目录路径）"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="训练轮数"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        help="批次大小"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出目录"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        
        # 设置日志
        logger = configure_logging_from_config(config)
        
        if args.verbose:
            logger.setLevel("DEBUG")
        
        logger.info(f"棋盘识别系统启动 - 命令: {args.command}")
        logger.info(f"配置文件: {args.config}")
        
        # 根据命令执行相应功能
        if args.command == "capture":
            run_capture(config, args, logger)
        elif args.command == "capture-gui":
            run_capture_gui(config, args, logger)
        elif args.command == "train":
            run_training(config, args, logger)
        elif args.command == "predict":
            run_prediction(config, args, logger)
        elif args.command == "validate":
            run_validation(config, args, logger)
        elif args.command == "export":
            run_export(config, args, logger)
        
        logger.info("任务完成")
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


def run_capture(config, args, logger):
    """运行截图功能"""
    logger.info("启动截图工具...")
    
    try:
        from .data_collection.screen_capture import ScreenCaptureImpl
        
        # 创建截图器
        capture = ScreenCaptureImpl(args.config)
        
        # 显示当前状态
        stats = capture.get_capture_stats()
        print(f"保存路径: {stats['save_path']}")
        print(f"图像格式: {stats['format']}")
        print(f"当前文件数: {stats['file_count']}")
        
        # 提供操作选项
        print("\n截图工具选项:")
        print("1. 选择截图区域")
        print("2. 手动截图")
        print("3. 开始自动截图")
        print("4. 查看统计信息")
        print("5. 退出")
        
        while True:
            try:
                choice = input("\n请选择操作 (1-5): ").strip()
                
                if choice == '1':
                    print("启动区域选择器...")
                    region = capture.select_region()
                    print(f"选择的区域: {region}")
                    
                elif choice == '2':
                    print("执行手动截图...")
                    filepath = capture.manual_capture()
                    if filepath:
                        print(f"截图保存至: {filepath}")
                    else:
                        print("截图失败")
                        
                elif choice == '3':
                    interval = input("请输入截图间隔(秒，默认2): ").strip()
                    try:
                        interval = int(interval) if interval else 2
                    except ValueError:
                        interval = 2
                    
                    print(f"开始自动截图，间隔: {interval}秒")
                    print("按 Ctrl+C 停止自动截图")
                    
                    capture.start_auto_capture(interval)
                    
                    try:
                        while capture._is_capturing:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n停止自动截图...")
                        capture.stop_capture()
                        
                elif choice == '4':
                    stats = capture.get_capture_stats()
                    print(f"\n=== 截图统计 ===")
                    print(f"截图数量: {stats['capture_count']}")
                    print(f"文件数量: {stats['file_count']}")
                    print(f"总大小: {stats['total_size_mb']} MB")
                    if 'storage' in stats:
                        storage = stats['storage']
                        print(f"磁盘使用: {storage.get('usage_percent', 0):.1f}%")
                        print(f"剩余空间: {storage.get('free_gb', 0):.2f} GB")
                        
                elif choice == '5':
                    print("退出截图工具")
                    break
                    
                else:
                    print("无效选择，请输入 1-5")
                    
            except KeyboardInterrupt:
                print("\n退出截图工具")
                break
            except Exception as e:
                print(f"操作失败: {e}")
                
    except Exception as e:
        logger.error(f"截图工具启动失败: {e}")
        print(f"错误: {e}")


def run_capture_gui(config, args, logger):
    """运行截图GUI界面"""
    logger.info("启动截图GUI界面...")
    
    try:
        from .data_collection.capture_gui import launch_capture_gui
        
        print("启动截图GUI界面...")
        print("GUI界面将提供可视化的截图控制功能")
        
        # 启动GUI界面
        launch_capture_gui(args.config)
        
    except ImportError as e:
        logger.error(f"GUI依赖库不可用: {e}")
        print(f"错误: GUI功能需要tkinter和PIL库支持")
        print("请安装必要的依赖: pip install pillow")
        print("如果tkinter不可用，请使用命令行模式: capture")
        
    except Exception as e:
        logger.error(f"截图GUI启动失败: {e}")
        print(f"错误: {e}")
        print("如果GUI启动失败，请使用命令行模式: capture")


def run_training(config, args, logger):
    """运行训练功能"""
    logger.info("启动模型训练...")
    
    if not args.data:
        raise ValueError("训练模式需要指定数据配置文件 (--data)")
    
    print("训练功能将在后续任务中实现")


def run_prediction(config, args, logger):
    """运行预测功能"""
    logger.info("启动模型预测...")
    
    if not args.source:
        raise ValueError("预测模式需要指定输入源 (--source)")
    
    print("预测功能将在后续任务中实现")


def run_validation(config, args, logger):
    """运行验证功能"""
    logger.info("启动模型验证...")
    
    if not args.model:
        raise ValueError("验证模式需要指定模型文件 (--model)")
    
    print("验证功能将在后续任务中实现")


def run_export(config, args, logger):
    """运行模型导出功能"""
    logger.info("启动模型导出...")
    
    if not args.model:
        raise ValueError("导出模式需要指定模型文件 (--model)")
    
    print("导出功能将在后续任务中实现")


if __name__ == "__main__":
    main()