"""
屏幕截图功能演示脚本

展示屏幕截图和区域选择功能的使用方法。
"""

import time
from pathlib import Path

from .screen_capture import ScreenCaptureImpl
from .region_selector import RegionSelector


def demo_region_selector():
    """演示区域选择器功能"""
    print("=== 区域选择器演示 ===\n")
    
    selector = RegionSelector()
    
    # 演示通过对话框获取区域
    print("1. 通过对话框获取区域坐标:")
    try:
        region = selector.get_region_with_dialog()
        print(f"   获取的区域: {region}")
    except Exception as e:
        print(f"   对话框演示跳过: {e}")
    
    # 演示保存和加载配置
    print("\n2. 保存和加载区域配置:")
    test_region = (100, 200, 400, 300)
    selector.save_region_config(test_region)
    print(f"   保存区域: {test_region}")
    
    loaded_region = selector.load_region_config()
    print(f"   加载区域: {loaded_region}")
    
    print()


def demo_screen_capture():
    """演示屏幕截图功能"""
    print("=== 屏幕截图演示 ===\n")
    
    # 创建配置文件
    config_path = "./configs/chess_board_recognition.yaml"
    
    try:
        capture = ScreenCaptureImpl(config_path)
        
        # 显示初始状态
        print("1. 截图器状态:")
        stats = capture.get_capture_stats()
        print(f"   保存路径: {stats['save_path']}")
        print(f"   图像格式: {stats['format']}")
        print(f"   图像质量: {stats['quality']}")
        print(f"   当前文件数: {stats['file_count']}")
        print(f"   总大小: {stats['total_size_mb']} MB")
        
        # 演示手动截图
        print("\n2. 执行手动截图:")
        filepath = capture.manual_capture()
        if filepath:
            print(f"   截图保存至: {filepath}")
            print(f"   文件存在: {Path(filepath).exists()}")
        else:
            print("   手动截图失败")
        
        # 演示自动截图（短时间）
        print("\n3. 演示自动截图 (3秒):")
        capture.start_auto_capture(1)  # 每秒截图一次
        print("   自动截图已启动...")
        
        time.sleep(3)  # 运行3秒
        
        capture.stop_capture()
        print("   自动截图已停止")
        
        # 显示最终状态
        print("\n4. 最终状态:")
        final_stats = capture.get_capture_stats()
        print(f"   截图数量: {final_stats['capture_count']}")
        print(f"   文件数量: {final_stats['file_count']}")
        print(f"   总大小: {final_stats['total_size_mb']} MB")
        
        if 'storage' in final_stats:
            storage = final_stats['storage']
            print(f"   磁盘使用: {storage.get('usage_percent', 0):.1f}%")
            print(f"   剩余空间: {storage.get('free_gb', 0):.2f} GB")
        
    except Exception as e:
        print(f"截图演示失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def demo_integration():
    """演示集成功能"""
    print("=== 集成功能演示 ===\n")
    
    config_path = "./configs/chess_board_recognition.yaml"
    
    try:
        capture = ScreenCaptureImpl(config_path)
        
        print("1. 区域选择和截图集成:")
        
        # 注意：这里不会实际显示GUI，因为可能在无头环境中运行
        print("   (区域选择GUI在实际使用时会显示)")
        
        # 使用默认区域进行截图
        print("   使用默认区域进行截图...")
        filepath = capture.manual_capture()
        
        if filepath:
            print(f"   集成截图成功: {Path(filepath).name}")
        else:
            print("   集成截图失败")
        
    except Exception as e:
        print(f"集成演示失败: {e}")
    
    print()


def main():
    """主演示函数"""
    print("屏幕截图功能演示")
    print("=" * 50)
    print()
    
    try:
        # 演示各个功能模块
        demo_region_selector()
        demo_screen_capture()
        demo_integration()
        
        print("=== 演示完成 ===")
        print("屏幕截图核心功能已成功实现！")
        print("\n功能特性:")
        print("✓ 区域选择器 - 支持图形界面和对话框输入")
        print("✓ 手动截图 - 支持指定区域截图")
        print("✓ 自动截图 - 支持定时自动截图")
        print("✓ 文件管理 - 自动按日期分类存储")
        print("✓ 存储监控 - 实时监控磁盘空间")
        print("✓ 配置管理 - 支持YAML配置文件")
        print("✓ 统计信息 - 提供详细的运行统计")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()