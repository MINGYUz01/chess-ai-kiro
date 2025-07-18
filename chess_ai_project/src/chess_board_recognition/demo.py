"""
棋盘识别系统演示脚本

展示系统的基本功能和接口使用方法。
"""

import numpy as np
from datetime import datetime
from pathlib import Path

from .core.interfaces import Detection, BoardState, CHESS_CLASSES
from .core.config import ConfigManager
from .core.logger import setup_logger
from .data_collection import ScreenCaptureImpl


def demo_core_interfaces():
    """演示核心接口功能"""
    print("=== 棋盘识别系统核心接口演示 ===\n")
    
    # 1. 演示Detection数据结构
    print("1. 创建检测结果对象:")
    detection = Detection(
        class_id=2,
        class_name="red_king",
        confidence=0.95,
        bbox=(100, 200, 150, 250),
        center=(125, 225)
    )
    print(f"   检测到: {detection.class_name}")
    print(f"   置信度: {detection.confidence}")
    print(f"   边界框: {detection.bbox}")
    print(f"   中心点: {detection.center}\n")
    
    # 2. 演示BoardState数据结构
    print("2. 创建棋局状态对象:")
    matrix = np.zeros((10, 9), dtype=int)
    matrix[0, 4] = 2  # 红帅在初始位置
    matrix[9, 4] = 9  # 黑将在初始位置
    
    board_state = BoardState(
        matrix=matrix,
        selected_piece=(0, 4),
        confidence=0.88,
        timestamp=datetime.now(),
        detections=[detection]
    )
    print(f"   棋局矩阵形状: {board_state.matrix.shape}")
    print(f"   选中棋子位置: {board_state.selected_piece}")
    print(f"   整体置信度: {board_state.confidence}")
    print(f"   检测结果数量: {len(board_state.detections)}\n")
    
    # 3. 演示棋子类别
    print("3. 棋子类别定义:")
    for class_id, class_name in list(CHESS_CLASSES.items())[:5]:
        print(f"   {class_id}: {class_name}")
    print("   ...\n")


def demo_config_manager():
    """演示配置管理功能"""
    print("=== 配置管理演示 ===\n")
    
    # 1. 创建配置管理器
    print("1. 创建配置管理器:")
    config_manager = ConfigManager()
    print("   配置管理器创建成功\n")
    
    # 2. 获取配置值
    print("2. 获取配置值:")
    confidence_threshold = config_manager.get("model.confidence_threshold")
    print(f"   模型置信度阈值: {confidence_threshold}")
    
    capture_interval = config_manager.get("capture.auto_interval")
    print(f"   自动截图间隔: {capture_interval}秒")
    
    training_epochs = config_manager.get("training.epochs")
    print(f"   训练轮数: {training_epochs}\n")
    
    # 3. 设置配置值
    print("3. 修改配置值:")
    config_manager.set("model.confidence_threshold", 0.7)
    new_threshold = config_manager.get("model.confidence_threshold")
    print(f"   新的置信度阈值: {new_threshold}\n")
    
    # 4. 配置验证
    print("4. 配置验证:")
    is_valid = config_manager.validate_config(config_manager.config)
    print(f"   配置有效性: {'有效' if is_valid else '无效'}\n")


def demo_logger():
    """演示日志系统功能"""
    print("=== 日志系统演示 ===\n")
    
    # 1. 创建日志记录器
    print("1. 创建日志记录器:")
    logger = setup_logger(
        name="demo_logger",
        level="INFO",
        console_output=True
    )
    print("   日志记录器创建成功\n")
    
    # 2. 记录不同级别的日志
    print("2. 记录日志消息:")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    print()


def demo_screen_capture():
    """演示屏幕截图功能"""
    print("=== 屏幕截图演示 ===\n")
    
    # 1. 创建截图器
    print("1. 创建屏幕截图器:")
    try:
        capture = ScreenCaptureImpl("./configs/chess_board_recognition.yaml")
        print("   截图器创建成功\n")
        
        # 2. 获取截图区域
        print("2. 获取截图区域:")
        region = capture.select_region()
        print(f"   截图区域: {region}\n")
        
        # 3. 获取统计信息
        print("3. 获取截图统计:")
        stats = capture.get_capture_stats()
        print(f"   统计信息: {stats}\n")
        
    except Exception as e:
        print(f"   截图器创建失败: {e}\n")


def demo_file_structure():
    """演示项目文件结构"""
    print("=== 项目结构演示 ===\n")
    
    base_path = Path(__file__).parent
    
    print("1. 核心模块结构:")
    core_path = base_path / "core"
    if core_path.exists():
        for file in core_path.glob("*.py"):
            print(f"   {file.name}")
    print()
    
    print("2. 功能模块结构:")
    modules = ["data_collection", "data_processing", "training", "inference", "system_management"]
    for module in modules:
        module_path = base_path / module
        if module_path.exists():
            print(f"   {module}/")
            for file in module_path.glob("*.py"):
                print(f"     {file.name}")
        else:
            print(f"   {module}/ (待实现)")
    print()


def main():
    """主演示函数"""
    print("棋盘识别系统演示")
    print("=" * 50)
    print()
    
    try:
        # 演示各个功能模块
        demo_core_interfaces()
        demo_config_manager()
        demo_logger()
        demo_screen_capture()
        demo_file_structure()
        
        print("=== 演示完成 ===")
        print("系统核心结构和接口已成功创建！")
        print("后续任务将实现具体的功能模块。")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()