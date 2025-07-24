#!/usr/bin/env python3
"""
核心数据模型演示脚本

展示Move类和ChessBoard类的基本功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from chess_ai_project.src.chinese_chess_ai_engine.rules_engine.move import Move
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine.chess_board import ChessBoard
from chess_ai_project.src.chinese_chess_ai_engine.config.config_manager import ConfigManager
from chess_ai_project.src.chinese_chess_ai_engine.utils.logger import setup_logger


def demo_move_class():
    """演示Move类的功能"""
    print("=" * 50)
    print("Move类功能演示")
    print("=" * 50)
    
    # 创建一个走法
    move = Move(
        from_pos=(9, 4),  # 红帅的初始位置
        to_pos=(8, 4),    # 向前一步
        piece=ChessBoard.RED_KING
    )
    
    print(f"走法对象: {move}")
    print(f"起始位置: {move.from_pos}")
    print(f"目标位置: {move.to_pos}")
    print(f"棋子类型: {move.piece}")
    
    # 坐标记法
    coord_notation = move.to_coordinate_notation()
    print(f"坐标记法: {coord_notation}")
    
    # 中文记法
    chinese_notation = move.to_chinese_notation()
    print(f"中文记法: {chinese_notation}")
    
    # 从坐标记法创建走法
    move2 = Move.from_coordinate_notation("e9e8", ChessBoard.RED_KING)
    print(f"从坐标记法创建: {move2}")
    
    print()


def demo_chess_board_class():
    """演示ChessBoard类的功能"""
    print("=" * 50)
    print("ChessBoard类功能演示")
    print("=" * 50)
    
    # 创建初始棋盘
    board = ChessBoard()
    print("初始棋盘:")
    print(board.to_visual_string())
    print()
    
    # FEN格式
    fen = board.to_fen()
    print(f"FEN格式: {fen}")
    print()
    
    # 执行一个走法（兵前进）
    move = Move((6, 0), (5, 0), ChessBoard.RED_PAWN)
    print(f"执行走法: {move.to_coordinate_notation()} ({move.to_chinese_notation()})")
    
    new_board = board.make_move(move)
    print("走法后的棋盘:")
    print(new_board.to_visual_string())
    print()
    
    # 撤销走法
    undone_board = new_board.undo_move()
    print("撤销走法后的棋盘:")
    print(undone_board.to_visual_string())
    print()
    
    # JSON序列化
    json_str = new_board.to_json()
    print("JSON格式 (前200字符):")
    print(json_str[:200] + "...")
    print()


def demo_config_manager():
    """演示配置管理器的功能"""
    print("=" * 50)
    print("配置管理器功能演示")
    print("=" * 50)
    
    # 创建配置管理器
    config_manager = ConfigManager("chess_ai_project/configs/chinese_chess_ai_engine")
    
    # 获取各种配置
    mcts_config = config_manager.get_mcts_config()
    print(f"MCTS配置 - 模拟次数: {mcts_config.num_simulations}")
    print(f"MCTS配置 - 探索常数: {mcts_config.c_puct}")
    print(f"MCTS配置 - 时间限制: {mcts_config.time_limit}秒")
    print()
    
    model_config = config_manager.get_model_config()
    print(f"模型配置 - 输入通道: {model_config.input_channels}")
    print(f"模型配置 - ResNet块数: {model_config.num_blocks}")
    print(f"模型配置 - 隐藏层通道: {model_config.hidden_channels}")
    print()
    
    ai_config = config_manager.get_ai_config()
    print(f"AI配置 - 搜索时间: {ai_config.search_time}秒")
    print(f"AI配置 - 最大模拟次数: {ai_config.max_simulations}")
    print(f"AI配置 - 难度级别: {ai_config.difficulty_level}")
    print()


def demo_logger():
    """演示日志系统的功能"""
    print("=" * 50)
    print("日志系统功能演示")
    print("=" * 50)
    
    # 设置日志记录器
    logger = setup_logger(
        name='demo',
        level='INFO',
        log_file='demo.log',
        log_dir='logs/chinese_chess_ai_engine',
        console_output=True
    )
    
    # 记录不同级别的日志
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    
    print("日志已记录到控制台和文件")
    print()


def main():
    """主函数"""
    print("中国象棋AI引擎 - 核心数据模型演示")
    print("=" * 60)
    print()
    
    try:
        # 演示各个组件
        demo_move_class()
        demo_chess_board_class()
        demo_config_manager()
        demo_logger()
        
        print("=" * 60)
        print("演示完成！核心数据模型工作正常。")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()