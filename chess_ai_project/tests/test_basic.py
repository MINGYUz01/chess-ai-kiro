"""
基础测试模块

测试项目的基本功能和导入。
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_project_import():
    """测试项目主模块是否可以正常导入"""
    try:
        import chess_ai_project
        assert chess_ai_project.__version__ == "0.1.0"
        assert chess_ai_project.__author__ == "Chess AI Kiro Team"
    except ImportError as e:
        pytest.fail(f"无法导入chess_ai_project模块: {e}")


def test_submodules_import():
    """测试子模块是否可以正常导入"""
    try:
        from chess_ai_project.src import chess_board_recognition
        from chess_ai_project.src import chinese_chess_ai_engine
        from chess_ai_project.src import real_time_analysis_system
        
        # 检查版本信息
        assert chess_board_recognition.__version__ == "0.1.0"
        assert chinese_chess_ai_engine.__version__ == "0.1.0"
        assert real_time_analysis_system.__version__ == "0.1.0"
        
    except ImportError as e:
        pytest.fail(f"无法导入子模块: {e}")


def test_config_file_exists():
    """测试配置文件是否存在"""
    config_path = project_root / "chess_ai_project" / "configs" / "default.yaml"
    assert config_path.exists(), "默认配置文件不存在"


def test_main_entry_points():
    """测试主入口文件是否存在"""
    main_files = [
        "chess_ai_project/main.py",
        "chess_ai_project/src/chess_board_recognition/main.py",
        "chess_ai_project/src/chinese_chess_ai_engine/main.py",
        "chess_ai_project/src/real_time_analysis_system/main.py",
    ]
    
    for main_file in main_files:
        file_path = project_root / main_file
        assert file_path.exists(), f"主入口文件 {main_file} 不存在"


def test_directory_structure():
    """测试项目目录结构是否正确"""
    expected_dirs = [
        "chess_ai_project",
        "chess_ai_project/src",
        "chess_ai_project/src/chess_board_recognition",
        "chess_ai_project/src/chinese_chess_ai_engine", 
        "chess_ai_project/src/real_time_analysis_system",
        "chess_ai_project/tests",
        "chess_ai_project/configs",
        "data",
        "models",
        "logs",
    ]
    
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"目录 {dir_path} 不存在"
        assert full_path.is_dir(), f"{dir_path} 不是目录"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])