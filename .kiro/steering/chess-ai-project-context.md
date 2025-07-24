# 象棋AI项目上下文信息

## 项目概述

这是一个综合性的中国象棋AI项目，包含三个主要模块：
1. **棋盘识别模块** - 使用计算机视觉识别实体棋盘
2. **象棋AI引擎** - 基于深度强化学习的对弈引擎
3. **实时分析系统** - 实时棋局分析和建议

## 项目目录结构

**重要：请始终使用以下现有的目录结构，不要创建重复的文件夹！**

```
chess-ai-kiro/                     # 项目根目录
├── .kiro/                         # Kiro AI助手配置
│   ├── specs/                     # 📋 项目规格文档 (需求、设计、任务)
│   │   ├── chess-board-recognition/    # 棋盘识别系统规格
│   │   ├── chinese-chess-ai-engine/    # AI引擎系统规格
│   │   └── real-time-analysis-system/  # 实时分析系统规格
│   ├── settings/                  # Kiro设置文件
│   └── steering/                  # 🎯 AI指导文档 (技术栈、结构等)
├── chess_ai_project/              # 🏗️ 主要工程代码
│   ├── src/                       # 源代码目录
│   │   ├── chess_board_recognition/    # 棋盘识别模块
│   │   ├── chinese_chess_ai_engine/    # AI引擎模块
│   │   └── real_time_analysis_system/  # 实时分析模块
│   ├── tests/                     # 🧪 测试代码
│   │   ├── chess_board_recognition/    # 棋盘识别测试
│   │   ├── chinese_chess_ai_engine/    # AI引擎测试
│   │   └── real_time_analysis_system/  # 实时分析测试
│   ├── configs/                   # ⚙️ 配置文件
│   │   ├── chess_board_recognition/    # 棋盘识别配置（要有详细注释）
│   │   ├── chinese_chess_ai_engine/    # AI引擎配置（要有详细注释）
│   │   ├── real_time_analysis_system/  # 实时分析配置（要有详细注释）
│   │   └── default.yaml           # 默认配置 (详细注释)
│   └── main.py                    # 主入口文件
├── data/                          # 📊 数据目录
│   ├── captures/                  # 屏幕截图数据
│   ├── annotations/               # 标注数据
│   └── sessions/                  # 对局会话数据
├── docs/                          # 📚 文档目录
│   ├── chess_board_recognition/   # 棋盘识别系统中各个子功能使用说明文档
│   ├── chinese_chess_ai_engine/   # AI引擎系统中各个子功能使用说明文档
│   └── real_time_analysis_system/ # 实时分析系统中各个子功能使用说明文档
├── models/                        # 🤖 模型文件目录
├── logs/                          # 📝 日志文件目录
│   ├── chess_board_recognition/   # 棋盘识别日志
│   ├── chinese_chess_ai_engine/   # AI引擎日志
│   └── real_time_analysis_system/ # 实时分析日志
├── .venv/                         # 🐍 Python虚拟环境
├── pyproject.toml                 # 📦 项目配置和依赖
├── requirements.txt               # 📋 依赖列表 (兼容性)
├── scripts.ps1                    # 🔧 Windows PowerShell脚本
├── Makefile                       # 🔧 Linux/Mac 自动化脚本
├── README.md                      # 📖 项目说明文档
├── LICENSE                        # 📄 MIT许可证
├── .gitignore                     # 🚫 Git忽略文件
└── .pre-commit-config.yaml        # 🔍 代码质量检查配置
```

## 各模块技术栈

### 1. 棋盘识别模块 (chess_board_recognition)

**技术栈**:
- OpenCV 4.12.0.88 - 计算机视觉库
- Ultralytics 8.3.167 - YOLO11实现，用于目标检测
- PyTorch 2.7.1 - 深度学习框架
- Pillow 11.3.0 - 图像处理库
- NumPy 2.2.6 - 数值计算库
- mss 10.0.0 - 屏幕截图工具

### 2. 象棋AI引擎 (chinese_chess_ai_engine)

**技术栈**:
- PyTorch 2.7.1 - 深度学习框架
- torchvision 0.22.1 - 计算机视觉工具包
- AlphaZero架构 - 强化学习算法
- MCTS - 蒙特卡洛树搜索
- FastAPI 0.116.1 - 现代Web框架
- uvicorn 0.35.0 - ASGI服务器
- pydantic 2.11.7 - 数据验证库
- YAML/JSON - 配置管理

### 3. 实时分析系统 (real_time_analysis_system)

**技术栈**:
- 集成上述两个模块
- WebSocket - 实时通信
- FastAPI - Web框架
- Rich 14.0.0 - 终端美化工具
- Click 8.2.1 - 命令行界面框架

## 开发规范

### 文件创建规则

1. **永远不要创建新的顶级模块目录**
   - 只使用现有的三个模块：`chess_board_recognition`、`chinese_chess_ai_engine`、`real_time_analysis_system`

2. **在现有模块内创建子模块**
   - 检查现有的子目录结构
   - 在适当的子目录中创建新文件

3. **测试文件位置**
   - 测试文件放在 `tests/` 对应的模块目录下
   - 保持与源代码相同的目录结构

4. **配置文件位置**
   - 配置文件放在 `configs/` 对应的模块目录下
   - 使用YAML格式作为首选

### 导入规范

```python
# 正确的导入方式
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine.move import Move
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine.chess_board import ChessBoard

# 模块内导入
from .rules_engine import ChessBoard, Move
from .config import ConfigManager
```

### 命名规范

- **文件名**: 使用snake_case，如 `chess_board.py`
- **类名**: 使用PascalCase，如 `ChessBoard`
- **函数名**: 使用snake_case，如 `make_move()`
- **常量**: 使用UPPER_CASE，如 `RED_KING`



## 重要提醒

### 对AI助手的指导

1. **检查现有结构**: 在创建任何新文件之前，先检查现有的目录结构
2. **使用现有模块**: 不要创建重复的模块或目录
3. **遵循命名规范**: 使用项目既定的命名规范
4. **查看已实现功能**: 检查已经实现的类和函数，避免重复实现
5. **参考现有代码**: 学习现有代码的风格和模式

### 常见错误避免

❌ **不要做**:
- 创建新的 `chess_ai_engine` 目录（已存在 `chinese_chess_ai_engine`）
- 在错误的位置创建文件
- 重复实现已存在的类或函数
- 忽略现有的配置和日志系统

✅ **应该做**:
- 使用现有的目录结构
- 检查已实现的功能
- 遵循项目规范
- 复用现有的工具和配置

## 技术栈总览

### 核心依赖
- Python 3.8+
- PyTorch 2.7.1 - 深度学习框架，支持GPU加速
- Ultralytics 8.3.167 - YOLO11实现，用于目标检测
- OpenCV 4.12.0.88 - 计算机视觉库
- NumPy 2.2.6 - 数值计算库
- FastAPI 0.116.1 - 现代Web框架
- uvicorn 0.35.0 - ASGI服务器
- Pandas 2.3.1 - 数据分析库
- SciPy 1.16.0 - 科学计算库
- scikit-learn 1.7.0 - 机器学习库

### 开发工具
- pytest 8.4.1 - 测试框架
- pytest-cov 6.2.1 - 覆盖率测试
- black - 代码格式化工具
- isort - 导入排序工具
- flake8 - 代码检查工具
- mypy - 类型检查工具
- Git - 版本控制

## 项目目标

最终目标是创建一个完整的象棋AI系统，能够：
1. 识别实体棋盘上的棋子
2. 提供专业级别的对弈能力
3. 实时分析棋局并给出建议
4. 支持人机对弈和AI训练

这个项目将展示计算机视觉、深度学习和强化学习在传统棋类游戏中的应用。