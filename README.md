# 中国象棋AI系统 (Chess AI Kiro)

一个基于深度学习的中国象棋AI系统，包含棋盘识别、AI引擎和实时分析功能。

## 🎯 项目概述

Chess AI Kiro 是一个综合性的中国象棋AI系统，由三个核心模块组成：

1. **棋盘识别系统** - 基于YOLO11的计算机视觉系统，用于识别屏幕上的象棋棋局
2. **象棋AI引擎** - 基于深度强化学习的高性能对弈系统，采用AlphaZero架构
3. **实时分析系统** - 整合前两个模块，提供实时的棋局分析和走法建议

## 🚀 主要特性

### 棋盘识别系统
- 🖼️ **智能屏幕截图** - 图形界面区域选择，支持自动/手动截图
- 📁 **文件管理** - 按日期自动分类存储，时间戳命名
- 💾 **存储监控** - 实时监控磁盘空间，防止存储溢出
- 🏷️ **数据标注** - 支持labelImg标注，17种棋子类别识别
- 🤖 **YOLO11训练** - 模型训练和推理，支持GPU加速
- ⚡ **高性能** - 单张图像识别时间 < 100ms
- 📊 **标准输出** - 10x9棋局矩阵，包含选中状态

### 象棋AI引擎
- 🧠 基于AlphaZero架构的神经网络
- 🔍 蒙特卡洛树搜索(MCTS)算法
- 💪 目标ELO等级分 > 2000
- ⚡ 单次分析时间 < 1秒
- 🔄 支持自对弈训练和模型评估

### 实时分析系统
- 👁️ 实时屏幕监控
- 📈 胜率计算和可视化
- 💡 走法建议和分析
- 🎨 直观的界面叠加显示
- ⚙️ 灵活的配置和控制

## 🛠️ 技术栈

- **深度学习**: PyTorch, YOLO11
- **计算机视觉**: OpenCV, PIL
- **神经网络**: ResNet + Attention机制
- **搜索算法**: 蒙特卡洛树搜索(MCTS)
- **API框架**: FastAPI
- **数据存储**: HDF5, SQLite
- **配置管理**: YAML, Pydantic
- **GUI**: Tkinter

## 📦 安装与环境配置

### 🔧 环境要求

- **Python**: >= 3.9 (推荐 3.10 或 3.11)
- **操作系统**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **内存**: >= 8GB RAM (推荐 16GB+)
- **存储**: >= 5GB 可用空间
- **GPU**: NVIDIA GPU with CUDA >= 11.8 (可选，用于加速训练和推理)

### 🚀 快速安装指南

#### Windows 用户 (推荐)

```powershell
# 1. 安装uv包管理器
pip install uv

# 2. 克隆项目
git clone https://github.com/MINGYUz01/chess-ai-kiro.git
cd chess-ai-kiro

# 3. 创建虚拟环境
uv venv

# 4. 激活虚拟环境
.venv\Scripts\activate

# 5. 安装项目依赖（已配置国内镜像加速）
uv pip install -r requirements.txt

# 6. 安装项目本身
uv pip install -e .

# 7. 验证安装
chess-ai-kiro info
```

#### Linux/macOS 用户

```bash
# 1. 安装uv包管理器
pip install uv

# 2. 克隆项目
git clone https://github.com/MINGYUz01/chess-ai-kiro.git
cd chess-ai-kiro

# 3. 创建虚拟环境
uv venv

# 4. 激活虚拟环境
source .venv/bin/activate

# 5. 安装项目依赖
uv pip install -r requirements.txt

# 6. 安装项目本身
uv pip install -e .

# 7. 验证安装
chess-ai-kiro info
```

### 🔧 开发环境安装

如果你想参与开发或运行测试，需要安装开发依赖：

#### Windows 用户
```powershell
# 安装开发依赖
.\scripts.ps1 install-dev

# 验证开发环境
.\scripts.ps1 test
```

#### Linux/macOS 用户
```bash
# 安装开发依赖
make install-dev

# 验证开发环境
make test
```

### 🎯 可选功能安装

#### GPU 加速支持
```bash
# 安装CUDA版本的PyTorch（根据你的CUDA版本选择）
# CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### ONNX 模型支持
```bash
uv pip install -e ".[onnx]"
```

#### TensorRT 加速支持
```bash
uv pip install -e ".[tensorrt]"
```

### ⚠️ 常见安装问题

#### 问题1: uv 命令未找到
```bash
# 解决方案：确保pip安装路径在PATH中
pip install --user uv
# 或者使用conda
conda install -c conda-forge uv
```

#### 问题2: 网络连接超时
```bash
# 解决方案：项目已默认配置国内镜像源
# 如果仍然遇到问题，可以尝试其他镜像源
uv pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
# 或者
uv pip install -i https://pypi.mirrors.ustc.edu.cn/simple -r requirements.txt
```

#### 问题3: CUDA版本不匹配
```bash
# 解决方案：检查CUDA版本
nvidia-smi
# 然后安装对应版本的PyTorch
```

## 🎮 详细使用指南

### 📋 使用前准备

1. **确保环境已激活**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

2. **验证安装**
   ```bash
   chess-ai-kiro info
   ```

### 🎯 完整使用流程

#### 阶段1: 棋盘识别系统

##### 1.1 数据收集 - 屏幕截图工具

**启动截图工具：**
```bash
# 启动交互式截图工具
chess-board-recognition capture

# 使用自定义配置
chess-board-recognition capture --config my_config.yaml

# 启用详细日志
chess-board-recognition capture --verbose
```

**截图工具功能菜单：**
```
截图工具选项:
1. 选择截图区域    # 图形界面选择棋盘区域
2. 手动截图        # 立即截取一张图片
3. 开始自动截图    # 定时自动截图
4. 查看统计信息    # 显示截图数量和存储状态
5. 退出
```

**详细操作步骤：**

1. **选择截图区域**
   ```bash
   # 选择选项1后会出现：
   # - 全屏透明覆盖层
   # - 用鼠标拖拽选择棋盘区域
   # - 按ESC取消，按Enter确认
   # - 区域会自动保存供后续使用
   ```

2. **手动截图**
   ```bash
   # 选择选项2立即截取当前区域
   # 文件自动保存到 data/captures/日期/screenshot_时间戳.jpg
   ```

3. **自动截图**
   ```bash
   # 选择选项3，输入截图间隔（秒）
   # 推荐间隔：2-5秒
   # 按Ctrl+C停止自动截图
   ```

4. **查看统计**
   ```bash
   # 显示：截图数量、文件大小、磁盘使用率等
   ```

**高级用法：**
```bash
# 运行演示脚本
python -m chess_ai_project.src.chess_board_recognition.data_collection.demo_capture

# 直接使用Python API
python -c "
from chess_ai_project.src.chess_board_recognition.data_collection import ScreenCaptureImpl
capture = ScreenCaptureImpl('./configs/chess_board_recognition.yaml')
filepath = capture.manual_capture()
print(f'截图保存至: {filepath}')
"
```

##### 1.2 数据标注
```bash
# 安装labelImg（如果还没安装）
pip install labelImg

# 启动标注工具
labelImg ./data/captures ./chess_ai_project/configs/classes.txt
```

**标注指南：**
- 标注所有可见的棋子
- 使用正确的类别名称（见配置文件）
- 确保边界框准确包围棋子

##### 1.3 模型训练
```bash
# 训练YOLO11模型
chess-board-recognition --mode train --data-dir ./data/annotations

# 查看训练进度
tensorboard --logdir ./runs
```

##### 1.4 模型测试
```bash
# 测试单张图像
chess-board-recognition --mode inference --image ./test_image.jpg

# 测试整个测试集
chess-board-recognition --mode inference --data-dir ./data/test
```

#### 阶段2: 象棋AI引擎

##### 2.1 启动API服务器
```bash
# 启动AI引擎服务（默认端口8000）
chess-ai-engine --mode server --port 8000

# 检查服务状态
curl http://localhost:8000/health
```

##### 2.2 开始训练AI模型
```bash
# 开始自对弈训练
chess-ai-engine --mode train --episodes 1000

# 监控训练进度
chess-ai-engine --mode train --episodes 1000 --verbose
```

##### 2.3 分析棋局
```bash
# 分析标准开局
chess-ai-engine --mode analyze --fen "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"

# 分析自定义棋局
chess-ai-engine --mode analyze --fen "你的FEN字符串"
```

#### 阶段3: 实时分析系统

##### 3.1 配置系统
```bash
# 复制并编辑配置文件
cp chess_ai_project/configs/default.yaml my_config.yaml
# 编辑 my_config.yaml 设置监控区域等参数
```

##### 3.2 启动实时分析
```bash
# 使用默认配置
real-time-analysis

# 使用自定义配置
real-time-analysis --config my_config.yaml

# 设置特定监控区域
real-time-analysis --region "100,100,800,600"
```

### 🔧 常用管理命令

#### Windows 用户
```powershell
# 查看系统状态
chess-ai-kiro info

# 运行测试
.\scripts.ps1 test

# 清理临时文件
.\scripts.ps1 clean

# 格式化代码
.\scripts.ps1 format

# 查看所有可用命令
.\scripts.ps1 help
```

#### Linux/macOS 用户
```bash
# 查看系统状态
chess-ai-kiro info

# 运行测试
make test

# 清理临时文件
make clean

# 格式化代码
make format

# 查看所有可用命令
make help
```

### 🎯 典型使用场景

#### 场景1: 开发者首次使用
```bash
# 1. 安装环境
git clone https://github.com/MINGYUz01/chess-ai-kiro.git
cd chess-ai-kiro
uv venv && .venv\Scripts\activate  # Windows
uv pip install -e .

# 2. 验证安装
chess-ai-kiro info

# 3. 运行测试
.\scripts.ps1 test  # Windows

# 4. 开始开发
.\scripts.ps1 format  # 格式化代码
```

#### 场景2: 训练自己的模型
```bash
# 1. 收集数据
chess-board-recognition --mode collect --region-select

# 2. 标注数据
labelImg ./data/captures

# 3. 训练模型
chess-board-recognition --mode train --data-dir ./data/annotations

# 4. 测试模型
chess-board-recognition --mode inference --image test.jpg
```

#### 场景3: 实时分析象棋对弈
```bash
# 1. 启动AI引擎
chess-ai-engine --mode server &

# 2. 配置监控区域
real-time-analysis --region "0,0,1920,1080"

# 3. 开始实时分析
# 系统会自动监控屏幕变化并提供走法建议
```

### ⚠️ 常见问题解决

#### 问题1: 命令未找到
```bash
# 确保虚拟环境已激活
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# 重新安装项目
uv pip install -e .
```

#### 问题2: 模型文件缺失
```bash
# 检查模型目录
ls models/

# 如果缺失，需要先训练模型
chess-board-recognition --mode train --data-dir ./data/annotations
```

#### 问题3: GPU不可用
```bash
# 检查CUDA安装
nvidia-smi

# 安装CUDA版本的PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📁 项目结构

```
chess-ai-kiro/
├── .kiro/                          # Kiro AI助手配置
│   ├── specs/                      # 项目规格文档
│   └── steering/                   # AI指导文档
├── chess_ai_project/               # 主要工程代码
│   ├── src/                        # 源代码
│   │   ├── chess_board_recognition/    # 棋盘识别模块
│   │   ├── chinese_chess_ai_engine/    # AI引擎模块
│   │   └── real_time_analysis_system/  # 实时分析模块
│   ├── tests/                      # 测试代码
│   └── configs/                    # 配置文件
├── data/                           # 数据目录
├── models/                         # 模型文件
├── docs/                           # 文档
└── pyproject.toml                  # 项目配置
```

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/chess_board_recognition/

# 运行覆盖率测试
pytest --cov=chess_ai_project --cov-report=html
```

## 📖 文档

详细文档请访问：[https://chess-ai-kiro.readthedocs.io/](https://chess-ai-kiro.readthedocs.io/)

## 🤝 贡献

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 开发环境设置

#### Windows用户（推荐使用PowerShell脚本）
```powershell
# 安装开发依赖
.\scripts.ps1 install-dev

# 运行代码格式化
.\scripts.ps1 format

# 运行代码检查
.\scripts.ps1 lint

# 运行测试
.\scripts.ps1 test

# 查看所有可用命令
.\scripts.ps1 help
```

#### Linux/Mac用户（使用Makefile）
```bash
# 安装开发依赖
make install-dev

# 运行代码格式化
make format

# 运行代码检查
make lint

# 运行测试
make test

# 查看所有可用命令
make help
```

#### 手动命令（所有平台）
```bash
# 安装开发依赖
uv pip install -e ".[dev]"

# 安装pre-commit钩子
pre-commit install

# 运行代码格式化
black chess_ai_project/
isort chess_ai_project/

# 运行类型检查
mypy chess_ai_project/
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Ultralytics](https://ultralytics.com/) - YOLO实现
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架

## 📞 联系我们

- 项目主页: [https://github.com/MINGYUz01/chess-ai-kiro](https://github.com/MINGYUz01/chess-ai-kiro)
- 问题反馈: [https://github.com/MINGYUz01/chess-ai-kiro/issues](https://github.com/MINGYUz01/chess-ai-kiro/issues)
- 邮箱: team@chess-ai-kiro.com

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！