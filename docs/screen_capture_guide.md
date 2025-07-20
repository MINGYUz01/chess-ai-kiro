# 屏幕截图功能使用指南

本指南详细介绍了棋盘识别系统中屏幕截图功能的使用方法。

## 🎯 功能概述

屏幕截图功能是棋盘识别系统的核心组件，用于收集训练数据。主要特性包括：

- 🖼️ **智能区域选择** - 图形界面直观选择棋盘区域
- 📸 **多种截图模式** - 支持手动和自动截图
- 📁 **智能文件管理** - 按日期自动分类存储
- 💾 **存储监控** - 实时监控磁盘空间使用
- ⚙️ **配置管理** - 支持YAML配置文件
- 📊 **统计信息** - 提供详细的运行统计

## 🚀 快速开始

### 启动截图工具

**命令行模式：**
```bash
# 使用默认配置启动
chess-board-recognition capture

# 使用自定义配置
chess-board-recognition capture --config my_config.yaml

# 启用详细日志
chess-board-recognition capture --verbose
```

**GUI模式（推荐）：**
```bash
# 启动GUI界面
chess-board-recognition capture-gui

# 使用自定义配置启动GUI
chess-board-recognition capture-gui --config my_config.yaml

# 或者使用快捷启动脚本
python chess_ai_project\src\chess_board_recognition\data_collection\launch_capture_gui.py
```

### 使用脚本启动

**Windows用户：**
```cmd
REM 启动GUI界面（推荐）
启动截屏GUI.bat

REM 或者使用Python命令
python chess_ai_project\src\chess_board_recognition\data_collection\launch_capture_gui.py

REM 命令行模式
python -m chess_ai_project.src.chess_board_recognition.main capture
```

**Linux/macOS用户：**
```bash
# 启动截图工具
make capture

# 运行演示
make demo-capture
```

## 📋 功能详解

### GUI界面模式（推荐）

启动GUI界面后，你将看到一个直观的可视化控制面板：

**主要功能区域：**
- **控制面板**：包含所有操作按钮和设置选项
- **预览面板**：实时显示截屏区域的预览图像
- **统计信息**：显示详细的截屏统计和系统状态
- **状态栏**：显示当前操作状态和时间信息

**GUI界面优势：**
- 🖼️ **实时预览** - 可以看到截屏区域的实时画面
- 🎛️ **可视化控制** - 所有功能都有对应的按钮，操作直观
- 📊 **实时统计** - 统计信息自动更新，无需手动查询
- 🔄 **自动截屏监控** - 自动截屏时可以实时查看状态和停止
- 💾 **预览保存** - 可以直接保存当前预览图像

### 命令行模式

启动命令行模式后，会显示以下功能菜单：

```
截图工具选项:
1. 选择截图区域    # 图形界面选择棋盘区域
2. 手动截图        # 立即截取一张图片
3. 开始自动截图    # 定时自动截图
4. 查看统计信息    # 显示截图数量和存储状态
5. 退出
```

### 1. 选择截图区域

选择此选项后会启动图形界面区域选择器：

**操作步骤：**
1. 屏幕会显示全屏透明覆盖层
2. 用鼠标拖拽选择棋盘区域
3. 按 `ESC` 取消选择
4. 按 `Enter` 确认选择
5. 选择的区域会自动保存供后续使用

**注意事项：**
- 确保选择的区域完整包含棋盘
- 避免选择过大的区域以减少文件大小
- 区域配置会保存到 `data/region_config.json`

### 2. 手动截图

立即截取当前选定区域的图像：

**特性：**
- 使用已保存的区域配置
- 自动生成时间戳文件名
- 按日期创建子目录
- 支持多种图像格式（JPG、PNG等）

**文件命名规则：**
```
data/captures/YYYYMMDD/screenshot_YYYYMMDD_HHMMSS.jpg
```

### 3. 开始自动截图

启动定时自动截图功能：

**配置选项：**
- 截图间隔：1-10秒可调（推荐2-5秒）
- 自动停止：按 `Ctrl+C` 停止
- 存储监控：自动检查磁盘空间

**使用建议：**
- 对弈过程中使用2-3秒间隔
- 快棋对局使用1-2秒间隔
- 慢棋对局使用3-5秒间隔

### 4. 查看统计信息

显示详细的截图统计信息：

```
=== 截图统计 ===
截图数量: 25
文件数量: 25
总大小: 15.6 MB
磁盘使用: 45.2%
剩余空间: 128.5 GB
```

## ⚙️ 配置文件

### 默认配置文件位置

```
chess_ai_project/configs/chess_board_recognition.yaml
```

### 配置文件结构

```yaml
# 截图配置
capture:
  region: [0, 0, 800, 600]  # x, y, width, height
  auto_interval: 2          # 自动截图间隔（秒）
  save_path: "./data/captures"
  format: "jpg"             # 图像格式
  quality: 95               # 图像质量 (1-100)
  max_storage_gb: 10        # 最大存储空间（GB）

# 日志配置
logging:
  level: "INFO"
  file: "./logs/chess_board_recognition.log"
  console_output: true
```

### 自定义配置

创建自定义配置文件：

```bash
# 复制默认配置
cp chess_ai_project/configs/chess_board_recognition.yaml my_config.yaml

# 编辑配置
# 修改 save_path、format、quality 等参数

# 使用自定义配置
chess-board-recognition capture --config my_config.yaml
```

## 🔧 高级用法

### Python API 使用

```python
from chess_ai_project.src.chess_board_recognition.data_collection import ScreenCaptureImpl

# 创建截图器
capture = ScreenCaptureImpl('./configs/chess_board_recognition.yaml')

# 手动截图
filepath = capture.manual_capture()
print(f'截图保存至: {filepath}')

# 自动截图
capture.start_auto_capture(interval=2)
# ... 等待一段时间
capture.stop_capture()

# 获取统计信息
stats = capture.get_capture_stats()
print(f'截图数量: {stats["capture_count"]}')
```

### 区域选择器 API

```python
from chess_ai_project.src.chess_board_recognition.data_collection import RegionSelector

# 创建区域选择器
selector = RegionSelector()

# 通过对话框输入坐标
region = selector.get_region_with_dialog()

# 保存区域配置
selector.save_region_config(region)

# 加载区域配置
saved_region = selector.load_region_config()
```

## 📁 文件组织结构

截图文件按以下结构组织：

```
data/
├── captures/                 # 截图保存目录
│   ├── 20250118/            # 按日期分类
│   │   ├── screenshot_20250118_090000.jpg
│   │   ├── screenshot_20250118_090002.jpg
│   │   └── ...
│   └── 20250119/
│       └── ...
├── region_config.json       # 区域配置文件
└── ...
```

## 🛠️ 故障排除

### 常见问题

**问题1: 区域选择界面不显示**
```bash
# 解决方案：检查显示设置
# 确保没有多显示器冲突
# 尝试重新启动应用程序
```

**问题2: 截图文件过大**
```yaml
# 解决方案：调整配置文件
capture:
  format: "jpg"      # 使用JPG格式
  quality: 85        # 降低质量设置
```

**问题3: 存储空间不足**
```bash
# 解决方案：清理旧文件或增加存储限制
# 手动清理：删除 data/captures/ 中的旧文件
# 配置清理：修改 max_storage_gb 参数
```

**问题4: 自动截图停止**
```bash
# 可能原因：
# 1. 存储空间不足
# 2. 权限问题
# 3. 系统资源不足

# 解决方案：
# 检查日志文件：logs/chess_board_recognition.log
# 重新启动应用程序
```

### 调试模式

启用详细日志进行调试：

```bash
# 启用详细日志
chess-board-recognition capture --verbose

# 查看日志文件
tail -f logs/chess_board_recognition.log
```

## 📊 性能优化

### 推荐设置

**高质量数据收集：**
```yaml
capture:
  format: "png"
  quality: 100
  auto_interval: 3
```

**快速数据收集：**
```yaml
capture:
  format: "jpg"
  quality: 85
  auto_interval: 1
```

**存储优化：**
```yaml
capture:
  format: "jpg"
  quality: 75
  max_storage_gb: 5
```

### 系统要求

- **内存**: 至少 2GB 可用内存
- **存储**: 至少 1GB 可用空间
- **CPU**: 支持多线程处理
- **显示**: 支持图形界面显示

## 🔗 相关链接

- [项目主页](https://github.com/chess-ai-kiro/chess-ai-kiro)
- [完整文档](https://chess-ai-kiro.readthedocs.io/)
- [问题反馈](https://github.com/chess-ai-kiro/chess-ai-kiro/issues)

## 📝 更新日志

### v0.1.0 (2025-01-18)
- ✅ 实现基础屏幕截图功能
- ✅ 添加图形界面区域选择
- ✅ 支持自动和手动截图模式
- ✅ 实现文件管理和存储监控
- ✅ 添加配置管理和统计功能