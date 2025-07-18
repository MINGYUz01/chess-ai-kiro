# 更新日志

本文档记录了Chess AI Kiro项目的所有重要更新和变更。

## [0.1.1] - 2025-01-18

### ✨ 新增功能
- **屏幕截图系统** - 完整实现棋盘识别系统的数据收集功能
  - 🖼️ 图形界面区域选择器，支持鼠标拖拽选择棋盘区域
  - 📸 手动和自动截图模式，支持定时自动截图
  - 📁 智能文件管理，按日期自动分类存储
  - 💾 存储空间监控，防止磁盘空间不足
  - ⚙️ 完整的配置管理系统
  - 📊 详细的统计信息和运行状态监控

### 🔧 技术实现
- **核心组件**:
  - `ScreenCaptureImpl` - 屏幕截图实现类
  - `RegionSelector` - 区域选择器，支持GUI和对话框输入
  - 多线程自动截图，不阻塞主程序
  - 基于pyautogui的稳定截图引擎

- **文件管理**:
  - 时间戳文件命名：`screenshot_YYYYMMDD_HHMMSS.jpg`
  - 按日期分类存储：`data/captures/YYYYMMDD/`
  - 支持多种图像格式（JPG、PNG等）
  - 可配置图像质量和存储限制

### 🧪 测试覆盖
- 12个单元测试全部通过
- 完整的功能演示脚本
- 集成测试验证
- 错误处理和边界条件测试

### 📚 文档更新
- 更新README.md，添加详细的屏幕截图功能说明
- 新增`docs/screen_capture_guide.md`专门的使用指南
- 更新Makefile和PowerShell脚本，添加新命令
- 更新项目完成状态文档

### 🎮 用户界面
- **命令行工具**:
  ```bash
  # 启动截图工具
  chess-board-recognition capture
  
  # 运行演示
  .\scripts.ps1 demo-capture  # Windows
  make demo-capture          # Linux/macOS
  ```

- **交互式菜单**:
  - 区域选择
  - 手动截图
  - 自动截图
  - 统计信息查看

### 🔄 配置管理
- 支持YAML配置文件
- 区域配置自动保存和加载
- 屏幕分辨率变化检测
- 参数验证和错误处理

---

## [0.1.0] - 2025-01-17

### 🎉 项目初始化
- **环境搭建** - 使用uv包管理器创建Python虚拟环境
- **项目结构** - 建立模块化的代码组织结构
- **核心接口** - 定义系统核心接口和数据结构
- **配置系统** - 实现YAML配置管理和验证
- **日志系统** - 统一的日志记录框架
- **测试框架** - pytest测试环境配置
- **开发工具** - PowerShell脚本和Makefile自动化工具

### 🏗️ 架构设计
- **三大核心模块**:
  - 棋盘识别系统 (chess_board_recognition)
  - 象棋AI引擎 (chinese_chess_ai_engine)  
  - 实时分析系统 (real_time_analysis_system)

- **技术栈选择**:
  - PyTorch + YOLO11 (深度学习)
  - OpenCV + PIL (计算机视觉)
  - FastAPI (Web框架)
  - Tkinter (GUI界面)

### 📦 依赖管理
- 核心依赖：PyTorch, Ultralytics, OpenCV, NumPy等
- 开发依赖：pytest, black, isort, mypy等
- 可选依赖：GPU加速、ONNX导出、TensorRT等

### 📖 文档体系
- 详细的README使用指南
- 完整的项目规格文档
- API接口文档
- 开发环境配置指南

---

## 版本说明

### 版本号格式
采用语义化版本控制 (Semantic Versioning)：`MAJOR.MINOR.PATCH`

- **MAJOR**: 不兼容的API修改
- **MINOR**: 向后兼容的功能性新增
- **PATCH**: 向后兼容的问题修正

### 更新类型标识
- ✨ 新增功能 (Features)
- 🔧 技术改进 (Technical)
- 🐛 问题修复 (Bug Fixes)
- 📚 文档更新 (Documentation)
- 🧪 测试相关 (Testing)
- 🔄 重构代码 (Refactoring)
- ⚡ 性能优化 (Performance)
- 🔒 安全修复 (Security)

---

## 路线图

### 即将发布 (v0.2.0)
- 数据标注支持和labelImg集成
- YOLO11模型训练框架
- 数据增强和质量控制

### 计划中 (v0.3.0)
- 棋局识别和推理引擎
- 结果处理和验证系统
- 性能优化和GPU加速

### 未来版本
- 象棋AI引擎开发
- 实时分析系统集成
- 用户界面和体验优化