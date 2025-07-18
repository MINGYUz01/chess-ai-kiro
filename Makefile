# Chess AI Kiro Makefile
# 提供常用的开发命令

.PHONY: help install install-dev test lint format clean capture demo-capture run-board run-ai run-analysis

# 默认目标
help:
	@echo "Chess AI Kiro - 中国象棋AI系统"
	@echo ""
	@echo "可用命令:"
	@echo "  install      - 安装项目依赖"
	@echo "  install-dev  - 安装开发依赖"
	@echo "  test         - 运行测试"
	@echo "  lint         - 运行代码检查"
	@echo "  format       - 格式化代码"
	@echo "  clean        - 清理临时文件"
	@echo ""
	@echo "棋盘识别系统:"
	@echo "  capture      - 启动屏幕截图工具"
	@echo "  demo-capture - 运行截图功能演示"
	@echo "  run-board    - 运行棋盘识别系统"
	@echo ""
	@echo "其他模块:"
	@echo "  run-ai       - 运行AI引擎"
	@echo "  run-analysis - 运行实时分析系统"

# 安装依赖
install:
	uv pip install -e .

# 安装开发依赖
install-dev:
	uv pip install -e ".[dev]"
	pre-commit install

# 运行测试
test:
	pytest tests/ -v --cov=chess_ai_project --cov-report=html --cov-report=term

# 代码检查
lint:
	flake8 chess_ai_project/
	mypy chess_ai_project/

# 格式化代码
format:
	black chess_ai_project/
	isort chess_ai_project/

# 清理临时文件
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# 启动屏幕截图工具
capture:
	chess-board-recognition capture

# 运行截图功能演示
demo-capture:
	python -m chess_ai_project.src.chess_board_recognition.data_collection.demo_capture

# 运行棋盘识别系统
run-board:
	chess-board-recognition --mode inference

# 运行AI引擎
run-ai:
	chess-ai-engine --mode server

# 运行实时分析系统
run-analysis:
	real-time-analysis

# 构建文档
docs:
	cd docs && make html

# 运行所有检查
check: lint test

# 发布准备
release: clean format lint test
	@echo "准备发布..."