# Chess AI Kiro PowerShell 脚本
# Windows用户的Makefile替代方案

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "Chess AI Kiro - 中国象棋AI系统" -ForegroundColor Blue
    Write-Host ""
    Write-Host "可用命令:" -ForegroundColor Green
    Write-Host "  help         - 显示此帮助信息"
    Write-Host "  install      - 安装项目依赖"
    Write-Host "  install-dev  - 安装开发依赖"
    Write-Host "  test         - 运行测试"
    Write-Host "  lint         - 运行代码检查"
    Write-Host "  format       - 格式化代码"
    Write-Host "  clean        - 清理临时文件"
    Write-Host ""
    Write-Host "棋盘识别系统:" -ForegroundColor Cyan
    Write-Host "  capture      - 启动屏幕截图工具"
    Write-Host "  demo-capture - 运行截图功能演示"
    Write-Host "  run-board    - 运行棋盘识别系统"
    Write-Host ""
    Write-Host "其他模块:" -ForegroundColor Cyan
    Write-Host "  run-ai       - 运行AI引擎"
    Write-Host "  run-analysis - 运行实时分析系统"
    Write-Host ""
    Write-Host "工具命令:" -ForegroundColor Cyan
    Write-Host "  check        - 运行所有检查"
    Write-Host "  release      - 发布准备"
    Write-Host ""
    Write-Host "使用方法: .\scripts.ps1 <命令名>" -ForegroundColor Yellow
    Write-Host "例如: .\scripts.ps1 install" -ForegroundColor Yellow
}

function Install-Dependencies {
    Write-Host "安装项目依赖..." -ForegroundColor Green
    uv pip install -e .
}

function Install-DevDependencies {
    Write-Host "安装开发依赖..." -ForegroundColor Green
    uv pip install -e ".[dev]"
    if (Get-Command pre-commit -ErrorAction SilentlyContinue) {
        pre-commit install
    } else {
        Write-Host "pre-commit 未安装，跳过钩子安装" -ForegroundColor Yellow
    }
}

function Run-Tests {
    Write-Host "运行测试..." -ForegroundColor Green
    if (Get-Command pytest -ErrorAction SilentlyContinue) {
        pytest tests/ -v --cov=chess_ai_project --cov-report=html --cov-report=term
    } else {
        Write-Host "pytest 未安装，请先运行: .\scripts.ps1 install-dev" -ForegroundColor Red
    }
}

function Run-Lint {
    Write-Host "运行代码检查..." -ForegroundColor Green
    if (Get-Command flake8 -ErrorAction SilentlyContinue) {
        flake8 chess_ai_project/
    } else {
        Write-Host "flake8 未安装" -ForegroundColor Yellow
    }
    
    if (Get-Command mypy -ErrorAction SilentlyContinue) {
        mypy chess_ai_project/
    } else {
        Write-Host "mypy 未安装" -ForegroundColor Yellow
    }
}

function Format-Code {
    Write-Host "格式化代码..." -ForegroundColor Green
    if (Get-Command black -ErrorAction SilentlyContinue) {
        black chess_ai_project/
    } else {
        Write-Host "black 未安装" -ForegroundColor Yellow
    }
    
    if (Get-Command isort -ErrorAction SilentlyContinue) {
        isort chess_ai_project/
    } else {
        Write-Host "isort 未安装" -ForegroundColor Yellow
    }
}

function Clean-Files {
    Write-Host "清理临时文件..." -ForegroundColor Green
    
    # 删除 Python 缓存文件
    Get-ChildItem -Path . -Recurse -Name "*.pyc" | Remove-Item -Force
    Get-ChildItem -Path . -Recurse -Name "__pycache__" -Directory | Remove-Item -Recurse -Force
    
    # 删除构建文件
    if (Test-Path "build") { Remove-Item -Path "build" -Recurse -Force }
    if (Test-Path "dist") { Remove-Item -Path "dist" -Recurse -Force }
    if (Test-Path "htmlcov") { Remove-Item -Path "htmlcov" -Recurse -Force }
    if (Test-Path ".coverage") { Remove-Item -Path ".coverage" -Force }
    if (Test-Path ".pytest_cache") { Remove-Item -Path ".pytest_cache" -Recurse -Force }
    if (Test-Path ".mypy_cache") { Remove-Item -Path ".mypy_cache" -Recurse -Force }
    
    # 删除 egg-info 目录
    Get-ChildItem -Path . -Name "*.egg-info" -Directory | Remove-Item -Recurse -Force
    
    Write-Host "清理完成!" -ForegroundColor Green
}

function Start-Capture {
    Write-Host "启动屏幕截图工具..." -ForegroundColor Green
    chess-board-recognition capture
}

function Run-DemoCapture {
    Write-Host "运行截图功能演示..." -ForegroundColor Green
    python -m chess_ai_project.src.chess_board_recognition.data_collection.demo_capture
}

function Run-BoardRecognition {
    Write-Host "启动棋盘识别系统..." -ForegroundColor Green
    chess-board-recognition --mode inference
}

function Run-AIEngine {
    Write-Host "启动AI引擎..." -ForegroundColor Green
    chess-ai-engine --mode server
}

function Run-RealTimeAnalysis {
    Write-Host "启动实时分析系统..." -ForegroundColor Green
    real-time-analysis
}

function Run-Check {
    Write-Host "运行所有检查..." -ForegroundColor Green
    Run-Lint
    Run-Tests
}

function Prepare-Release {
    Write-Host "准备发布..." -ForegroundColor Green
    Clean-Files
    Format-Code
    Run-Lint
    Run-Tests
    Write-Host "发布准备完成!" -ForegroundColor Green
}

# 主逻辑
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "install" { Install-Dependencies }
    "install-dev" { Install-DevDependencies }
    "test" { Run-Tests }
    "lint" { Run-Lint }
    "format" { Format-Code }
    "clean" { Clean-Files }
    "capture" { Start-Capture }
    "demo-capture" { Run-DemoCapture }
    "run-board" { Run-BoardRecognition }
    "run-ai" { Run-AIEngine }
    "run-analysis" { Run-RealTimeAnalysis }
    "check" { Run-Check }
    "release" { Prepare-Release }
    default { 
        Write-Host "未知命令: $Command" -ForegroundColor Red
        Write-Host "使用 '.\scripts.ps1 help' 查看可用命令" -ForegroundColor Yellow
    }
}