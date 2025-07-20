@echo off
chcp 65001 >nul
title 棋盘截屏工具 - GUI界面

echo ================================================
echo 棋盘截屏工具 - GUI界面
echo ================================================
echo.

echo 正在启动GUI界面...
echo.

REM 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请确保已安装Python
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 启动GUI
python chess_ai_project\src\chess_board_recognition\data_collection\launch_capture_gui.py

if errorlevel 1 (
    echo.
    echo 启动失败，可能的解决方案:
    echo 1. 安装必要依赖: pip install pillow pyautogui
    echo 2. 确保系统支持GUI界面
    echo 3. 检查配置文件是否正确
    echo.
    pause
)

echo.
echo 程序已退出
pause