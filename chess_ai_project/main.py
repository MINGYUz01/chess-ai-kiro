#!/usr/bin/env python3
"""
Chess AI Kiro 主入口文件

提供统一的命令行接口来启动各个子系统。
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chess_ai_project import __version__, __description__

console = Console()


def print_banner():
    """打印项目横幅"""
    banner_text = Text()
    banner_text.append("🏛️ Chess AI Kiro 🏛️\n", style="bold blue")
    banner_text.append(f"版本: {__version__}\n", style="green")
    banner_text.append(__description__, style="white")
    
    panel = Panel(
        banner_text,
        title="中国象棋AI系统",
        title_align="center",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)


@click.group()
@click.version_option(version=__version__, prog_name="Chess AI Kiro")
@click.option('--debug', is_flag=True, help='启用调试模式')
@click.option('--config', type=click.Path(exists=True), help='配置文件路径')
def cli(debug: bool, config: Optional[str]):
    """中国象棋AI系统 - 基于深度学习的棋盘识别、AI引擎和实时分析系统"""
    if debug:
        console.print("[yellow]调试模式已启用[/yellow]")
    
    if config:
        console.print(f"[green]使用配置文件: {config}[/green]")


@cli.command()
@click.option('--mode', type=click.Choice(['collect', 'train', 'inference']), 
              default='inference', help='运行模式')
@click.option('--config', type=click.Path(), help='配置文件路径')
def board_recognition(mode: str, config: Optional[str]):
    """启动棋盘识别系统"""
    console.print(f"[blue]启动棋盘识别系统 - 模式: {mode}[/blue]")
    
    if mode == 'collect':
        console.print("[yellow]数据收集模式尚未实现[/yellow]")
    elif mode == 'train':
        console.print("[yellow]模型训练模式尚未实现[/yellow]")
    elif mode == 'inference':
        console.print("[yellow]推理模式尚未实现[/yellow]")


@cli.command()
@click.option('--mode', type=click.Choice(['server', 'train', 'analyze']), 
              default='server', help='运行模式')
@click.option('--port', type=int, default=8000, help='API服务端口')
@click.option('--config', type=click.Path(), help='配置文件路径')
def ai_engine(mode: str, port: int, config: Optional[str]):
    """启动象棋AI引擎"""
    console.print(f"[blue]启动象棋AI引擎 - 模式: {mode}[/blue]")
    
    if mode == 'server':
        console.print(f"[green]API服务器将在端口 {port} 启动[/green]")
        console.print("[yellow]API服务器模式尚未实现[/yellow]")
    elif mode == 'train':
        console.print("[yellow]训练模式尚未实现[/yellow]")
    elif mode == 'analyze':
        console.print("[yellow]分析模式尚未实现[/yellow]")


@cli.command()
@click.option('--config', type=click.Path(), 
              default='chess_ai_project/configs/default.yaml', 
              help='配置文件路径')
def real_time_analysis(config: str):
    """启动实时分析系统"""
    console.print("[blue]启动实时分析系统[/blue]")
    console.print(f"[green]使用配置文件: {config}[/green]")
    console.print("[yellow]实时分析系统尚未实现[/yellow]")


@cli.command()
def info():
    """显示系统信息"""
    print_banner()
    
    # 显示系统状态
    status_text = Text()
    status_text.append("📊 系统状态\n", style="bold yellow")
    status_text.append("• 棋盘识别系统: ", style="white")
    status_text.append("开发中\n", style="yellow")
    status_text.append("• 象棋AI引擎: ", style="white")
    status_text.append("开发中\n", style="yellow")
    status_text.append("• 实时分析系统: ", style="white")
    status_text.append("开发中\n", style="yellow")
    
    console.print(Panel(status_text, title="系统状态", border_style="yellow"))


@cli.command()
def test():
    """运行测试套件"""
    console.print("[blue]运行测试套件[/blue]")
    
    try:
        import pytest
        console.print("[green]使用pytest运行测试...[/green]")
        # 这里可以添加pytest的调用
        console.print("[yellow]测试功能尚未完全实现[/yellow]")
    except ImportError:
        console.print("[red]pytest未安装，请先安装开发依赖: uv pip install -e \".[dev]\"[/red]")


def main():
    """主入口函数"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]程序被用户中断[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]发生错误: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()