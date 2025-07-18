#!/usr/bin/env python3
"""
实时分析系统主入口文件
"""

import sys
from pathlib import Path
import click
from rich.console import Console

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

console = Console()


@click.command()
@click.option('--config', type=click.Path(), 
              default='chess_ai_project/configs/default.yaml', 
              help='配置文件路径')
@click.option('--region', type=str, help='监控区域 (x,y,width,height)')
@click.option('--interval', type=int, default=2, help='监控间隔（秒）')
@click.option('--no-overlay', is_flag=True, help='禁用界面叠加')
def main(config: str, region: str, interval: int, no_overlay: bool):
    """实时分析系统"""
    console.print("[blue]实时分析系统[/blue]")
    console.print(f"配置文件: {config}")
    console.print(f"监控间隔: {interval}秒")
    
    if region:
        console.print(f"监控区域: {region}")
    
    if no_overlay:
        console.print("[yellow]界面叠加已禁用[/yellow]")
    
    console.print("[green]启动实时分析系统...[/green]")
    # TODO: 实现实时分析功能
    
    console.print("[yellow]功能正在开发中...[/yellow]")


if __name__ == "__main__":
    main()