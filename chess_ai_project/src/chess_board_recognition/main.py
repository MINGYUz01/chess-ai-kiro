#!/usr/bin/env python3
"""
棋盘识别系统主入口文件
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
@click.option('--mode', type=click.Choice(['collect', 'train', 'inference']), 
              default='inference', help='运行模式')
@click.option('--config', type=click.Path(), help='配置文件路径')
@click.option('--data-dir', type=click.Path(), help='数据目录路径')
@click.option('--model-path', type=click.Path(), help='模型文件路径')
@click.option('--image', type=click.Path(exists=True), help='输入图像路径')
@click.option('--region-select', is_flag=True, help='启用区域选择')
def main(mode: str, config: str, data_dir: str, model_path: str, image: str, region_select: bool):
    """棋盘识别系统"""
    console.print(f"[blue]棋盘识别系统 - 模式: {mode}[/blue]")
    
    if mode == 'collect':
        console.print("[green]启动数据收集模式[/green]")
        if region_select:
            console.print("[yellow]区域选择功能尚未实现[/yellow]")
        # TODO: 实现数据收集功能
        
    elif mode == 'train':
        console.print("[green]启动模型训练模式[/green]")
        if data_dir:
            console.print(f"数据目录: {data_dir}")
        # TODO: 实现模型训练功能
        
    elif mode == 'inference':
        console.print("[green]启动推理模式[/green]")
        if image:
            console.print(f"输入图像: {image}")
        if model_path:
            console.print(f"模型路径: {model_path}")
        # TODO: 实现推理功能
    
    console.print("[yellow]功能正在开发中...[/yellow]")


if __name__ == "__main__":
    main()