"""
YOLO11训练脚本

该脚本演示了如何使用YOLO11训练框架进行模型训练。
"""

import os
import argparse
import time
from pathlib import Path

from ..system_management.logger import setup_logger
from .trainer import YOLO11Trainer
from .config_validator import TrainingConfigValidator, DataConfigGenerator
from .monitor import TrainingMonitor

# 设置日志记录器
logger = setup_logger(__name__)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLO11训练脚本')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--data', type=str, required=True, help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小')
    parser.add_argument('--img-size', type=int, default=None, help='图像尺寸')
    parser.add_argument('--device', type=str, default=None, help='设备 (auto, cpu, cuda, 0, 1, ...)')
    parser.add_argument('--workers', type=int, default=None, help='数据加载线程数')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练检查点路径')
    parser.add_argument('--name', type=str, default=None, help='训练运行名称')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 加载和验证配置
        config_validator = TrainingConfigValidator()
        
        # 如果提供了配置文件，则加载配置
        if args.config and os.path.exists(args.config):
            config = config_validator.load_config(args.config)
            logger.info(f"已加载配置文件: {args.config}")
        else:
            config = config_validator.generate_default_config()
            logger.info("使用默认配置")
        
        # 更新命令行参数
        override_config = {}
        if args.epochs is not None:
            override_config['epochs'] = args.epochs
        if args.batch_size is not None:
            override_config['batch_size'] = args.batch_size
        if args.img_size is not None:
            override_config['image_size'] = args.img_size
        if args.device is not None:
            override_config['device'] = args.device
        if args.workers is not None:
            override_config['workers'] = args.workers
        
        # 合并配置
        if override_config:
            config = config_validator.merge_configs(config, override_config)
            logger.info(f"已合并命令行参数: {override_config}")
        
        # 验证配置
        valid, config, errors = config_validator.validate_config(config)
        if not valid:
            for error in errors:
                logger.warning(f"配置错误: {error}")
            logger.warning("已自动修正配置错误")
        
        # 初始化训练器
        trainer = YOLO11Trainer(args.config)
        
        # 加载模型
        if args.model:
            trainer.load_model(args.model)
        
        # 设置训练监控
        monitor = TrainingMonitor(trainer.metrics_file)
        monitor.add_callback(lambda status: print(f"训练进度: {status['current_epoch']}/{status['total_epochs']} "
                                                 f"({status['progress_percentage']:.1f}%)"))
        
        # 开始训练
        if args.resume:
            logger.info(f"从检查点恢复训练: {args.resume}")
            monitor.start_monitoring(trainer)
            results = trainer.resume_training(args.resume, name=args.name)
        else:
            logger.info(f"开始训练，数据配置: {args.data}")
            monitor.start_monitoring(trainer)
            results = trainer.train(args.data, name=args.name)
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 生成训练报告
        report_path = os.path.join(os.path.dirname(trainer.metrics_file), 'training_report.json')
        report = monitor.generate_training_report(report_path)
        
        # 绘制指标图表
        plot_path = os.path.join(os.path.dirname(trainer.metrics_file), 'metrics_plot.png')
        monitor.plot_metrics(plot_path)
        
        logger.info(f"训练完成，结果: {results}")
        logger.info(f"训练报告已保存至: {report_path}")
        logger.info(f"指标图表已保存至: {plot_path}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())