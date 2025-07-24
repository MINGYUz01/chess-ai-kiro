"""
YOLO11超参数优化和模型评估脚本

该脚本演示了如何使用超参数优化、模型评估和导出功能。
"""

import os
import argparse
import time
from pathlib import Path

from ..system_management.logger import setup_logger
from .trainer import YOLO11Trainer
from .hyperparameter_optimizer import HyperparameterOptimizer
from .evaluator import ModelEvaluator
from .model_exporter import ModelExporter

# 设置日志记录器
logger = setup_logger(__name__)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLO11超参数优化和模型评估脚本')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--data', type=str, required=True, help='数据配置文件路径')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--mode', type=str, choices=['optimize', 'evaluate', 'export', 'all'], default='all', help='运行模式')
    parser.add_argument('--output-dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--search-method', type=str, choices=['grid', 'random', 'bayesian'], default='random', help='超参数搜索方法')
    parser.add_argument('--trials', type=int, default=10, help='超参数搜索试验次数')
    parser.add_argument('--epochs', type=int, default=5, help='每次试验的训练轮次')
    parser.add_argument('--export-formats', type=str, default='onnx', help='导出格式，用逗号分隔')
    
    return parser.parse_args()

def run_hyperparameter_optimization(args):
    """
    运行超参数优化
    """
    logger.info("开始超参数优化")
    
    # 初始化训练器
    trainer = YOLO11Trainer(args.config)
    if args.model:
        trainer.load_model(args.model)
    
    # 初始化超参数优化器
    optimizer = HyperparameterOptimizer(
        trainer=trainer,
        data_yaml_path=args.data,
        output_dir=os.path.join(args.output_dir, 'hyperparameter_search')
    )
    
    # 定义参数空间
    if args.search_method == 'grid':
        # 网格搜索参数空间
        param_grid = {
            'batch': [8, 16, 32],
            'lr0': [0.0005, 0.001, 0.01],
            'weight_decay': [0.0005, 0.001, 0.005],
        }
        
        # 执行网格搜索
        best_params = optimizer.grid_search(param_grid, epochs=args.epochs)
    elif args.search_method == 'bayesian':
        # 贝叶斯优化参数空间
        param_space = {
            'batch': (8, 32),
            'lr0': (0.0001, 0.01),
            'weight_decay': (0.0001, 0.01),
        }
        
        # 执行贝叶斯优化
        best_params = optimizer.bayesian_optimization(param_space, num_trials=args.trials, epochs=args.epochs)
    else:
        # 随机搜索参数空间
        param_space = {
            'batch': (8, 32),
            'lr0': (0.0001, 0.01),
            'weight_decay': (0.0001, 0.01),
            'momentum': (0.8, 0.99),
            'warmup_epochs': [1, 3, 5],
        }
        
        # 执行随机搜索
        best_params = optimizer.random_search(param_space, num_trials=args.trials, epochs=args.epochs)
    
    # 绘制搜索历史
    plot_path = os.path.join(args.output_dir, 'hyperparameter_search', 'search_history.png')
    optimizer.plot_search_history(save_path=plot_path)
    
    # 生成优化报告
    report_path = os.path.join(args.output_dir, 'hyperparameter_search', 'optimization_report.json')
    optimizer.generate_optimization_report(save_path=report_path)
    
    logger.info(f"超参数优化完成，最佳参数: {best_params}")
    logger.info(f"搜索历史图表已保存至: {plot_path}")
    logger.info(f"优化报告已保存至: {report_path}")
    
    return best_params

def run_model_evaluation(args, model_path=None):
    """
    运行模型评估
    """
    logger.info("开始模型评估")
    
    # 使用指定的模型路径或参数中的模型路径
    model_path = model_path or args.model
    if not model_path:
        logger.error("未指定模型路径")
        return
    
    # 初始化模型评估器
    evaluator = ModelEvaluator(
        model_path=model_path,
        output_dir=os.path.join(args.output_dir, 'evaluation')
    )
    
    # 在数据集上评估模型
    results = evaluator.evaluate_on_dataset(args.data)
    
    # 绘制评估结果
    plot_path = os.path.join(args.output_dir, 'evaluation', 'evaluation_results.png')
    evaluator.plot_evaluation_results(save_path=plot_path)
    
    # 生成评估报告
    report_path = os.path.join(args.output_dir, 'evaluation', 'evaluation_report.json')
    evaluator.generate_evaluation_report(save_path=report_path)
    
    logger.info(f"模型评估完成")
    logger.info(f"评估结果图表已保存至: {plot_path}")
    logger.info(f"评估报告已保存至: {report_path}")
    
    return results

def run_model_export(args, model_path=None):
    """
    运行模型导出
    """
    logger.info("开始模型导出")
    
    # 使用指定的模型路径或参数中的模型路径
    model_path = model_path or args.model
    if not model_path:
        logger.error("未指定模型路径")
        return
    
    # 初始化模型导出器
    exporter = ModelExporter(
        model_path=model_path,
        output_dir=os.path.join(args.output_dir, 'exported_models')
    )
    
    # 解析导出格式
    formats = [f.strip() for f in args.export_formats.split(',')]
    
    # 导出模型
    results = exporter.export_to_multiple_formats(formats)
    
    # 比较模型大小
    size_comparison = exporter.compare_model_sizes(results)
    
    logger.info(f"模型导出完成，结果: {results}")
    logger.info(f"模型大小比较: {size_comparison}")
    
    return results

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 根据运行模式执行相应功能
        if args.mode == 'optimize' or args.mode == 'all':
            best_params = run_hyperparameter_optimization(args)
            
            # 如果是all模式，使用最佳参数训练模型
            if args.mode == 'all' and best_params:
                # 初始化训练器
                trainer = YOLO11Trainer(args.config)
                
                # 使用最佳参数训练模型
                logger.info(f"使用最佳参数训练模型: {best_params}")
                best_model_dir = os.path.join(args.output_dir, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)
                
                train_args = {
                    **best_params,
                    'epochs': 100,  # 使用更多轮次进行完整训练
                    'project': best_model_dir,
                    'name': f'best_params_{time.strftime("%Y%m%d_%H%M%S")}',
                    'exist_ok': True,
                }
                
                results = trainer.train(args.data, **train_args)
                
                # 获取最佳模型路径
                best_model_path = os.path.join(best_model_dir, train_args['name'], 'weights', 'best.pt')
                
                # 更新模型路径
                args.model = best_model_path
        
        if args.mode == 'evaluate' or args.mode == 'all':
            run_model_evaluation(args)
        
        if args.mode == 'export' or args.mode == 'all':
            run_model_export(args)
        
        logger.info("脚本执行完成")
        
    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())