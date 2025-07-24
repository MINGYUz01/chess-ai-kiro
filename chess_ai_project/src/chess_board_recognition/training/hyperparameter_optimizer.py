"""
超参数优化模块

该模块提供了用于优化YOLO11模型超参数的类和函数。
"""

import os
import json
import yaml
import time
import random
import logging
import itertools
from typing import Dict, Any, List, Tuple, Optional, Callable, Union

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .trainer import YOLO11Trainer
from ..system_management.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

class HyperparameterOptimizer:
    """
    超参数优化器类
    
    该类用于搜索和优化YOLO11模型的超参数。
    """
    
    def __init__(self, trainer: YOLO11Trainer, data_yaml_path: str, output_dir: str = './hyperparameter_search'):
        """
        初始化超参数优化器
        
        参数:
            trainer: YOLO11Trainer实例
            data_yaml_path: 数据配置文件路径
            output_dir: 输出目录
        """
        self.trainer = trainer
        self.data_yaml_path = data_yaml_path
        self.output_dir = output_dir
        self.search_history = []
        self.best_params = None
        self.best_metric = float('inf')  # 对于损失等指标，越小越好
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"超参数优化器初始化完成，输出目录: {output_dir}")
    
    def _evaluate_params(self, params: Dict[str, Any], epochs: int = 10) -> float:
        """
        评估超参数组合
        
        参数:
            params: 超参数字典
            epochs: 训练轮次
            
        返回:
            评估指标（越小越好）
        """
        try:
            # 设置训练参数
            train_args = {
                'data': self.data_yaml_path,
                'epochs': epochs,
                **params,
                'project': self.output_dir,
                'name': f'search_{time.strftime("%Y%m%d_%H%M%S")}',
                'exist_ok': True,
            }
            
            logger.info(f"评估超参数: {params}")
            
            # 执行训练
            results = self.trainer.train(**train_args)
            
            # 获取评估指标
            if hasattr(results, 'results_dict') and results.results_dict:
                # 使用验证集上的最佳指标
                metric = results.results_dict.get('metrics/mAP50-95(B)', float('inf'))
                if metric == float('inf'):
                    # 如果没有mAP50-95，则使用损失
                    metric = results.results_dict.get('metrics/loss(B)', float('inf'))
            else:
                # 如果没有结果字典，则使用训练器的最佳指标
                metric = self.trainer.best_metric
            
            # 记录结果
            result = {
                'params': params,
                'metric': metric,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.search_history.append(result)
            
            # 更新最佳参数
            if metric < self.best_metric:
                self.best_metric = metric
                self.best_params = params.copy()
                logger.info(f"发现新的最佳参数: {params}, 指标: {metric}")
            
            # 保存搜索历史
            self._save_search_history()
            
            return metric
        except Exception as e:
            logger.error(f"评估超参数失败: {e}")
            return float('inf')
    
    def _save_search_history(self) -> None:
        """
        保存搜索历史
        """
        try:
            history_path = os.path.join(self.output_dir, 'search_history.json')
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.search_history, f, ensure_ascii=False, indent=2)
            
            # 保存最佳参数
            if self.best_params:
                best_params_path = os.path.join(self.output_dir, 'best_params.yaml')
                with open(best_params_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.best_params, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"保存搜索历史失败: {e}")
    
    def grid_search(self, param_grid: Dict[str, List[Any]], epochs: int = 10) -> Dict[str, Any]:
        """
        网格搜索超参数
        
        参数:
            param_grid: 参数网格，格式为 {参数名: [参数值列表]}
            epochs: 每次评估的训练轮次
            
        返回:
            最佳参数字典
        """
        try:
            # 生成所有参数组合
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(itertools.product(*param_values))
            
            logger.info(f"开始网格搜索，参数组合数: {len(param_combinations)}")
            
            # 评估每个参数组合
            for i, values in enumerate(tqdm(param_combinations, desc="网格搜索进度")):
                params = {name: value for name, value in zip(param_names, values)}
                
                # 评估参数
                metric = self._evaluate_params(params, epochs)
                
                logger.info(f"参数组合 {i+1}/{len(param_combinations)}: {params}, 指标: {metric}")
            
            logger.info(f"网格搜索完成，最佳参数: {self.best_params}, 最佳指标: {self.best_metric}")
            return self.best_params
        except Exception as e:
            logger.error(f"网格搜索失败: {e}")
            return {}
    
    def random_search(self, param_space: Dict[str, Tuple], num_trials: int = 10, epochs: int = 10) -> Dict[str, Any]:
        """
        随机搜索超参数
        
        参数:
            param_space: 参数空间，格式为 {参数名: (最小值, 最大值)} 或 {参数名: [可选值列表]}
            num_trials: 随机试验次数
            epochs: 每次评估的训练轮次
            
        返回:
            最佳参数字典
        """
        try:
            logger.info(f"开始随机搜索，试验次数: {num_trials}")
            
            # 执行随机搜索
            for i in tqdm(range(num_trials), desc="随机搜索进度"):
                # 随机生成参数
                params = {}
                for name, space in param_space.items():
                    if isinstance(space, (list, tuple)) and len(space) == 2 and all(isinstance(x, (int, float)) for x in space):
                        # 连续参数空间
                        min_val, max_val = space
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            # 整数参数
                            params[name] = random.randint(min_val, max_val)
                        else:
                            # 浮点数参数
                            params[name] = random.uniform(min_val, max_val)
                    elif isinstance(space, (list, tuple)):
                        # 离散参数空间
                        params[name] = random.choice(space)
                    else:
                        logger.warning(f"无效的参数空间: {name}: {space}")
                
                # 评估参数
                metric = self._evaluate_params(params, epochs)
                
                logger.info(f"随机试验 {i+1}/{num_trials}: {params}, 指标: {metric}")
            
            logger.info(f"随机搜索完成，最佳参数: {self.best_params}, 最佳指标: {self.best_metric}")
            return self.best_params
        except Exception as e:
            logger.error(f"随机搜索失败: {e}")
            return {}
    
    def bayesian_optimization(self, param_space: Dict[str, Tuple], num_trials: int = 10, epochs: int = 10) -> Dict[str, Any]:
        """
        贝叶斯优化超参数
        
        参数:
            param_space: 参数空间，格式为 {参数名: (最小值, 最大值)} 或 {参数名: [可选值列表]}
            num_trials: 优化试验次数
            epochs: 每次评估的训练轮次
            
        返回:
            最佳参数字典
        """
        try:
            # 尝试导入贝叶斯优化库
            try:
                from skopt import gp_minimize
                from skopt.space import Real, Integer, Categorical
                from skopt.utils import use_named_args
            except ImportError:
                logger.error("贝叶斯优化需要scikit-optimize库，请安装: pip install scikit-optimize")
                return self.random_search(param_space, num_trials, epochs)
            
            logger.info(f"开始贝叶斯优化，试验次数: {num_trials}")
            
            # 构建搜索空间
            dimensions = []
            dimension_names = []
            
            for name, space in param_space.items():
                dimension_names.append(name)
                
                if isinstance(space, (list, tuple)) and len(space) == 2 and all(isinstance(x, (int, float)) for x in space):
                    # 连续参数空间
                    min_val, max_val = space
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # 整数参数
                        dimensions.append(Integer(min_val, max_val, name=name))
                    else:
                        # 浮点数参数
                        dimensions.append(Real(min_val, max_val, name=name))
                elif isinstance(space, (list, tuple)):
                    # 离散参数空间
                    dimensions.append(Categorical(space, name=name))
                else:
                    logger.warning(f"无效的参数空间: {name}: {space}")
            
            # 定义目标函数
            @use_named_args(dimensions=dimensions)
            def objective(**params):
                return self._evaluate_params(params, epochs)
            
            # 执行贝叶斯优化
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=num_trials,
                random_state=42,
                verbose=True,
                n_jobs=-1
            )
            
            # 提取最佳参数
            best_params = {name: value for name, value in zip(dimension_names, result.x)}
            self.best_params = best_params
            self.best_metric = result.fun
            
            logger.info(f"贝叶斯优化完成，最佳参数: {self.best_params}, 最佳指标: {self.best_metric}")
            return self.best_params
        except Exception as e:
            logger.error(f"贝叶斯优化失败: {e}")
            return {}
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        获取最佳参数
        
        返回:
            最佳参数字典
        """
        return self.best_params or {}
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """
        获取搜索历史
        
        返回:
            搜索历史列表
        """
        return self.search_history
    
    def plot_search_history(self, save_path: Optional[str] = None) -> None:
        """
        绘制搜索历史
        
        参数:
            save_path: 图表保存路径，如果为None则显示图表
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.search_history:
                logger.warning("没有搜索历史可供绘制")
                return
            
            # 提取指标
            metrics = [result['metric'] for result in self.search_history]
            trials = list(range(1, len(metrics) + 1))
            
            # 创建图表
            fig = plt.figure(figsize=(10, 6))
            plt.plot(trials, metrics, 'b-', marker='o')
            plt.axhline(y=min(metrics), color='r', linestyle='--', label=f'最佳指标: {min(metrics):.4f}')
            
            plt.title('超参数搜索历史')
            plt.xlabel('试验次数')
            plt.ylabel('评估指标')
            plt.grid(True)
            plt.legend()
            
            # 保存或显示图表
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"搜索历史图表已保存至: {save_path}")
                plt.close(fig)
            else:
                plt.show()
                
        except ImportError:
            logger.error("绘制搜索历史需要matplotlib库")
        except Exception as e:
            logger.error(f"绘制搜索历史失败: {e}")
    
    def generate_optimization_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成优化报告
        
        参数:
            save_path: 报告保存路径，如果为None则不保存
            
        返回:
            优化报告字典
        """
        try:
            if not self.search_history:
                logger.warning("没有搜索历史可供生成报告")
                return {"error": "没有搜索历史"}
            
            # 计算统计信息
            metrics = [result['metric'] for result in self.search_history]
            
            # 确保每个搜索历史项都有时间戳
            for item in self.search_history:
                if 'timestamp' not in item:
                    item['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            report = {
                "搜索试验次数": len(self.search_history),
                "最佳指标": min(metrics),
                "最差指标": max(metrics),
                "平均指标": sum(metrics) / len(metrics),
                "标准差": np.std(metrics) if len(metrics) > 1 else 0,
                "最佳参数": self.best_params,
                "开始时间": self.search_history[0]['timestamp'],
                "结束时间": self.search_history[-1]['timestamp'],
            }
            
            # 保存报告
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                logger.info(f"优化报告已保存至: {save_path}")
            
            return report
        except Exception as e:
            logger.error(f"生成优化报告失败: {e}")
            return {"error": str(e)}