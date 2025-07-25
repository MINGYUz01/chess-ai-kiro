"""
搜索性能优化器

实现搜索算法的性能优化和调优功能。
"""

import time
import threading
import psutil
import gc
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
import numpy as np

from .mcts_node import MCTSNode, MCTSConfig
from .time_manager import TimeManager


@dataclass
class SearchMetrics:
    """搜索性能指标"""
    nodes_searched: int = 0
    max_depth: int = 0
    time_used: float = 0.0
    memory_used: float = 0.0
    cache_hit_rate: float = 0.0
    nodes_per_second: float = 0.0
    branching_factor: float = 0.0
    tree_size: int = 0


class SearchOptimizer:
    """
    搜索性能优化器
    
    监控和优化MCTS搜索的性能表现。
    """
    
    def __init__(self):
        """初始化搜索优化器"""
        self.logger = logging.getLogger(__name__)
        
        # 性能监控
        self.metrics_history: List[SearchMetrics] = []
        self.optimization_suggestions: List[str] = []
        
        # 系统资源监控
        self.memory_threshold = 0.8  # 内存使用阈值
        self.cpu_threshold = 0.9     # CPU使用阈值
        
        # 优化参数
        self.auto_gc_enabled = True
        self.memory_cleanup_threshold = 1000  # MB
        
        # 线程安全
        self._lock = threading.Lock()
    
    def analyze_search_performance(
        self,
        root: MCTSNode,
        search_time: float,
        config: MCTSConfig
    ) -> SearchMetrics:
        """
        分析搜索性能
        
        Args:
            root: 搜索根节点
            search_time: 搜索用时
            config: MCTS配置
            
        Returns:
            SearchMetrics: 性能指标
        """
        # 收集基本指标
        tree_info = root.get_tree_info()
        
        metrics = SearchMetrics(
            nodes_searched=tree_info['total_nodes'],
            max_depth=tree_info['max_depth'],
            time_used=search_time,
            nodes_per_second=tree_info['total_nodes'] / search_time if search_time > 0 else 0,
            tree_size=tree_info['total_nodes']
        )
        
        # 计算分支因子
        if root.children:
            total_children = sum(len(child.children) for child in root.children.values())
            metrics.branching_factor = total_children / len(root.children) if root.children else 0
        
        # 获取内存使用
        try:
            process = psutil.Process()
            metrics.memory_used = process.memory_info().rss / 1024 / 1024  # MB
        except:
            metrics.memory_used = 0.0
        
        # 记录历史
        with self._lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-50:]
        
        # 生成优化建议
        self._generate_optimization_suggestions(metrics, config)
        
        return metrics
    
    def _generate_optimization_suggestions(
        self,
        metrics: SearchMetrics,
        config: MCTSConfig
    ):
        """
        生成优化建议
        
        Args:
            metrics: 性能指标
            config: MCTS配置
        """
        suggestions = []
        
        # 搜索速度优化
        if metrics.nodes_per_second < 1000:
            suggestions.append("搜索速度较慢，考虑减少模拟次数或优化神经网络推理")
        
        # 内存使用优化
        if metrics.memory_used > self.memory_cleanup_threshold:
            suggestions.append("内存使用过高，建议启用自动垃圾回收或减少搜索深度")
        
        # 搜索深度优化
        if metrics.max_depth < 5:
            suggestions.append("搜索深度较浅，考虑增加模拟次数或调整UCB参数")
        elif metrics.max_depth > 50:
            suggestions.append("搜索深度过深，可能存在无限循环，检查终止条件")
        
        # 分支因子优化
        if metrics.branching_factor > 30:
            suggestions.append("分支因子过大，考虑增加剪枝或减少合法走法")
        elif metrics.branching_factor < 5:
            suggestions.append("分支因子较小，可能搜索不够充分")
        
        # 时间使用优化
        if len(self.metrics_history) >= 3:
            recent_times = [m.time_used for m in self.metrics_history[-3:]]
            if max(recent_times) / min(recent_times) > 3:
                suggestions.append("搜索时间波动较大，考虑使用自适应时间管理")
        
        # 更新建议
        with self._lock:
            self.optimization_suggestions = suggestions
    
    def optimize_config(self, config: MCTSConfig) -> MCTSConfig:
        """
        基于历史性能优化配置
        
        Args:
            config: 当前配置
            
        Returns:
            MCTSConfig: 优化后的配置
        """
        if len(self.metrics_history) < 5:
            return config
        
        # 分析最近的性能数据
        recent_metrics = self.metrics_history[-10:]
        avg_nps = sum(m.nodes_per_second for m in recent_metrics) / len(recent_metrics)
        avg_depth = sum(m.max_depth for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_used for m in recent_metrics) / len(recent_metrics)
        
        # 创建优化后的配置
        optimized_config = MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            temperature=config.temperature,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            max_depth=config.max_depth,
            time_limit=config.time_limit
        )
        
        # 根据性能调整参数
        if avg_nps < 500:
            # 搜索速度慢，减少模拟次数
            optimized_config.num_simulations = int(config.num_simulations * 0.8)
            self.logger.info("搜索速度慢，减少模拟次数")
        elif avg_nps > 2000:
            # 搜索速度快，可以增加模拟次数
            optimized_config.num_simulations = int(config.num_simulations * 1.2)
            self.logger.info("搜索速度快，增加模拟次数")
        
        if avg_depth < 8:
            # 搜索深度浅，调整UCB参数
            optimized_config.c_puct = min(config.c_puct * 1.1, 2.0)
            self.logger.info("搜索深度浅，增加探索参数")
        elif avg_depth > 25:
            # 搜索深度深，限制最大深度
            optimized_config.max_depth = min(config.max_depth, 30)
            self.logger.info("搜索深度过深，限制最大深度")
        
        if avg_memory > self.memory_cleanup_threshold:
            # 内存使用高，减少模拟次数
            optimized_config.num_simulations = int(config.num_simulations * 0.9)
            optimized_config.max_depth = min(config.max_depth, 20)
            self.logger.info("内存使用高，减少搜索规模")
        
        return optimized_config
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """
        监控系统资源使用
        
        Returns:
            Dict[str, float]: 资源使用情况
        """
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            
            # 进程内存使用
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            resources = {
                'cpu_percent': cpu_percent / 100.0,
                'memory_percent': memory_percent,
                'process_memory_mb': process_memory,
                'available_memory_mb': memory.available / 1024 / 1024
            }
            
            # 检查资源警告
            if cpu_percent / 100.0 > self.cpu_threshold:
                self.logger.warning(f"CPU使用率过高: {cpu_percent:.1f}%")
            
            if memory_percent > self.memory_threshold:
                self.logger.warning(f"内存使用率过高: {memory_percent*100:.1f}%")
            
            return resources
            
        except Exception as e:
            self.logger.error(f"资源监控失败: {e}")
            return {}
    
    def cleanup_memory(self, force: bool = False):
        """
        清理内存
        
        Args:
            force: 是否强制清理
        """
        if not self.auto_gc_enabled and not force:
            return
        
        try:
            # 获取清理前的内存使用
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # 执行垃圾回收
            collected = gc.collect()
            
            # 获取清理后的内存使用
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_freed = memory_before - memory_after
            
            if memory_freed > 0:
                self.logger.info(
                    f"内存清理完成: 释放{memory_freed:.1f}MB, "
                    f"回收{collected}个对象"
                )
            
        except Exception as e:
            self.logger.error(f"内存清理失败: {e}")
    
    def should_cleanup_memory(self) -> bool:
        """
        判断是否需要清理内存
        
        Returns:
            bool: 是否需要清理
        """
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb > self.memory_cleanup_threshold
        except:
            return False
    
    def optimize_search_tree(self, root: MCTSNode) -> int:
        """
        优化搜索树结构
        
        Args:
            root: 根节点
            
        Returns:
            int: 清理的节点数
        """
        cleaned_nodes = 0
        
        def cleanup_node(node: MCTSNode) -> int:
            count = 0
            
            # 清理访问次数很少的子节点
            children_to_remove = []
            for move, child in node.children.items():
                if child.visit_count < 2:  # 访问次数少于2的节点
                    children_to_remove.append(move)
                    count += 1 + cleanup_node(child)
            
            # 移除低访问次数的子节点
            for move in children_to_remove:
                del node.children[move]
            
            return count
        
        try:
            cleaned_nodes = cleanup_node(root)
            if cleaned_nodes > 0:
                self.logger.info(f"搜索树优化完成: 清理{cleaned_nodes}个节点")
        except Exception as e:
            self.logger.error(f"搜索树优化失败: {e}")
        
        return cleaned_nodes
    
    def benchmark_search_performance(
        self,
        searcher,
        test_positions: List,
        simulations_per_position: int = 100
    ) -> Dict[str, float]:
        """
        基准测试搜索性能
        
        Args:
            searcher: 搜索器对象
            test_positions: 测试位置列表
            simulations_per_position: 每个位置的模拟次数
            
        Returns:
            Dict[str, float]: 基准测试结果
        """
        results = {
            'total_positions': len(test_positions),
            'total_time': 0.0,
            'avg_time_per_position': 0.0,
            'avg_nodes_per_second': 0.0,
            'avg_depth': 0.0,
            'memory_peak': 0.0
        }
        
        if not test_positions:
            return results
        
        total_time = 0.0
        total_nodes = 0
        total_depth = 0
        peak_memory = 0.0
        
        self.logger.info(f"开始基准测试: {len(test_positions)}个位置")
        
        for i, position in enumerate(test_positions):
            try:
                # 监控内存
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024
                
                # 执行搜索
                start_time = time.time()
                root = searcher.search(position, simulations_per_position)
                search_time = time.time() - start_time
                
                # 收集指标
                tree_info = root.get_tree_info()
                total_time += search_time
                total_nodes += tree_info['total_nodes']
                total_depth += tree_info['max_depth']
                
                # 监控内存峰值
                memory_after = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, memory_after)
                
                # 定期清理内存
                if (i + 1) % 10 == 0:
                    self.cleanup_memory()
                
                self.logger.info(
                    f"位置 {i+1}/{len(test_positions)}: "
                    f"{search_time:.2f}s, {tree_info['total_nodes']}节点, "
                    f"深度{tree_info['max_depth']}"
                )
                
            except Exception as e:
                self.logger.error(f"位置 {i+1} 测试失败: {e}")
        
        # 计算平均值
        if len(test_positions) > 0:
            results.update({
                'total_time': total_time,
                'avg_time_per_position': total_time / len(test_positions),
                'avg_nodes_per_second': total_nodes / total_time if total_time > 0 else 0,
                'avg_depth': total_depth / len(test_positions),
                'memory_peak': peak_memory
            })
        
        self.logger.info(f"基准测试完成: {results}")
        return results
    
    def get_performance_report(self) -> Dict:
        """
        获取性能报告
        
        Returns:
            Dict: 性能报告
        """
        with self._lock:
            if not self.metrics_history:
                return {'message': '暂无性能数据'}
            
            recent_metrics = self.metrics_history[-20:]  # 最近20次搜索
            
            report = {
                'total_searches': len(self.metrics_history),
                'recent_searches': len(recent_metrics),
                'avg_nodes_per_second': sum(m.nodes_per_second for m in recent_metrics) / len(recent_metrics),
                'avg_search_depth': sum(m.max_depth for m in recent_metrics) / len(recent_metrics),
                'avg_memory_usage': sum(m.memory_used for m in recent_metrics) / len(recent_metrics),
                'avg_search_time': sum(m.time_used for m in recent_metrics) / len(recent_metrics),
                'optimization_suggestions': self.optimization_suggestions.copy(),
                'system_resources': self.monitor_system_resources()
            }
            
            # 性能趋势分析
            if len(self.metrics_history) >= 10:
                old_metrics = self.metrics_history[-20:-10] if len(self.metrics_history) >= 20 else self.metrics_history[:-10]
                new_metrics = self.metrics_history[-10:]
                
                old_avg_nps = sum(m.nodes_per_second for m in old_metrics) / len(old_metrics)
                new_avg_nps = sum(m.nodes_per_second for m in new_metrics) / len(new_metrics)
                
                if new_avg_nps > old_avg_nps * 1.1:
                    report['performance_trend'] = 'improving'
                elif new_avg_nps < old_avg_nps * 0.9:
                    report['performance_trend'] = 'declining'
                else:
                    report['performance_trend'] = 'stable'
            
            return report
    
    def reset_metrics(self):
        """重置性能指标"""
        with self._lock:
            self.metrics_history.clear()
            self.optimization_suggestions.clear()
        
        self.logger.info("性能指标已重置")
    
    def set_optimization_parameters(
        self,
        memory_threshold: float = 0.8,
        cpu_threshold: float = 0.9,
        auto_gc_enabled: bool = True,
        memory_cleanup_threshold: float = 1000
    ):
        """
        设置优化参数
        
        Args:
            memory_threshold: 内存使用阈值
            cpu_threshold: CPU使用阈值
            auto_gc_enabled: 是否启用自动垃圾回收
            memory_cleanup_threshold: 内存清理阈值(MB)
        """
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.auto_gc_enabled = auto_gc_enabled
        self.memory_cleanup_threshold = memory_cleanup_threshold
        
        self.logger.info(f"优化参数已更新: 内存阈值{memory_threshold}, CPU阈值{cpu_threshold}")


class SearchProfiler:
    """
    搜索性能分析器
    
    详细分析搜索过程的性能瓶颈。
    """
    
    def __init__(self):
        """初始化性能分析器"""
        self.logger = logging.getLogger(__name__)
        self.profiling_data: Dict[str, List[float]] = {}
        self.is_profiling = False
        self._lock = threading.Lock()
    
    def start_profiling(self):
        """开始性能分析"""
        with self._lock:
            self.is_profiling = True
            self.profiling_data.clear()
        self.logger.info("开始性能分析")
    
    def stop_profiling(self) -> Dict[str, Dict[str, float]]:
        """
        停止性能分析
        
        Returns:
            Dict[str, Dict[str, float]]: 分析结果
        """
        with self._lock:
            self.is_profiling = False
            
            results = {}
            for operation, times in self.profiling_data.items():
                if times:
                    results[operation] = {
                        'total_time': sum(times),
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'call_count': len(times)
                    }
            
            self.logger.info(f"性能分析完成: {len(results)}个操作")
            return results
    
    def record_operation(self, operation_name: str, execution_time: float):
        """
        记录操作时间
        
        Args:
            operation_name: 操作名称
            execution_time: 执行时间
        """
        if not self.is_profiling:
            return
        
        with self._lock:
            if operation_name not in self.profiling_data:
                self.profiling_data[operation_name] = []
            self.profiling_data[operation_name].append(execution_time)
    
    def profile_operation(self, operation_name: str):
        """
        操作性能分析装饰器
        
        Args:
            operation_name: 操作名称
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if self.is_profiling:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    self.record_operation(operation_name, end_time - start_time)
                    return result
                else:
                    return func(*args, **kwargs)
            return wrapper
        return decorator