"""
并行搜索器

实现多线程并行MCTS搜索以提高搜索效率。
"""

import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Dict, List, Optional, Tuple, Callable
from queue import Queue, Empty
import copy

from .mcts_node import MCTSNode, MCTSConfig
from .mcts_searcher import MCTSSearcher
from .time_manager import TimeManager, TimeAllocation, AdaptiveTimeManager
from .search_optimizer import SearchOptimizer, SearchMetrics
from ..rules_engine import ChessBoard, Move
from ..neural_network import ChessNet, InferenceEngine


class ParallelSearcher:
    """
    并行搜索器
    
    使用多线程并行执行MCTS搜索，提高搜索效率。
    """
    
    def __init__(
        self,
        model: ChessNet,
        num_workers: int = 4,
        config: Optional[MCTSConfig] = None,
        shared_inference_engine: bool = True,
        enable_time_management: bool = True,
        enable_optimization: bool = True
    ):
        """
        初始化并行搜索器
        
        Args:
            model: 神经网络模型
            num_workers: 工作线程数
            config: MCTS配置
            shared_inference_engine: 是否共享推理引擎
        """
        self.model = model
        self.num_workers = num_workers
        self.config = config or MCTSConfig()
        
        # 创建推理引擎
        if shared_inference_engine:
            # 共享推理引擎（支持批处理）
            self.shared_inference_engine = InferenceEngine(
                model=model,
                device='auto',
                batch_size=max(4, num_workers)
            )
            self.searchers = [
                MCTSSearcher(model, config, self.shared_inference_engine)
                for _ in range(num_workers)
            ]
        else:
            # 每个工作线程独立的推理引擎
            self.shared_inference_engine = None
            self.searchers = [
                MCTSSearcher(model, config)
                for _ in range(num_workers)
            ]
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 时间管理
        self.enable_time_management = enable_time_management
        if enable_time_management:
            self.time_manager = AdaptiveTimeManager()
        else:
            self.time_manager = None
        
        # 性能优化
        self.enable_optimization = enable_optimization
        if enable_optimization:
            self.optimizer = SearchOptimizer()
        else:
            self.optimizer = None
        
        # 统计信息
        self.stats = {
            'total_searches': 0,
            'total_parallel_time': 0.0,
            'total_simulations': 0,
            'worker_stats': [{'searches': 0, 'simulations': 0} for _ in range(num_workers)]
        }
        
        # 同步锁
        self.stats_lock = threading.Lock()
    
    def parallel_search(
        self,
        board: ChessBoard,
        total_simulations: int,
        time_limit: Optional[float] = None,
        merge_method: str = 'weighted_average'
    ) -> Tuple[Move, float, Dict[Move, float]]:
        """
        并行搜索
        
        Args:
            board: 棋盘状态
            total_simulations: 总模拟次数
            time_limit: 时间限制
            merge_method: 结果合并方法
            
        Returns:
            Tuple[Move, float, Dict[Move, float]]: (最佳走法, 评估值, 动作概率)
        """
        start_time = time.time()
        
        # 分配模拟次数给各个工作线程
        sims_per_worker = total_simulations // self.num_workers
        remaining_sims = total_simulations % self.num_workers
        
        simulation_counts = [sims_per_worker] * self.num_workers
        for i in range(remaining_sims):
            simulation_counts[i] += 1
        
        # 提交搜索任务
        futures = []
        for i, (searcher, num_sims) in enumerate(zip(self.searchers, simulation_counts)):
            if num_sims > 0:
                future = self.executor.submit(
                    self._worker_search,
                    searcher,
                    board,
                    num_sims,
                    time_limit,
                    i
                )
                futures.append((i, future))
        
        # 收集结果
        search_results = []
        completed_workers = 0
        
        for worker_id, future in futures:
            try:
                result = future.result(timeout=time_limit)
                search_results.append((worker_id, result))
                completed_workers += 1
            except Exception as e:
                self.logger.warning(f"工作线程{worker_id}搜索失败: {e}")
        
        if not search_results:
            raise RuntimeError("所有工作线程都失败了")
        
        # 合并结果
        merged_result = self._merge_search_results(search_results, merge_method)
        
        # 更新统计信息
        search_time = time.time() - start_time
        with self.stats_lock:
            self.stats['total_searches'] += 1
            self.stats['total_parallel_time'] += search_time
            self.stats['total_simulations'] += sum(simulation_counts[:completed_workers])
        
        self.logger.info(
            f"并行搜索完成: {completed_workers}/{self.num_workers}个工作线程, "
            f"总模拟{sum(simulation_counts[:completed_workers])}次, "
            f"耗时{search_time:.3f}s"
        )
        
        return merged_result
    
    def _worker_search(
        self,
        searcher: MCTSSearcher,
        board: ChessBoard,
        num_simulations: int,
        time_limit: Optional[float],
        worker_id: int
    ) -> Dict:
        """
        工作线程搜索函数
        
        Args:
            searcher: MCTS搜索器
            board: 棋盘状态
            num_simulations: 模拟次数
            time_limit: 时间限制
            worker_id: 工作线程ID
            
        Returns:
            Dict: 搜索结果
        """
        try:
            # 创建棋盘副本以避免线程冲突
            board_copy = copy.deepcopy(board)
            
            # 执行搜索
            root = searcher.search(board_copy, num_simulations, time_limit)
            
            # 收集结果
            result = {
                'root': root,
                'best_move': searcher.get_best_move(root),
                'evaluation': root.average_value,
                'visit_count': root.visit_count,
                'action_probabilities': searcher.get_action_probabilities(root),
                'principal_variation': searcher.get_principal_variation(root),
                'worker_id': worker_id
            }
            
            # 更新工作线程统计
            with self.stats_lock:
                self.stats['worker_stats'][worker_id]['searches'] += 1
                self.stats['worker_stats'][worker_id]['simulations'] += root.visit_count
            
            return result
            
        except Exception as e:
            self.logger.error(f"工作线程{worker_id}搜索异常: {e}")
            raise
    
    def _merge_search_results(
        self,
        results: List[Tuple[int, Dict]],
        merge_method: str
    ) -> Tuple[Move, float, Dict[Move, float]]:
        """
        合并搜索结果
        
        Args:
            results: 搜索结果列表
            merge_method: 合并方法
            
        Returns:
            Tuple[Move, float, Dict[Move, float]]: 合并后的结果
        """
        if not results:
            raise ValueError("没有搜索结果可合并")
        
        if len(results) == 1:
            # 只有一个结果，直接返回
            _, result = results[0]
            return (
                result['best_move'],
                result['evaluation'],
                result['action_probabilities']
            )
        
        if merge_method == 'weighted_average':
            return self._merge_weighted_average(results)
        elif merge_method == 'majority_vote':
            return self._merge_majority_vote(results)
        elif merge_method == 'best_worker':
            return self._merge_best_worker(results)
        else:
            raise ValueError(f"未知的合并方法: {merge_method}")
    
    def _merge_weighted_average(
        self,
        results: List[Tuple[int, Dict]]
    ) -> Tuple[Move, float, Dict[Move, float]]:
        """
        加权平均合并
        
        Args:
            results: 搜索结果列表
            
        Returns:
            Tuple[Move, float, Dict[Move, float]]: 合并后的结果
        """
        # 计算权重（基于访问次数）
        total_visits = sum(result['visit_count'] for _, result in results)
        if total_visits == 0:
            # 如果没有访问，使用均匀权重
            weights = [1.0 / len(results)] * len(results)
        else:
            weights = [result['visit_count'] / total_visits for _, result in results]
        
        # 加权平均评估值
        weighted_evaluation = sum(
            weight * result['evaluation']
            for weight, (_, result) in zip(weights, results)
        )
        
        # 合并动作概率
        all_moves = set()
        for _, result in results:
            all_moves.update(result['action_probabilities'].keys())
        
        merged_probabilities = {}
        for move in all_moves:
            weighted_prob = sum(
                weight * result['action_probabilities'].get(move, 0.0)
                for weight, (_, result) in zip(weights, results)
            )
            merged_probabilities[move] = weighted_prob
        
        # 选择概率最高的走法作为最佳走法
        if merged_probabilities:
            best_move = max(merged_probabilities.keys(), key=lambda m: merged_probabilities[m])
        else:
            best_move = results[0][1]['best_move']
        
        return best_move, weighted_evaluation, merged_probabilities
    
    def _merge_majority_vote(
        self,
        results: List[Tuple[int, Dict]]
    ) -> Tuple[Move, float, Dict[Move, float]]:
        """
        多数投票合并
        
        Args:
            results: 搜索结果列表
            
        Returns:
            Tuple[Move, float, Dict[Move, float]]: 合并后的结果
        """
        # 统计最佳走法的投票
        move_votes = {}
        for _, result in results:
            best_move = result['best_move']
            if best_move:
                move_votes[best_move] = move_votes.get(best_move, 0) + 1
        
        if not move_votes:
            # 没有有效走法，返回第一个结果
            _, first_result = results[0]
            return (
                first_result['best_move'],
                first_result['evaluation'],
                first_result['action_probabilities']
            )
        
        # 选择得票最多的走法
        best_move = max(move_votes.keys(), key=lambda m: move_votes[m])
        
        # 计算支持该走法的结果的平均评估值
        supporting_results = [
            result for _, result in results
            if result['best_move'] == best_move
        ]
        
        avg_evaluation = sum(
            result['evaluation'] for result in supporting_results
        ) / len(supporting_results)
        
        # 合并动作概率（简单平均）
        all_moves = set()
        for _, result in results:
            all_moves.update(result['action_probabilities'].keys())
        
        merged_probabilities = {}
        for move in all_moves:
            avg_prob = sum(
                result['action_probabilities'].get(move, 0.0)
                for _, result in results
            ) / len(results)
            merged_probabilities[move] = avg_prob
        
        return best_move, avg_evaluation, merged_probabilities
    
    def _merge_best_worker(
        self,
        results: List[Tuple[int, Dict]]
    ) -> Tuple[Move, float, Dict[Move, float]]:
        """
        选择最佳工作线程的结果
        
        Args:
            results: 搜索结果列表
            
        Returns:
            Tuple[Move, float, Dict[Move, float]]: 最佳结果
        """
        # 选择访问次数最多的结果
        best_worker_id, best_result = max(
            results,
            key=lambda x: x[1]['visit_count']
        )
        
        return (
            best_result['best_move'],
            best_result['evaluation'],
            best_result['action_probabilities']
        )
    
    def batch_analyze(
        self,
        boards: List[ChessBoard],
        simulations_per_board: int,
        time_limit_per_board: Optional[float] = None
    ) -> List[Dict]:
        """
        批量分析多个棋局
        
        Args:
            boards: 棋盘列表
            simulations_per_board: 每个棋盘的模拟次数
            time_limit_per_board: 每个棋盘的时间限制
            
        Returns:
            List[Dict]: 分析结果列表
        """
        results = []
        
        # 提交所有分析任务
        futures = []
        for i, board in enumerate(boards):
            future = self.executor.submit(
                self._analyze_single_board,
                board,
                simulations_per_board,
                time_limit_per_board,
                i
            )
            futures.append((i, future))
        
        # 收集结果
        for board_id, future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"分析棋盘{board_id}失败: {e}")
                results.append({
                    'board_id': board_id,
                    'error': str(e),
                    'best_move': None,
                    'evaluation': 0.0
                })
        
        return results
    
    def _analyze_single_board(
        self,
        board: ChessBoard,
        num_simulations: int,
        time_limit: Optional[float],
        board_id: int
    ) -> Dict:
        """
        分析单个棋盘
        
        Args:
            board: 棋盘状态
            num_simulations: 模拟次数
            time_limit: 时间限制
            board_id: 棋盘ID
            
        Returns:
            Dict: 分析结果
        """
        try:
            best_move, evaluation, action_probs = self.parallel_search(
                board,
                num_simulations,
                time_limit
            )
            
            return {
                'board_id': board_id,
                'best_move': best_move,
                'evaluation': evaluation,
                'action_probabilities': action_probs,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"分析棋盘{board_id}异常: {e}")
            return {
                'board_id': board_id,
                'error': str(e),
                'success': False
            }
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self.stats_lock:
            stats = copy.deepcopy(self.stats)
        
        # 计算平均值
        if stats['total_searches'] > 0:
            stats['avg_time_per_search'] = stats['total_parallel_time'] / stats['total_searches']
            stats['avg_simulations_per_search'] = stats['total_simulations'] / stats['total_searches']
        else:
            stats['avg_time_per_search'] = 0.0
            stats['avg_simulations_per_search'] = 0.0
        
        # 添加工作线程效率
        stats['worker_efficiency'] = []
        for i, worker_stat in enumerate(stats['worker_stats']):
            efficiency = {
                'worker_id': i,
                'searches': worker_stat['searches'],
                'simulations': worker_stat['simulations'],
                'avg_sims_per_search': (
                    worker_stat['simulations'] / worker_stat['searches']
                    if worker_stat['searches'] > 0 else 0
                )
            }
            stats['worker_efficiency'].append(efficiency)
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        with self.stats_lock:
            self.stats = {
                'total_searches': 0,
                'total_parallel_time': 0.0,
                'total_simulations': 0,
                'worker_stats': [{'searches': 0, 'simulations': 0} for _ in range(self.num_workers)]
            }
        
        # 重置各个搜索器的统计
        for searcher in self.searchers:
            searcher.reset_stats()
    
    def shutdown(self):
        """关闭并行搜索器"""
        self.executor.shutdown(wait=True)
        
        if self.shared_inference_engine:
            self.shared_inference_engine.stop_batch_processing()
            self.shared_inference_engine.clear_cache()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.shutdown()