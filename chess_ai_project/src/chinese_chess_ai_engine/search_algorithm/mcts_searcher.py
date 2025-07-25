"""
蒙特卡洛树搜索器

实现MCTS算法的核心搜索逻辑。
"""

import time
import random
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable

from .mcts_node import MCTSNode, MCTSConfig
from ..rules_engine import ChessBoard, Move
from ..neural_network import ChessNet, InferenceEngine


class MCTSSearcher:
    """
    蒙特卡洛树搜索器
    
    实现MCTS算法的选择、扩展、模拟、回传四个步骤。
    """
    
    def __init__(
        self,
        model: ChessNet,
        config: Optional[MCTSConfig] = None,
        inference_engine: Optional[InferenceEngine] = None
    ):
        """
        初始化MCTS搜索器
        
        Args:
            model: 神经网络模型
            config: MCTS配置
            inference_engine: 推理引擎（可选）
        """
        self.model = model
        self.config = config or MCTSConfig()
        
        # 创建或使用现有的推理引擎
        if inference_engine is not None:
            self.inference_engine = inference_engine
        else:
            self.inference_engine = InferenceEngine(model=model, device='auto')
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.stats = {
            'total_simulations': 0,
            'total_search_time': 0.0,
            'nodes_created': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def search(
        self,
        root_board: ChessBoard,
        num_simulations: Optional[int] = None,
        time_limit: Optional[float] = None
    ) -> MCTSNode:
        """
        执行MCTS搜索
        
        Args:
            root_board: 根节点棋盘状态
            num_simulations: 模拟次数（覆盖配置）
            time_limit: 时间限制（覆盖配置）
            
        Returns:
            MCTSNode: 搜索完成的根节点
        """
        start_time = time.time()
        
        # 使用参数或配置中的值
        num_sims = num_simulations or self.config.num_simulations
        time_lim = time_limit or self.config.time_limit
        
        # 创建根节点
        root = MCTSNode(board=root_board)
        
        # 如果是终端状态，直接返回
        if root.is_terminal:
            return root
        
        # 为根节点添加Dirichlet噪声
        self._add_dirichlet_noise(root)
        
        # 执行模拟
        simulations_done = 0
        
        for sim in range(num_sims):
            # 检查时间限制
            if time_lim and (time.time() - start_time) > time_lim:
                self.logger.info(f"达到时间限制，完成{sim}次模拟")
                break
            
            # 执行一次模拟
            self._simulate_once(root)
            simulations_done += 1
        
        # 更新统计信息
        search_time = time.time() - start_time
        self.stats['total_simulations'] += simulations_done
        self.stats['total_search_time'] += search_time
        
        self.logger.info(
            f"MCTS搜索完成: {simulations_done}次模拟, "
            f"耗时{search_time:.3f}s, "
            f"根节点访问{root.visit_count}次"
        )
        
        return root
    
    def _simulate_once(self, root: MCTSNode):
        """
        执行一次MCTS模拟
        
        Args:
            root: 根节点
        """
        # 1. 选择阶段：从根节点选择到叶节点
        leaf = self._select_leaf(root)
        
        # 2. 扩展和评估阶段
        value = self._expand_and_evaluate(leaf)
        
        # 3. 回传阶段
        leaf.backup(value)
    
    def _select_leaf(self, root: MCTSNode) -> MCTSNode:
        """
        选择叶节点（选择阶段）
        
        Args:
            root: 根节点
            
        Returns:
            MCTSNode: 选中的叶节点
        """
        current = root
        depth = 0
        max_depth = min(self.config.max_depth, 50)  # 限制最大深度
        
        # 沿着树向下选择，直到找到叶节点
        while (current.is_expanded and 
               not current.is_terminal and 
               depth < max_depth and
               current.children):  # 确保有子节点
            try:
                current = current.select_best_child(self.config.c_puct)
                depth += 1
            except (ValueError, KeyError):
                # 如果选择失败，返回当前节点
                break
        
        return current
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        扩展和评估节点
        
        Args:
            node: 要扩展的节点
            
        Returns:
            float: 节点的评估值
        """
        # 如果是终端节点，直接返回游戏结果
        if node.is_terminal:
            winner = node.board.get_winner()
            if winner is None:
                return 0.0  # 平局
            elif winner == node.board.current_player:
                return 1.0  # 当前玩家获胜
            else:
                return -1.0  # 当前玩家失败
        
        # 使用神经网络评估
        value, policy = self._evaluate_with_network(node.board)
        
        # 如果节点未扩展，进行扩展
        if not node.is_expanded:
            move_priors = self._policy_to_move_priors(node.board, policy)
            node.expand(move_priors)
            self.stats['nodes_created'] += len(node.children)
        
        return value
    
    def _evaluate_with_network(self, board: ChessBoard) -> Tuple[float, np.ndarray]:
        """
        使用神经网络评估棋盘
        
        Args:
            board: 棋盘状态
            
        Returns:
            Tuple[float, np.ndarray]: (价值评估, 策略分布)
        """
        try:
            value, policy = self.inference_engine.predict(board)
            self.stats['cache_hits'] += 1
            return value, policy
        except Exception as e:
            self.logger.warning(f"神经网络评估失败: {e}")
            self.stats['cache_misses'] += 1
            # 返回默认值
            return 0.0, np.ones(8100) / 8100
    
    def _policy_to_move_priors(self, board: ChessBoard, policy: np.ndarray) -> Dict[Move, float]:
        """
        将策略分布转换为走法先验概率
        
        Args:
            board: 棋盘状态
            policy: 策略分布
            
        Returns:
            Dict[Move, float]: 走法先验概率字典
        """
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return {}
        
        # 简化版本：为所有合法走法分配均匀概率
        uniform_prob = 1.0 / len(legal_moves)
        move_priors = {move: uniform_prob for move in legal_moves}
        
        return move_priors
    
    def _add_dirichlet_noise(self, root: MCTSNode):
        """
        为根节点添加Dirichlet噪声（增加探索性）
        
        Args:
            root: 根节点
        """
        if not root.is_expanded:
            # 先扩展根节点
            legal_moves = root.get_legal_moves()
            if legal_moves:
                uniform_priors = {move: 1.0 / len(legal_moves) for move in legal_moves}
                root.expand(uniform_priors)
        
        if not root.children:
            return
        
        # 生成Dirichlet噪声
        alpha = self.config.dirichlet_alpha
        epsilon = self.config.dirichlet_epsilon
        
        noise = np.random.dirichlet([alpha] * len(root.children))
        
        # 应用噪声到先验概率
        for i, child in enumerate(root.children.values()):
            original_prior = child.prior_probability
            child.prior_probability = (1 - epsilon) * original_prior + epsilon * noise[i]
    
    def get_best_move(self, root: MCTSNode) -> Optional[Move]:
        """
        获取最佳走法
        
        Args:
            root: 根节点
            
        Returns:
            Optional[Move]: 最佳走法
        """
        if not root.children:
            return None
        
        return root.select_best_move()
    
    def get_action_probabilities(
        self,
        root: MCTSNode,
        temperature: Optional[float] = None
    ) -> Dict[Move, float]:
        """
        获取动作概率分布
        
        Args:
            root: 根节点
            temperature: 温度参数（覆盖配置）
            
        Returns:
            Dict[Move, float]: 动作概率字典
        """
        temp = temperature if temperature is not None else self.config.temperature
        return root.get_action_probabilities(temp)
    
    def get_principal_variation(self, root: MCTSNode, max_depth: int = 10) -> List[Move]:
        """
        获取主要变化
        
        Args:
            root: 根节点
            max_depth: 最大深度
            
        Returns:
            List[Move]: 主要变化的走法序列
        """
        return root.get_principal_variation(max_depth)
    
    def analyze_position(
        self,
        board: ChessBoard,
        num_simulations: Optional[int] = None,
        return_details: bool = False
    ) -> Dict:
        """
        分析棋局位置
        
        Args:
            board: 棋盘状态
            num_simulations: 模拟次数
            return_details: 是否返回详细信息
            
        Returns:
            Dict: 分析结果
        """
        # 执行搜索
        root = self.search(board, num_simulations)
        
        # 基本结果
        result = {
            'best_move': self.get_best_move(root),
            'evaluation': root.average_value,
            'visit_count': root.visit_count,
            'principal_variation': self.get_principal_variation(root),
            'action_probabilities': self.get_action_probabilities(root)
        }
        
        if return_details:
            # 添加详细信息
            result.update({
                'tree_info': root.get_tree_info(),
                'top_moves': self._get_top_moves(root, top_k=5),
                'search_stats': self.get_stats()
            })
        
        return result
    
    def _get_top_moves(self, root: MCTSNode, top_k: int = 5) -> List[Dict]:
        """
        获取前k个最佳走法
        
        Args:
            root: 根节点
            top_k: 返回的走法数量
            
        Returns:
            List[Dict]: 前k个走法的信息
        """
        if not root.children:
            return []
        
        # 按访问次数排序
        sorted_children = sorted(
            root.children.items(),
            key=lambda x: x[1].visit_count,
            reverse=True
        )
        
        top_moves = []
        for move, child in sorted_children[:top_k]:
            top_moves.append({
                'move': move,
                'visits': child.visit_count,
                'value': child.average_value,
                'prior': child.prior_probability,
                'visit_percentage': child.visit_count / root.visit_count if root.visit_count > 0 else 0
            })
        
        return top_moves
    
    def get_stats(self) -> Dict:
        """
        获取搜索统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        
        # 计算平均值
        if stats['total_simulations'] > 0:
            stats['avg_time_per_simulation'] = stats['total_search_time'] / stats['total_simulations']
        else:
            stats['avg_time_per_simulation'] = 0.0
        
        # 添加缓存命中率
        total_evaluations = stats['cache_hits'] + stats['cache_misses']
        if total_evaluations > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_evaluations
        else:
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_simulations': 0,
            'total_search_time': 0.0,
            'nodes_created': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def set_config(self, config: MCTSConfig):
        """
        设置新的配置
        
        Args:
            config: 新的MCTS配置
        """
        self.config = config
        self.logger.info(f"MCTS配置已更新: {config.to_dict()}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        # 清理资源
        if hasattr(self.inference_engine, 'clear_cache'):
            self.inference_engine.clear_cache()