"""
蒙特卡洛树搜索节点

实现MCTS算法中的树节点数据结构和相关操作。
"""

import math
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..rules_engine import ChessBoard, Move


@dataclass
class MCTSNode:
    """
    MCTS树节点
    
    表示搜索树中的一个节点，包含棋局状态、统计信息和子节点。
    """
    
    # 棋局状态
    board: ChessBoard
    move: Optional[Move] = None  # 到达此节点的走法
    parent: Optional['MCTSNode'] = None
    
    # 子节点
    children: Dict[Move, 'MCTSNode'] = field(default_factory=dict)
    
    # 统计信息
    visit_count: int = 0
    value_sum: float = 0.0
    prior_probability: float = 0.0  # 先验概率（来自神经网络）
    
    # 线程安全
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    # 缓存的合法走法
    _legal_moves: Optional[List[Move]] = field(default=None, init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保每个节点都有独立的锁
        if not hasattr(self, '_lock') or self._lock is None:
            self._lock = threading.Lock()
    
    @property
    def is_expanded(self) -> bool:
        """
        检查节点是否已扩展
        
        Returns:
            bool: 是否已扩展
        """
        return len(self.children) > 0
    
    @property
    def is_terminal(self) -> bool:
        """
        检查节点是否为终端节点
        
        Returns:
            bool: 是否为终端节点
        """
        return self.board.is_game_over()
    
    @property
    def average_value(self) -> float:
        """
        获取平均价值
        
        Returns:
            float: 平均价值
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_legal_moves(self) -> List[Move]:
        """
        获取合法走法（带缓存）
        
        Returns:
            List[Move]: 合法走法列表
        """
        if self._legal_moves is None:
            self._legal_moves = self.board.get_legal_moves()
        return self._legal_moves
    
    def ucb_score(self, c_puct: float = 1.0, parent_visits: Optional[int] = None) -> float:
        """
        计算UCB分数（Upper Confidence Bound）
        
        Args:
            c_puct: 探索常数
            parent_visits: 父节点访问次数
            
        Returns:
            float: UCB分数
        """
        if self.visit_count == 0:
            return float('inf')  # 未访问的节点优先级最高
        
        if parent_visits is None:
            parent_visits = self.parent.visit_count if self.parent else 1
        
        # Q值（平均价值）
        q_value = self.average_value
        
        # U值（探索项）
        exploration_term = c_puct * self.prior_probability * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return q_value + exploration_term
    
    def puct_score(self, c_puct: float = 1.0) -> float:
        """
        计算PUCT分数（Polynomial Upper Confidence Trees）
        
        Args:
            c_puct: 探索常数
            
        Returns:
            float: PUCT分数
        """
        if self.visit_count == 0:
            return float('inf')
        
        parent_visits = self.parent.visit_count if self.parent else 1
        
        # 价值项
        value_term = self.average_value
        
        # 探索项
        exploration_term = c_puct * self.prior_probability * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return value_term + exploration_term
    
    def select_best_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """
        选择最佳子节点（基于UCB分数）
        
        Args:
            c_puct: 探索常数
            
        Returns:
            MCTSNode: 最佳子节点
        """
        if not self.children:
            raise ValueError("节点没有子节点")
        
        best_child = None
        best_score = float('-inf')
        
        for child in self.children.values():
            score = child.ucb_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def select_most_visited_child(self) -> 'MCTSNode':
        """
        选择访问次数最多的子节点
        
        Returns:
            MCTSNode: 访问次数最多的子节点
        """
        if not self.children:
            raise ValueError("节点没有子节点")
        
        return max(self.children.values(), key=lambda child: child.visit_count)
    
    def select_best_move(self) -> Move:
        """
        选择最佳走法（基于访问次数）
        
        Returns:
            Move: 最佳走法
        """
        if not self.children:
            raise ValueError("节点没有子节点")
        
        best_child = self.select_most_visited_child()
        return best_child.move
    
    def expand(self, move_priors: Dict[Move, float]) -> List['MCTSNode']:
        """
        扩展节点（添加子节点）
        
        Args:
            move_priors: 走法先验概率字典
            
        Returns:
            List[MCTSNode]: 新创建的子节点列表
        """
        with self._lock:
            if self.is_expanded:
                return list(self.children.values())
            
            legal_moves = self.get_legal_moves()
            new_children = []
            
            for move in legal_moves:
                # 创建新的棋盘状态
                new_board = self.board.make_move(move)
                
                # 获取先验概率
                prior_prob = move_priors.get(move, 1.0 / len(legal_moves))
                
                # 创建子节点
                child = MCTSNode(
                    board=new_board,
                    move=move,
                    parent=self,
                    prior_probability=prior_prob
                )
                
                self.children[move] = child
                new_children.append(child)
            
            return new_children
    
    def backup(self, value: float):
        """
        回传价值（更新统计信息）
        
        Args:
            value: 要回传的价值
        """
        with self._lock:
            self.visit_count += 1
            self.value_sum += value
        
        # 递归回传到父节点
        if self.parent is not None:
            # 对手的价值是相反的
            self.parent.backup(-value)
    
    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[Move, float]:
        """
        获取动作概率分布
        
        Args:
            temperature: 温度参数（控制探索程度）
            
        Returns:
            Dict[Move, float]: 动作概率字典
        """
        if not self.children:
            return {}
        
        if temperature == 0:
            # 温度为0时，选择访问次数最多的动作
            best_child = self.select_most_visited_child()
            return {move: 1.0 if child == best_child else 0.0 
                   for move, child in self.children.items()}
        
        # 计算基于访问次数的概率
        visit_counts = {move: child.visit_count for move, child in self.children.items()}
        
        if temperature == 1.0:
            # 温度为1时，概率正比于访问次数
            total_visits = sum(visit_counts.values())
            if total_visits == 0:
                # 如果没有访问过，使用均匀分布
                prob = 1.0 / len(self.children)
                return {move: prob for move in self.children.keys()}
            
            return {move: count / total_visits for move, count in visit_counts.items()}
        
        # 其他温度值，使用softmax
        import numpy as np
        
        # 将访问次数转换为对数概率
        log_probs = {move: math.log(max(count, 1)) / temperature 
                    for move, count in visit_counts.items()}
        
        # 计算softmax
        max_log_prob = max(log_probs.values())
        exp_probs = {move: math.exp(log_prob - max_log_prob) 
                    for move, log_prob in log_probs.items()}
        
        total_exp = sum(exp_probs.values())
        return {move: exp_prob / total_exp for move, exp_prob in exp_probs.items()}
    
    def get_principal_variation(self, max_depth: int = 10) -> List[Move]:
        """
        获取主要变化（最佳路径）
        
        Args:
            max_depth: 最大深度
            
        Returns:
            List[Move]: 主要变化的走法序列
        """
        pv = []
        current = self
        depth = 0
        
        while current.children and depth < max_depth:
            best_child = current.select_most_visited_child()
            if best_child.move:
                pv.append(best_child.move)
            current = best_child
            depth += 1
        
        return pv
    
    def get_tree_info(self) -> Dict:
        """
        获取树的统计信息
        
        Returns:
            Dict: 树的统计信息
        """
        def count_nodes(node: 'MCTSNode') -> int:
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count
        
        def max_depth(node: 'MCTSNode') -> int:
            if not node.children:
                return 0
            return 1 + max(max_depth(child) for child in node.children.values())
        
        return {
            'total_nodes': count_nodes(self),
            'max_depth': max_depth(self),
            'root_visits': self.visit_count,
            'root_value': self.average_value,
            'children_count': len(self.children),
            'is_expanded': self.is_expanded,
            'is_terminal': self.is_terminal
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        move_str = str(self.move) if self.move else "Root"
        return (f"MCTSNode({move_str}, visits={self.visit_count}, "
                f"value={self.average_value:.3f}, children={len(self.children)})")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


class MCTSConfig:
    """
    MCTS配置类
    
    包含MCTS算法的各种参数配置。
    """
    
    def __init__(
        self,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        max_depth: int = 100,
        time_limit: Optional[float] = None
    ):
        """
        初始化MCTS配置
        
        Args:
            num_simulations: 模拟次数
            c_puct: PUCT探索常数
            temperature: 温度参数
            dirichlet_alpha: Dirichlet噪声alpha参数
            dirichlet_epsilon: Dirichlet噪声epsilon参数
            max_depth: 最大搜索深度
            time_limit: 时间限制（秒）
        """
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_depth = max_depth
        self.time_limit = time_limit
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'num_simulations': self.num_simulations,
            'c_puct': self.c_puct,
            'temperature': self.temperature,
            'dirichlet_alpha': self.dirichlet_alpha,
            'dirichlet_epsilon': self.dirichlet_epsilon,
            'max_depth': self.max_depth,
            'time_limit': self.time_limit
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MCTSConfig':
        """从字典创建配置"""
        return cls(**config_dict)