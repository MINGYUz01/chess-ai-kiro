"""
象棋AI分析和决策核心

实现位置分析、最佳走法计算、胜率评估等核心功能。
"""

import time
import logging
import math
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np

from ..rules_engine import ChessBoard, Move
from ..neural_network import ChessNet, ModelManager
from ..search_algorithm import MCTSSearcher, MCTSConfig
from ..training_framework import BoardEncoder


@dataclass
class AnalysisResult:
    """
    分析结果数据结构
    """
    best_move: Move
    evaluation: float
    win_probability: Tuple[float, float]  # (red_win_prob, black_win_prob)
    principal_variation: List[Move]
    top_moves: List[Tuple[Move, float]]
    search_depth: int
    nodes_searched: int
    time_used: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AIConfig:
    """
    AI配置类
    """
    model_path: str
    search_time: float = 5.0
    max_simulations: int = 1000
    difficulty_level: int = 5
    use_opening_book: bool = True
    use_endgame_tablebase: bool = True
    device: str = 'auto'
    
    # MCTS参数
    c_puct: float = 1.0
    temperature: float = 0.1
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # 难度调整参数
    min_search_time: float = 0.1
    max_search_time: float = 30.0
    min_simulations: int = 50
    max_simulations: int = 5000
    
    # 随机性参数
    randomness_factor: float = 0.1  # 难度越低随机性越高
    
    def __post_init__(self):
        # 自动设置设备
        if self.device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'


class DifficultyManager:
    """
    难度管理器
    
    根据难度级别调整AI的搜索参数和随机性。
    """
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_search_params(self, difficulty_level: int) -> Dict[str, Any]:
        """
        根据难度级别获取搜索参数
        
        Args:
            difficulty_level: 难度级别 (1-10)
            
        Returns:
            Dict[str, Any]: 搜索参数
        """
        # 确保难度级别在有效范围内
        difficulty_level = max(1, min(10, difficulty_level))
        
        # 线性插值计算参数
        ratio = (difficulty_level - 1) / 9.0  # 0.0 到 1.0
        
        # 搜索时间
        search_time = (
            self.config.min_search_time + 
            ratio * (self.config.max_search_time - self.config.min_search_time)
        )
        
        # 模拟次数
        simulations = int(
            self.config.min_simulations + 
            ratio * (self.config.max_simulations - self.config.min_simulations)
        )
        
        # 温度参数（难度越低温度越高，随机性越大）
        temperature = 0.5 - ratio * 0.4  # 0.5 到 0.1
        
        # 随机性因子
        randomness = self.config.randomness_factor * (1.0 - ratio)
        
        return {
            'search_time': search_time,
            'simulations': simulations,
            'temperature': temperature,
            'randomness': randomness,
            'c_puct': self.config.c_puct,
            'dirichlet_alpha': self.config.dirichlet_alpha,
            'dirichlet_epsilon': self.config.dirichlet_epsilon
        }
    
    def should_add_randomness(self, difficulty_level: int) -> bool:
        """
        判断是否应该添加随机性
        
        Args:
            difficulty_level: 难度级别
            
        Returns:
            bool: 是否添加随机性
        """
        params = self.get_search_params(difficulty_level)
        return random.random() < params['randomness']


class ChessAI:
    """
    象棋AI分析和决策核心
    
    提供位置分析、最佳走法计算、胜率评估等功能。
    """
    
    def __init__(self, model_path: str, config: Optional[AIConfig] = None):
        """
        初始化象棋AI
        
        Args:
            model_path: 模型文件路径
            config: AI配置
        """
        self.config = config or AIConfig(model_path=model_path)
        self.logger = logging.getLogger(__name__)
        
        # 设置设备
        self.device = torch.device(self.config.device)
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 创建组件
        self.board_encoder = BoardEncoder()
        self.difficulty_manager = DifficultyManager(self.config)
        
        # 创建MCTS配置
        self.base_mcts_config = MCTSConfig(
            num_simulations=self.config.max_simulations,
            c_puct=self.config.c_puct,
            temperature=self.config.temperature,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon
        )
        
        # 统计信息
        self.stats = {
            'positions_analyzed': 0,
            'total_search_time': 0.0,
            'total_nodes_searched': 0,
            'average_depth': 0.0
        }
        
        self.logger.info(f"ChessAI初始化完成，设备: {self.device}")
    
    def _load_model(self, model_path: str) -> ChessNet:
        """
        加载神经网络模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            ChessNet: 加载的模型
        """
        try:
            if Path(model_path).exists():
                # 使用ModelManager加载模型
                model_manager = ModelManager()
                model = model_manager.load_model(model_path)
                model.to(self.device)
                model.eval()
                self.logger.info(f"成功加载模型: {model_path}")
                return model
            else:
                # 如果模型文件不存在，创建一个新模型
                self.logger.warning(f"模型文件不存在: {model_path}，创建新模型")
                model = ChessNet()
                model.to(self.device)
                model.eval()
                return model
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            # 创建默认模型
            model = ChessNet()
            model.to(self.device)
            model.eval()
            return model
    
    def analyze_position(
        self, 
        board: ChessBoard, 
        depth: Optional[int] = None,
        time_limit: Optional[float] = None
    ) -> AnalysisResult:
        """
        分析棋局位置
        
        Args:
            board: 棋盘状态
            depth: 搜索深度（以模拟次数表示）
            time_limit: 时间限制
            
        Returns:
            AnalysisResult: 分析结果
        """
        start_time = time.time()
        
        # 获取搜索参数
        search_params = self.difficulty_manager.get_search_params(self.config.difficulty_level)
        
        # 使用提供的参数或默认参数
        simulations = depth or search_params['simulations']
        search_time = time_limit or search_params['search_time']
        
        # 创建MCTS搜索器
        mcts_config = MCTSConfig(
            num_simulations=simulations,
            c_puct=search_params['c_puct'],
            temperature=search_params['temperature'],
            dirichlet_alpha=search_params['dirichlet_alpha'],
            dirichlet_epsilon=search_params['dirichlet_epsilon']
        )
        
        searcher = MCTSSearcher(self.model, mcts_config)
        
        # 执行搜索
        root_node = searcher.search(board, simulations)
        
        # 获取分析结果
        best_move = searcher.get_best_move(root_node)
        evaluation = searcher.get_value_estimate(root_node)
        action_probs = searcher.get_action_probabilities(root_node, temperature=0.1)
        
        # 获取合法走法
        legal_moves = board.get_legal_moves()
        
        # 计算胜率
        win_prob_red, win_prob_black = self._calculate_win_probability(evaluation, board.current_player)
        
        # 获取主要变化
        principal_variation = self._get_principal_variation(root_node, max_depth=10)
        
        # 获取最佳走法列表
        top_moves = self._get_top_moves(legal_moves, action_probs, num_moves=5)
        
        # 计算搜索统计
        search_time_used = time.time() - start_time
        nodes_searched = getattr(root_node, 'visit_count', simulations)
        search_depth = len(principal_variation)
        
        # 更新统计信息
        self._update_stats(search_time_used, nodes_searched, search_depth)
        
        # 创建分析结果
        result = AnalysisResult(
            best_move=best_move,
            evaluation=evaluation,
            win_probability=(win_prob_red, win_prob_black),
            principal_variation=principal_variation,
            top_moves=top_moves,
            search_depth=search_depth,
            nodes_searched=nodes_searched,
            time_used=search_time_used,
            metadata={
                'difficulty_level': self.config.difficulty_level,
                'simulations': simulations,
                'temperature': search_params['temperature'],
                'current_player': board.current_player
            }
        )
        
        self.logger.debug(
            f"位置分析完成: 最佳走法={best_move.to_coordinate_notation() if best_move else 'None'}, "
            f"评估={evaluation:.3f}, 时间={search_time_used:.2f}s"
        )
        
        return result
    
    def get_best_move(
        self, 
        board: ChessBoard, 
        time_limit: float = None
    ) -> Optional[Move]:
        """
        获取最佳走法
        
        Args:
            board: 棋盘状态
            time_limit: 时间限制
            
        Returns:
            Optional[Move]: 最佳走法
        """
        if time_limit is None:
            time_limit = self.config.search_time
        
        try:
            analysis = self.analyze_position(board, time_limit=time_limit)
            
            # 根据难度级别决定是否添加随机性
            if self.difficulty_manager.should_add_randomness(self.config.difficulty_level):
                # 从前几个最佳走法中随机选择
                if analysis.top_moves:
                    num_choices = min(3, len(analysis.top_moves))
                    chosen_move, _ = random.choice(analysis.top_moves[:num_choices])
                    return chosen_move
            
            return analysis.best_move
            
        except Exception as e:
            self.logger.error(f"获取最佳走法失败: {e}")
            # 返回第一个合法走法作为备选
            legal_moves = board.get_legal_moves()
            return legal_moves[0] if legal_moves else None
    
    def get_top_moves(
        self, 
        board: ChessBoard, 
        num_moves: int = 5
    ) -> List[Tuple[Move, float]]:
        """
        获取最佳走法列表
        
        Args:
            board: 棋盘状态
            num_moves: 返回的走法数量
            
        Returns:
            List[Tuple[Move, float]]: 走法和评估值的列表
        """
        try:
            analysis = self.analyze_position(board)
            return analysis.top_moves[:num_moves]
        except Exception as e:
            self.logger.error(f"获取最佳走法列表失败: {e}")
            return []
    
    def evaluate_position(self, board: ChessBoard) -> float:
        """
        评估棋局位置
        
        Args:
            board: 棋盘状态
            
        Returns:
            float: 位置评估值 (-1到1)
        """
        try:
            # 编码棋盘
            board_tensor = self.board_encoder.encode_board(board)
            board_tensor = board_tensor.unsqueeze(0).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                value, _ = self.model(board_tensor)
                evaluation = value.item()
            
            # 从当前玩家角度返回评估
            return evaluation * board.current_player
            
        except Exception as e:
            self.logger.error(f"位置评估失败: {e}")
            return 0.0
    
    def calculate_win_probability(self, board: ChessBoard) -> Tuple[float, float]:
        """
        计算胜率
        
        Args:
            board: 棋盘状态
            
        Returns:
            Tuple[float, float]: (红方胜率, 黑方胜率)
        """
        evaluation = self.evaluate_position(board)
        return self._calculate_win_probability(evaluation, board.current_player)
    
    def set_difficulty_level(self, level: int):
        """
        设置AI难度级别
        
        Args:
            level: 难度级别 (1-10)
        """
        level = max(1, min(10, level))
        self.config.difficulty_level = level
        self.logger.info(f"AI难度级别设置为: {level}")
    
    def get_difficulty_level(self) -> int:
        """
        获取当前难度级别
        
        Returns:
            int: 当前难度级别
        """
        return self.config.difficulty_level
    
    def _calculate_win_probability(self, evaluation: float, current_player: int) -> Tuple[float, float]:
        """
        根据评估值计算胜率
        
        Args:
            evaluation: 评估值
            current_player: 当前玩家
            
        Returns:
            Tuple[float, float]: (红方胜率, 黑方胜率)
        """
        # 使用sigmoid函数将评估值转换为胜率
        # evaluation是从当前玩家角度的评估
        win_prob = 1.0 / (1.0 + math.exp(-evaluation * 4.0))  # 放大系数4.0
        
        if current_player == 1:  # 红方回合
            red_win_prob = win_prob
            black_win_prob = 1.0 - win_prob
        else:  # 黑方回合
            red_win_prob = 1.0 - win_prob
            black_win_prob = win_prob
        
        return red_win_prob, black_win_prob
    
    def _get_principal_variation(self, root_node, max_depth: int = 10) -> List[Move]:
        """
        获取主要变化
        
        Args:
            root_node: 根节点
            max_depth: 最大深度
            
        Returns:
            List[Move]: 主要变化走法列表
        """
        pv = []
        current_node = root_node
        
        for _ in range(max_depth):
            if not hasattr(current_node, 'children') or not current_node.children:
                break
            
            # 选择访问次数最多的子节点
            best_child = max(
                current_node.children.values(),
                key=lambda node: getattr(node, 'visit_count', 0)
            )
            
            if hasattr(best_child, 'move') and best_child.move:
                pv.append(best_child.move)
                current_node = best_child
            else:
                break
        
        return pv
    
    def _get_top_moves(
        self, 
        legal_moves: List[Move], 
        action_probs: np.ndarray, 
        num_moves: int = 5
    ) -> List[Tuple[Move, float]]:
        """
        获取最佳走法列表
        
        Args:
            legal_moves: 合法走法列表
            action_probs: 动作概率
            num_moves: 返回的走法数量
            
        Returns:
            List[Tuple[Move, float]]: 走法和概率的列表
        """
        if len(action_probs) != len(legal_moves):
            # 如果长度不匹配，使用均匀分布
            action_probs = np.ones(len(legal_moves)) / len(legal_moves)
        
        # 创建走法和概率的配对
        move_probs = list(zip(legal_moves, action_probs))
        
        # 按概率排序
        move_probs.sort(key=lambda x: x[1], reverse=True)
        
        return move_probs[:num_moves]
    
    def _update_stats(self, search_time: float, nodes_searched: int, search_depth: int):
        """
        更新统计信息
        
        Args:
            search_time: 搜索时间
            nodes_searched: 搜索节点数
            search_depth: 搜索深度
        """
        self.stats['positions_analyzed'] += 1
        self.stats['total_search_time'] += search_time
        self.stats['total_nodes_searched'] += nodes_searched
        
        # 更新平均深度
        total_positions = self.stats['positions_analyzed']
        self.stats['average_depth'] = (
            (self.stats['average_depth'] * (total_positions - 1) + search_depth) 
            / total_positions
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取AI统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.stats.copy()
        
        if stats['positions_analyzed'] > 0:
            stats['average_search_time'] = stats['total_search_time'] / stats['positions_analyzed']
            stats['average_nodes_per_position'] = stats['total_nodes_searched'] / stats['positions_analyzed']
            stats['nodes_per_second'] = (
                stats['total_nodes_searched'] / stats['total_search_time'] 
                if stats['total_search_time'] > 0 else 0
            )
        
        stats['difficulty_level'] = self.config.difficulty_level
        stats['device'] = str(self.device)
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'positions_analyzed': 0,
            'total_search_time': 0.0,
            'total_nodes_searched': 0,
            'average_depth': 0.0
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"ChessAI(difficulty={self.config.difficulty_level}, device={self.device})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()