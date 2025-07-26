"""
自对弈数据生成器

实现AlphaZero风格的自对弈数据生成系统。
"""

import time
import random
import logging
import threading
from typing import List, Dict, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from pathlib import Path
import uuid

from .training_example import TrainingExample, TrainingDataset
from .board_encoder import BoardEncoder
from ..rules_engine import ChessBoard, Move
from ..neural_network import ChessNet
from ..search_algorithm import MCTSSearcher, MCTSConfig


class SelfPlayConfig:
    """
    自对弈配置类
    """
    
    def __init__(
        self,
        num_games: int = 100,
        max_game_length: int = 200,
        mcts_simulations: int = 800,
        temperature_threshold: int = 30,
        temperature_high: float = 1.0,
        temperature_low: float = 0.1,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        num_workers: int = 4,
        save_interval: int = 10,
        resign_threshold: float = -0.9,
        resign_check_moves: int = 10
    ):
        """
        初始化自对弈配置
        
        Args:
            num_games: 生成的游戏数量
            max_game_length: 最大游戏长度
            mcts_simulations: MCTS模拟次数
            temperature_threshold: 温度切换阈值
            temperature_high: 高温度值
            temperature_low: 低温度值
            c_puct: MCTS探索常数
            dirichlet_alpha: Dirichlet噪声alpha参数
            dirichlet_epsilon: Dirichlet噪声epsilon参数
            num_workers: 并行工作线程数
            save_interval: 保存间隔
            resign_threshold: 认输阈值
            resign_check_moves: 认输检查步数
        """
        self.num_games = num_games
        self.max_game_length = max_game_length
        self.mcts_simulations = mcts_simulations
        self.temperature_threshold = temperature_threshold
        self.temperature_high = temperature_high
        self.temperature_low = temperature_low
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.num_workers = num_workers
        self.save_interval = save_interval
        self.resign_threshold = resign_threshold
        self.resign_check_moves = resign_check_moves


class GameResult:
    """
    游戏结果类
    """
    
    def __init__(
        self,
        winner: int,
        game_length: int,
        training_examples: List[TrainingExample],
        game_id: str,
        metadata: Optional[Dict] = None
    ):
        """
        初始化游戏结果
        
        Args:
            winner: 获胜者 (1: 红方, -1: 黑方, 0: 平局)
            game_length: 游戏长度
            training_examples: 训练样本列表
            game_id: 游戏ID
            metadata: 元数据
        """
        self.winner = winner
        self.game_length = game_length
        self.training_examples = training_examples
        self.game_id = game_id
        self.metadata = metadata or {}


class SelfPlayGenerator:
    """
    自对弈数据生成器
    
    使用神经网络和MCTS进行自对弈，生成训练数据。
    """
    
    def __init__(
        self,
        model: ChessNet,
        config: Optional[SelfPlayConfig] = None,
        board_encoder: Optional[BoardEncoder] = None
    ):
        """
        初始化自对弈生成器
        
        Args:
            model: 神经网络模型
            config: 自对弈配置
            board_encoder: 棋盘编码器
        """
        self.model = model
        self.config = config or SelfPlayConfig()
        self.board_encoder = board_encoder or BoardEncoder()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建MCTS配置
        self.mcts_config = MCTSConfig(
            num_simulations=self.config.mcts_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon
        )
        
        # 统计信息
        self.stats = {
            'games_played': 0,
            'total_moves': 0,
            'red_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'resignations': 0,
            'average_game_length': 0.0,
            'total_time': 0.0
        }
        
        # 线程锁
        self._lock = threading.Lock()
    
    def generate_game(self, game_id: Optional[str] = None) -> GameResult:
        """
        生成单个自对弈游戏
        
        Args:
            game_id: 游戏ID，如果为None则自动生成
            
        Returns:
            GameResult: 游戏结果
        """
        if game_id is None:
            game_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        # 初始化游戏状态
        board = ChessBoard()
        training_examples = []
        move_history = []
        board_history = []
        
        # 创建MCTS搜索器
        searcher = MCTSSearcher(self.model, self.mcts_config)
        
        # 游戏循环
        for move_number in range(1, self.config.max_game_length + 1):
            # 检查游戏是否结束
            if board.is_game_over():
                break
            
            # 保存当前棋盘状态
            board_history.append(board.copy())
            
            # 编码棋盘状态（只使用当前状态，不包含历史）
            board_tensor = self.board_encoder.encode_board(board)
            
            # 获取合法走法
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            # 执行MCTS搜索
            root_node = searcher.search(board, self.config.mcts_simulations)
            
            # 获取策略概率
            temperature = (self.config.temperature_high 
                         if move_number <= self.config.temperature_threshold 
                         else self.config.temperature_low)
            
            action_probs = searcher.get_action_probabilities(root_node, temperature)
            
            # 创建策略目标向量
            move_probs = {}
            for i, move in enumerate(legal_moves):
                move_probs[move] = action_probs[i] if i < len(action_probs) else 0.0
            
            policy_target = self.board_encoder.create_policy_target(legal_moves, move_probs)
            
            # 确保action_probs长度与legal_moves匹配并归一化
            if len(action_probs) != len(legal_moves):
                # 如果长度不匹配，截断或填充
                if len(action_probs) > len(legal_moves):
                    action_probs = action_probs[:len(legal_moves)]
                else:
                    # 填充0
                    padded_probs = np.zeros(len(legal_moves))
                    padded_probs[:len(action_probs)] = action_probs
                    action_probs = padded_probs
            
            # 归一化概率
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
            else:
                # 如果所有概率都是0，使用均匀分布
                action_probs = np.ones(len(legal_moves)) / len(legal_moves)
            
            # 选择走法
            if move_number <= self.config.temperature_threshold:
                # 高温度阶段：按概率采样
                move_idx = np.random.choice(len(legal_moves), p=action_probs)
                selected_move = legal_moves[move_idx]
            else:
                # 低温度阶段：选择最佳走法
                best_move_idx = np.argmax(action_probs)
                selected_move = legal_moves[best_move_idx]
            
            # 获取当前局面评估
            value_estimate = searcher.get_value_estimate(root_node)
            
            # 创建训练样本（暂时不知道最终结果）
            training_example = TrainingExample(
                board_tensor=board_tensor,
                policy_target=policy_target,
                value_target=value_estimate,  # 临时值，稍后会更新
                game_result=0,  # 临时值，稍后会更新
                move_number=move_number,
                current_player=board.current_player,
                original_board=board.copy(),
                actual_move=selected_move,
                metadata={
                    'game_id': game_id,
                    'temperature': temperature,
                    'mcts_simulations': self.config.mcts_simulations,
                    'is_final_position': False
                }
            )
            
            training_examples.append(training_example)
            move_history.append(selected_move)
            
            # 执行走法
            board = board.make_move(selected_move)
            
            # 检查认输条件
            if (move_number >= self.config.resign_check_moves and 
                value_estimate < self.config.resign_threshold):
                # AI认输
                winner = -board.current_player  # 对手获胜
                break
        
        # 确定游戏结果
        if board.is_game_over():
            winner = board.get_winner()
        elif not hasattr(locals(), 'winner'):
            # 达到最大步数，判定为平局
            winner = 0
        
        # 更新训练样本的最终结果
        for i, example in enumerate(training_examples):
            # 从当前玩家角度计算价值目标
            if winner == 0:
                value_target = 0.0  # 平局
            elif winner == example.current_player:
                value_target = 1.0  # 当前玩家获胜
            else:
                value_target = -1.0  # 当前玩家失败
            
            # 更新训练样本
            example.value_target = value_target
            example.game_result = winner
            
            # 标记最后一个位置
            if i == len(training_examples) - 1:
                example.metadata['is_final_position'] = True
        
        # 更新统计信息
        with self._lock:
            self.stats['games_played'] += 1
            self.stats['total_moves'] += len(training_examples)
            
            if winner == 1:
                self.stats['red_wins'] += 1
            elif winner == -1:
                self.stats['black_wins'] += 1
            else:
                self.stats['draws'] += 1
            
            # 检查是否是认输
            if hasattr(locals(), 'winner') and 'value_estimate' in locals():
                if value_estimate < self.config.resign_threshold:
                    self.stats['resignations'] += 1
            
            # 更新平均游戏长度
            total_games = self.stats['games_played']
            self.stats['average_game_length'] = (
                (self.stats['average_game_length'] * (total_games - 1) + len(training_examples)) 
                / total_games
            )
            
            # 更新总时间
            game_time = time.time() - start_time
            self.stats['total_time'] += game_time
        
        # 创建游戏结果
        game_result = GameResult(
            winner=winner,
            game_length=len(training_examples),
            training_examples=training_examples,
            game_id=game_id,
            metadata={
                'move_history': [move.to_coordinate_notation() for move in move_history],
                'game_time': game_time,
                'final_board': board.to_fen(),
                'resignation': hasattr(locals(), 'winner') and 'value_estimate' in locals() and value_estimate < self.config.resign_threshold
            }
        )
        
        self.logger.info(f"游戏 {game_id} 完成: 获胜者={winner}, 步数={len(training_examples)}, 时间={game_time:.2f}s")
        
        return game_result
    
    def generate_games_parallel(
        self, 
        num_games: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[GameResult]:
        """
        并行生成多个自对弈游戏
        
        Args:
            num_games: 游戏数量，如果为None则使用配置中的数量
            progress_callback: 进度回调函数 (completed, total)
            
        Returns:
            List[GameResult]: 游戏结果列表
        """
        if num_games is None:
            num_games = self.config.num_games
        
        self.logger.info(f"开始生成 {num_games} 个自对弈游戏，使用 {self.config.num_workers} 个工作线程")
        
        start_time = time.time()
        game_results = []
        completed_games = 0
        
        # 使用线程池并行生成游戏
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # 提交所有任务
            future_to_game_id = {
                executor.submit(self.generate_game, f"game_{i:04d}"): f"game_{i:04d}"
                for i in range(num_games)
            }
            
            # 收集结果
            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    game_result = future.result()
                    game_results.append(game_result)
                    completed_games += 1
                    
                    # 调用进度回调
                    if progress_callback:
                        progress_callback(completed_games, num_games)
                    
                    # 定期保存
                    if completed_games % self.config.save_interval == 0:
                        self.logger.info(f"已完成 {completed_games}/{num_games} 个游戏")
                        
                except Exception as e:
                    self.logger.error(f"游戏 {game_id} 生成失败: {e}")
        
        total_time = time.time() - start_time
        self.logger.info(f"所有游戏生成完成，总时间: {total_time:.2f}s")
        
        return game_results
    
    def collect_training_data(
        self, 
        num_games: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> TrainingDataset:
        """
        收集训练数据
        
        Args:
            num_games: 游戏数量
            output_path: 输出路径，如果提供则保存数据集
            
        Returns:
            TrainingDataset: 训练数据集
        """
        # 生成游戏
        game_results = self.generate_games_parallel(num_games)
        
        # 收集所有训练样本
        all_examples = []
        for game_result in game_results:
            all_examples.extend(game_result.training_examples)
        
        # 创建数据集
        dataset = TrainingDataset(all_examples)
        
        # 保存数据集
        if output_path:
            dataset.save_to_file(output_path)
            self.logger.info(f"训练数据集已保存到: {output_path}")
        
        return dataset
    
    def save_training_data(
        self, 
        game_results: List[GameResult], 
        output_path: str,
        format: str = 'pickle'
    ):
        """
        保存训练数据到文件
        
        Args:
            game_results: 游戏结果列表
            output_path: 输出文件路径
            format: 保存格式 ('pickle', 'json')
        """
        # 收集所有训练样本
        all_examples = []
        for game_result in game_results:
            all_examples.extend(game_result.training_examples)
        
        # 创建数据集并保存
        dataset = TrainingDataset(all_examples)
        dataset.save_to_file(output_path, format)
        
        self.logger.info(f"已保存 {len(all_examples)} 个训练样本到 {output_path}")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        获取生成统计信息
        
        Returns:
            Dict[str, any]: 统计信息
        """
        with self._lock:
            stats = self.stats.copy()
        
        # 计算额外统计信息
        if stats['games_played'] > 0:
            stats['red_win_rate'] = stats['red_wins'] / stats['games_played']
            stats['black_win_rate'] = stats['black_wins'] / stats['games_played']
            stats['draw_rate'] = stats['draws'] / stats['games_played']
            stats['resignation_rate'] = stats['resignations'] / stats['games_played']
            stats['average_time_per_game'] = stats['total_time'] / stats['games_played']
            
            if stats['total_moves'] > 0:
                stats['average_time_per_move'] = stats['total_time'] / stats['total_moves']
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        with self._lock:
            self.stats = {
                'games_played': 0,
                'total_moves': 0,
                'red_wins': 0,
                'black_wins': 0,
                'draws': 0,
                'resignations': 0,
                'average_game_length': 0.0,
                'total_time': 0.0
            }
    
    def play_game_against_baseline(
        self, 
        baseline_model: ChessNet,
        num_games: int = 10
    ) -> Dict[str, any]:
        """
        与基准模型对弈
        
        Args:
            baseline_model: 基准模型
            num_games: 对弈局数
            
        Returns:
            Dict[str, any]: 对弈结果统计
        """
        results = {
            'new_model_wins': 0,
            'baseline_wins': 0,
            'draws': 0,
            'games': []
        }
        
        for game_idx in range(num_games):
            # 随机决定哪个模型执红
            new_model_color = 1 if random.random() < 0.5 else -1
            baseline_color = -new_model_color
            
            board = ChessBoard()
            move_count = 0
            
            while not board.is_game_over() and move_count < self.config.max_game_length:
                current_player = board.current_player
                
                if current_player == new_model_color:
                    # 新模型走棋
                    searcher = MCTSSearcher(self.model, self.mcts_config)
                    root_node = searcher.search(board, self.config.mcts_simulations)
                    action_probs = searcher.get_action_probabilities(root_node, 0.1)  # 低温度
                    legal_moves = board.get_legal_moves()
                    best_move_idx = np.argmax(action_probs)
                    move = legal_moves[best_move_idx]
                else:
                    # 基准模型走棋
                    searcher = MCTSSearcher(baseline_model, self.mcts_config)
                    root_node = searcher.search(board, self.config.mcts_simulations)
                    action_probs = searcher.get_action_probabilities(root_node, 0.1)  # 低温度
                    legal_moves = board.get_legal_moves()
                    best_move_idx = np.argmax(action_probs)
                    move = legal_moves[best_move_idx]
                
                board = board.make_move(move)
                move_count += 1
            
            # 记录结果
            winner = board.get_winner() if board.is_game_over() else 0
            
            game_result = {
                'winner': winner,
                'new_model_color': new_model_color,
                'baseline_color': baseline_color,
                'move_count': move_count
            }
            
            results['games'].append(game_result)
            
            if winner == new_model_color:
                results['new_model_wins'] += 1
            elif winner == baseline_color:
                results['baseline_wins'] += 1
            else:
                results['draws'] += 1
        
        # 计算胜率
        total_games = len(results['games'])
        results['new_model_win_rate'] = results['new_model_wins'] / total_games
        results['baseline_win_rate'] = results['baseline_wins'] / total_games
        results['draw_rate'] = results['draws'] / total_games
        
        self.logger.info(f"对弈结果: 新模型胜率={results['new_model_win_rate']:.2%}")
        
        return results
