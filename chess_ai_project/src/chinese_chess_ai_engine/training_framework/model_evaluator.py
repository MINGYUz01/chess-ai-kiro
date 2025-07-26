"""
模型评估器

实现模型性能评估、ELO等级分计算和基准测试功能。
"""

import time
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .training_config import EvaluationConfig
from .self_play_generator import GameResult
from ..rules_engine import ChessBoard, Move
from ..neural_network import ChessNet
from ..search_algorithm import MCTSSearcher, MCTSConfig


@dataclass
class EvaluationResult:
    """
    评估结果数据结构
    """
    model_name: str
    opponent_name: str
    wins: int
    losses: int
    draws: int
    total_games: int
    win_rate: float
    elo_change: float
    average_game_length: float
    average_time_per_move: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def score(self) -> float:
        """计算得分 (胜=1分, 平=0.5分, 负=0分)"""
        return (self.wins + 0.5 * self.draws) / self.total_games if self.total_games > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_name': self.model_name,
            'opponent_name': self.opponent_name,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'total_games': self.total_games,
            'win_rate': self.win_rate,
            'score': self.score,
            'elo_change': self.elo_change,
            'average_game_length': self.average_game_length,
            'average_time_per_move': self.average_time_per_move,
            'metadata': self.metadata
        }


@dataclass
class BenchmarkPosition:
    """
    基准测试局面
    """
    fen: str
    description: str
    best_moves: List[str]  # 最佳走法列表（坐标记法）
    difficulty: int = 1    # 难度等级 1-5
    category: str = "general"  # 类别：opening, middlegame, endgame, tactics
    expected_eval: Optional[float] = None  # 期望评估值
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'fen': self.fen,
            'description': self.description,
            'best_moves': self.best_moves,
            'difficulty': self.difficulty,
            'category': self.category,
            'expected_eval': self.expected_eval
        }


class ELOCalculator:
    """
    ELO等级分计算器
    """
    
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        """
        初始化ELO计算器
        
        Args:
            k_factor: K因子，控制等级分变化幅度
            initial_rating: 初始等级分
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        计算期望得分
        
        Args:
            rating_a: 玩家A的等级分
            rating_b: 玩家B的等级分
            
        Returns:
            float: 玩家A的期望得分
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    def update_rating(
        self,
        rating: float,
        opponent_rating: float,
        score: float
    ) -> float:
        """
        更新等级分
        
        Args:
            rating: 当前等级分
            opponent_rating: 对手等级分
            score: 实际得分 (1=胜, 0.5=平, 0=负)
            
        Returns:
            float: 新的等级分
        """
        expected = self.expected_score(rating, opponent_rating)
        new_rating = rating + self.k_factor * (score - expected)
        return new_rating
    
    def calculate_rating_change(
        self,
        rating: float,
        opponent_rating: float,
        score: float
    ) -> float:
        """
        计算等级分变化
        
        Args:
            rating: 当前等级分
            opponent_rating: 对手等级分
            score: 实际得分
            
        Returns:
            float: 等级分变化量
        """
        expected = self.expected_score(rating, opponent_rating)
        return self.k_factor * (score - expected)
    
    def win_probability(self, rating_a: float, rating_b: float) -> float:
        """
        计算胜率
        
        Args:
            rating_a: 玩家A的等级分
            rating_b: 玩家B的等级分
            
        Returns:
            float: 玩家A的胜率
        """
        return self.expected_score(rating_a, rating_b)


class ModelEvaluator:
    """
    模型评估器
    
    实现模型性能评估、对弈测试和基准测试功能。
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        初始化评估器
        
        Args:
            config: 评估配置
        """
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(__name__)
        self.elo_calculator = ELOCalculator(
            k_factor=self.config.k_factor,
            initial_rating=self.config.initial_elo
        )
        
        # 模型等级分记录
        self.model_ratings: Dict[str, float] = {}
        
        # 基准测试局面
        self.benchmark_positions: List[BenchmarkPosition] = []
        self._load_default_positions()
    
    def _load_default_positions(self):
        """加载默认的基准测试局面"""
        # 这里添加一些经典的象棋测试局面
        default_positions = [
            BenchmarkPosition(
                fen="rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
                description="开局标准局面",
                best_moves=["h2e2", "b0c2", "h0g2"],
                difficulty=1,
                category="opening"
            ),
            BenchmarkPosition(
                fen="r1ba1a3/4kn3/2n1b4/p1p1p1p1p/4c4/6P2/P1P1P3P/1CcC5/4A4/2BAKAB2 w - - 0 1",
                description="中局复杂局面",
                best_moves=["c3c7", "e1f2"],
                difficulty=3,
                category="middlegame"
            ),
            BenchmarkPosition(
                fen="3k5/4a4/4ba3/9/9/9/9/4BA3/4A4/3K5 w - - 0 1",
                description="残局基础局面",
                best_moves=["d0e1", "e1f2"],
                difficulty=2,
                category="endgame"
            )
        ]
        
        self.benchmark_positions.extend(default_positions)
    
    def add_benchmark_position(self, position: BenchmarkPosition):
        """
        添加基准测试局面
        
        Args:
            position: 基准测试局面
        """
        self.benchmark_positions.append(position)
    
    def load_benchmark_positions(self, filepath: str):
        """
        从文件加载基准测试局面
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            positions_data = json.load(f)
        
        for pos_data in positions_data:
            position = BenchmarkPosition(**pos_data)
            self.benchmark_positions.append(position)
    
    def evaluate_against_baseline(
        self,
        new_model: ChessNet,
        baseline_model: ChessNet,
        new_model_name: str = "new_model",
        baseline_name: str = "baseline"
    ) -> EvaluationResult:
        """
        与基准模型对弈评估
        
        Args:
            new_model: 新模型
            baseline_model: 基准模型
            new_model_name: 新模型名称
            baseline_name: 基准模型名称
            
        Returns:
            EvaluationResult: 评估结果
        """
        self.logger.info(f"开始评估 {new_model_name} vs {baseline_name}")
        
        # 创建MCTS配置
        mcts_config = MCTSConfig(
            num_simulations=400,  # 评估时使用较少的模拟次数
            c_puct=1.0,
            temperature=0.1  # 低温度，更确定性的走法
        )
        
        # 统计结果
        wins = 0
        losses = 0
        draws = 0
        total_time = 0.0
        total_moves = 0
        
        # 进行对弈
        for game_idx in range(self.config.num_games):
            # 随机决定先手
            new_model_color = 1 if random.random() < 0.5 else -1
            baseline_color = -new_model_color
            
            game_start_time = time.time()
            result, move_count = self._play_game(
                new_model, baseline_model,
                new_model_color, baseline_color,
                mcts_config
            )
            game_time = time.time() - game_start_time
            
            # 统计结果
            if result == new_model_color:
                wins += 1
            elif result == baseline_color:
                losses += 1
            else:
                draws += 1
            
            total_time += game_time
            total_moves += move_count
            
            # 进度报告
            if (game_idx + 1) % 10 == 0:
                current_win_rate = wins / (game_idx + 1)
                self.logger.info(
                    f"进度: {game_idx + 1}/{self.config.num_games}, "
                    f"胜率: {current_win_rate:.2%}"
                )
        
        # 计算指标
        win_rate = wins / self.config.num_games
        score = (wins + 0.5 * draws) / self.config.num_games
        
        # 计算ELO变化
        new_model_rating = self.model_ratings.get(new_model_name, self.config.initial_elo)
        baseline_rating = self.model_ratings.get(baseline_name, self.config.initial_elo)
        elo_change = self.elo_calculator.calculate_rating_change(
            new_model_rating, baseline_rating, score
        )
        
        # 更新等级分
        self.model_ratings[new_model_name] = new_model_rating + elo_change
        
        # 创建评估结果
        result = EvaluationResult(
            model_name=new_model_name,
            opponent_name=baseline_name,
            wins=wins,
            losses=losses,
            draws=draws,
            total_games=self.config.num_games,
            win_rate=win_rate,
            elo_change=elo_change,
            average_game_length=total_moves / self.config.num_games,
            average_time_per_move=total_time / total_moves if total_moves > 0 else 0.0,
            metadata={
                'new_model_rating': self.model_ratings[new_model_name],
                'baseline_rating': baseline_rating,
                'score': score
            }
        )
        
        self.logger.info(
            f"评估完成: {new_model_name} vs {baseline_name} - "
            f"胜率: {win_rate:.2%}, ELO变化: {elo_change:+.1f}"
        )
        
        return result
    
    def _play_game(
        self,
        model1: ChessNet,
        model2: ChessNet,
        model1_color: int,
        model2_color: int,
        mcts_config: MCTSConfig
    ) -> Tuple[int, int]:
        """
        进行单局对弈
        
        Args:
            model1: 模型1
            model2: 模型2
            model1_color: 模型1的颜色
            model2_color: 模型2的颜色
            mcts_config: MCTS配置
            
        Returns:
            Tuple[int, int]: (获胜者, 走法数)
        """
        board = ChessBoard()
        move_count = 0
        
        # 创建搜索器
        searcher1 = MCTSSearcher(model1, mcts_config)
        searcher2 = MCTSSearcher(model2, mcts_config)
        
        while not board.is_game_over() and move_count < self.config.max_game_length:
            current_player = board.current_player
            
            # 选择对应的搜索器
            if current_player == model1_color:
                searcher = searcher1
            else:
                searcher = searcher2
            
            # 获取合法走法
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            # 执行搜索
            root_node = searcher.search(board, mcts_config.num_simulations)
            action_probs = searcher.get_action_probabilities(root_node, temperature=0.1)
            
            # 选择最佳走法
            if len(action_probs) > 0:
                best_move_idx = np.argmax(action_probs)
                if best_move_idx < len(legal_moves):
                    move = legal_moves[best_move_idx]
                else:
                    move = legal_moves[0]  # 备选方案
            else:
                move = legal_moves[0]  # 备选方案
            
            # 执行走法
            board = board.make_move(move)
            move_count += 1
        
        # 确定获胜者
        winner = board.get_winner() if board.is_game_over() else 0
        
        return winner, move_count
    
    def tournament_evaluation(
        self,
        models: Dict[str, ChessNet],
        num_games_per_pair: int = 20
    ) -> Dict[str, Any]:
        """
        锦标赛评估
        
        Args:
            models: 模型字典 {名称: 模型}
            num_games_per_pair: 每对模型的对弈局数
            
        Returns:
            Dict[str, Any]: 锦标赛结果
        """
        self.logger.info(f"开始锦标赛评估，{len(models)} 个模型")
        
        model_names = list(models.keys())
        results = {}
        
        # 初始化等级分
        for name in model_names:
            if name not in self.model_ratings:
                self.model_ratings[name] = self.config.initial_elo
        
        # 进行循环赛
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i >= j:  # 避免重复对弈
                    continue
                
                self.logger.info(f"对弈: {name1} vs {name2}")
                
                # 临时设置对弈局数
                original_num_games = self.config.num_games
                self.config.num_games = num_games_per_pair
                
                # 进行对弈
                result = self.evaluate_against_baseline(
                    models[name1], models[name2], name1, name2
                )
                
                # 恢复原设置
                self.config.num_games = original_num_games
                
                # 记录结果
                pair_key = f"{name1}_vs_{name2}"
                results[pair_key] = result
        
        # 计算排名
        rankings = self._calculate_rankings(results, model_names)
        
        tournament_result = {
            'models': model_names,
            'pairwise_results': {k: v.to_dict() for k, v in results.items()},
            'rankings': rankings,
            'final_ratings': self.model_ratings.copy()
        }
        
        self.logger.info("锦标赛评估完成")
        return tournament_result
    
    def _calculate_rankings(
        self,
        results: Dict[str, EvaluationResult],
        model_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        计算排名
        
        Args:
            results: 对弈结果
            model_names: 模型名称列表
            
        Returns:
            List[Dict[str, Any]]: 排名列表
        """
        # 统计每个模型的总得分
        scores = defaultdict(float)
        games_played = defaultdict(int)
        
        for result in results.values():
            model_name = result.model_name
            opponent_name = result.opponent_name
            
            # 模型得分
            model_score = result.wins + 0.5 * result.draws
            opponent_score = result.losses + 0.5 * result.draws
            
            scores[model_name] += model_score
            scores[opponent_name] += opponent_score
            
            games_played[model_name] += result.total_games
            games_played[opponent_name] += result.total_games
        
        # 计算胜率和排名
        rankings = []
        for name in model_names:
            total_score = scores[name]
            total_games = games_played[name]
            win_rate = total_score / total_games if total_games > 0 else 0.0
            
            rankings.append({
                'model_name': name,
                'rating': self.model_ratings[name],
                'total_score': total_score,
                'total_games': total_games,
                'win_rate': win_rate
            })
        
        # 按等级分排序
        rankings.sort(key=lambda x: x['rating'], reverse=True)
        
        # 添加排名
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def benchmark_performance(
        self,
        model: ChessNet,
        model_name: str = "model",
        positions: Optional[List[BenchmarkPosition]] = None
    ) -> Dict[str, Any]:
        """
        基准性能测试
        
        Args:
            model: 要测试的模型
            model_name: 模型名称
            positions: 测试局面列表
            
        Returns:
            Dict[str, Any]: 基准测试结果
        """
        if positions is None:
            positions = self.benchmark_positions
        
        self.logger.info(f"开始基准测试: {model_name}, {len(positions)} 个局面")
        
        # 创建MCTS搜索器
        mcts_config = MCTSConfig(
            num_simulations=800,
            c_puct=1.0,
            temperature=0.1
        )
        searcher = MCTSSearcher(model, mcts_config)
        
        results = []
        correct_predictions = 0
        total_time = 0.0
        
        for i, position in enumerate(positions):
            start_time = time.time()
            
            # 创建棋盘
            board = ChessBoard(position.fen)
            
            # 获取合法走法
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                continue
            
            # 执行搜索
            root_node = searcher.search(board, mcts_config.num_simulations)
            action_probs = searcher.get_action_probabilities(root_node, temperature=0.1)
            
            # 获取最佳走法
            if len(action_probs) > 0:
                best_move_idx = np.argmax(action_probs)
                if best_move_idx < len(legal_moves):
                    best_move = legal_moves[best_move_idx]
                    best_move_str = best_move.to_coordinate_notation()
                else:
                    best_move_str = ""
            else:
                best_move_str = ""
            
            # 获取评估值
            value_estimate = searcher.get_value_estimate(root_node)
            
            search_time = time.time() - start_time
            total_time += search_time
            
            # 检查是否正确
            is_correct = best_move_str in position.best_moves
            if is_correct:
                correct_predictions += 1
            
            # 记录结果
            result = {
                'position_index': i,
                'fen': position.fen,
                'description': position.description,
                'category': position.category,
                'difficulty': position.difficulty,
                'best_moves': position.best_moves,
                'predicted_move': best_move_str,
                'evaluation': value_estimate,
                'expected_eval': position.expected_eval,
                'is_correct': is_correct,
                'search_time': search_time
            }
            results.append(result)
            
            # 进度报告
            if (i + 1) % 10 == 0:
                accuracy = correct_predictions / (i + 1)
                self.logger.info(
                    f"进度: {i + 1}/{len(positions)}, "
                    f"准确率: {accuracy:.2%}"
                )
        
        # 计算总体统计
        total_positions = len(results)
        overall_accuracy = correct_predictions / total_positions if total_positions > 0 else 0.0
        average_time = total_time / total_positions if total_positions > 0 else 0.0
        
        # 按类别统计
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        difficulty_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for result in results:
            category = result['category']
            difficulty = result['difficulty']
            is_correct = result['is_correct']
            
            category_stats[category]['total'] += 1
            difficulty_stats[difficulty]['total'] += 1
            
            if is_correct:
                category_stats[category]['correct'] += 1
                difficulty_stats[difficulty]['correct'] += 1
        
        # 计算各类别准确率
        for category in category_stats:
            stats = category_stats[category]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        for difficulty in difficulty_stats:
            stats = difficulty_stats[difficulty]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        benchmark_result = {
            'model_name': model_name,
            'total_positions': total_positions,
            'correct_predictions': correct_predictions,
            'overall_accuracy': overall_accuracy,
            'average_search_time': average_time,
            'category_stats': dict(category_stats),
            'difficulty_stats': dict(difficulty_stats),
            'detailed_results': results
        }
        
        self.logger.info(
            f"基准测试完成: {model_name} - "
            f"准确率: {overall_accuracy:.2%}, "
            f"平均时间: {average_time:.2f}s"
        )
        
        return benchmark_result
    
    def calculate_elo_rating(
        self,
        game_results: List[GameResult]
    ) -> Dict[str, float]:
        """
        根据游戏结果计算ELO等级分
        
        Args:
            game_results: 游戏结果列表
            
        Returns:
            Dict[str, float]: 模型等级分字典
        """
        # 统计对弈结果
        matchups = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
        
        for result in game_results:
            model1 = result.metadata.get('model1_name', 'model1')
            model2 = result.metadata.get('model2_name', 'model2')
            winner = result.winner
            
            if winner == 1:  # model1获胜
                matchups[(model1, model2)]['wins'] += 1
                matchups[(model2, model1)]['losses'] += 1
            elif winner == -1:  # model2获胜
                matchups[(model1, model2)]['losses'] += 1
                matchups[(model2, model1)]['wins'] += 1
            else:  # 平局
                matchups[(model1, model2)]['draws'] += 1
                matchups[(model2, model1)]['draws'] += 1
        
        # 获取所有模型名称
        all_models = set()
        for (model1, model2) in matchups.keys():
            all_models.add(model1)
            all_models.add(model2)
        
        # 初始化等级分
        ratings = {}
        for model in all_models:
            ratings[model] = self.model_ratings.get(model, self.config.initial_elo)
        
        # 迭代更新等级分
        for _ in range(10):  # 多次迭代以收敛
            new_ratings = ratings.copy()
            
            for (model1, model2), stats in matchups.items():
                total_games = stats['wins'] + stats['losses'] + stats['draws']
                if total_games == 0:
                    continue
                
                score = (stats['wins'] + 0.5 * stats['draws']) / total_games
                
                rating_change = self.elo_calculator.calculate_rating_change(
                    ratings[model1], ratings[model2], score
                )
                
                new_ratings[model1] += rating_change / len(matchups)  # 平均分配变化
            
            ratings = new_ratings
        
        # 更新内部记录
        self.model_ratings.update(ratings)
        
        return ratings
    
    def save_evaluation_report(
        self,
        results: Dict[str, Any],
        filepath: str
    ):
        """
        保存评估报告
        
        Args:
            results: 评估结果
            filepath: 保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"评估报告已保存到: {filepath}")
    
    def get_model_rating(self, model_name: str) -> float:
        """
        获取模型等级分
        
        Args:
            model_name: 模型名称
            
        Returns:
            float: 模型等级分
        """
        return self.model_ratings.get(model_name, self.config.initial_elo)
    
    def set_model_rating(self, model_name: str, rating: float):
        """
        设置模型等级分
        
        Args:
            model_name: 模型名称
            rating: 等级分
        """
        self.model_ratings[model_name] = rating
    
    def get_all_ratings(self) -> Dict[str, float]:
        """
        获取所有模型等级分
        
        Returns:
            Dict[str, float]: 所有模型等级分
        """
        return self.model_ratings.copy()