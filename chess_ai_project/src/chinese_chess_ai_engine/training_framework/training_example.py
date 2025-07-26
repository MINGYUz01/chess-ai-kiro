"""
训练样本数据结构

定义强化学习训练中使用的数据样本格式。
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pickle
import json
from pathlib import Path

from ..rules_engine import ChessBoard, Move


@dataclass
class TrainingExample:
    """
    训练样本类
    
    包含一个训练样本的所有信息，用于神经网络训练。
    """
    
    # 棋盘状态的张量表示 (20通道, 10行, 9列)
    board_tensor: torch.Tensor
    
    # 策略目标 (所有可能走法的概率分布)
    policy_target: np.ndarray
    
    # 价值目标 (从当前玩家角度的胜负评估)
    value_target: float
    
    # 游戏最终结果 (1: 红方胜, -1: 黑方胜, 0: 平局)
    game_result: int
    
    # 走法编号 (在游戏中的第几步)
    move_number: int
    
    # 当前玩家 (1: 红方, -1: 黑方)
    current_player: int
    
    # 原始棋盘状态 (用于调试和验证)
    original_board: Optional[ChessBoard] = None
    
    # 实际执行的走法 (用于分析)
    actual_move: Optional[Move] = None
    
    # 元数据
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.metadata is None:
            self.metadata = {}
        
        # 验证数据维度
        self._validate_dimensions()
    
    def _validate_dimensions(self):
        """验证数据维度的正确性"""
        # 验证棋盘张量维度
        if self.board_tensor.shape != (20, 10, 9):
            raise ValueError(f"棋盘张量维度错误: 期望(20, 10, 9), 实际{self.board_tensor.shape}")
        
        # 验证策略目标维度
        if self.policy_target.shape != (8100,):
            raise ValueError(f"策略目标维度错误: 期望(8100,), 实际{self.policy_target.shape}")
        
        # 验证价值目标范围
        if not -1.0 <= self.value_target <= 1.0:
            raise ValueError(f"价值目标超出范围[-1, 1]: {self.value_target}")
        
        # 验证游戏结果
        if self.game_result not in [-1, 0, 1]:
            raise ValueError(f"游戏结果必须是-1, 0, 1之一: {self.game_result}")
        
        # 验证当前玩家
        if self.current_player not in [-1, 1]:
            raise ValueError(f"当前玩家必须是-1或1: {self.current_player}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        return {
            'board_tensor': self.board_tensor.numpy().tolist(),  # 转换为列表以支持JSON序列化
            'policy_target': self.policy_target.tolist(),  # 转换为列表以支持JSON序列化
            'value_target': self.value_target,
            'game_result': self.game_result,
            'move_number': self.move_number,
            'current_player': self.current_player,
            'actual_move': {
                'from_pos': self.actual_move.from_pos,
                'to_pos': self.actual_move.to_pos,
                'piece': self.actual_move.piece
            } if self.actual_move else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        """
        从字典创建训练样本
        
        Args:
            data: 字典数据
            
        Returns:
            TrainingExample: 训练样本对象
        """
        # 重建Move对象
        actual_move = None
        if data.get('actual_move'):
            move_data = data['actual_move']
            actual_move = Move(
                from_pos=tuple(move_data['from_pos']),
                to_pos=tuple(move_data['to_pos']),
                piece=move_data['piece']
            )
        
        # 处理numpy数组的重建
        board_tensor = data['board_tensor']
        if isinstance(board_tensor, list):
            board_tensor = np.array(board_tensor)
        board_tensor = torch.from_numpy(board_tensor.astype(np.float32))
        
        policy_target = data['policy_target']
        if isinstance(policy_target, list):
            policy_target = np.array(policy_target, dtype=np.float32)
        
        return cls(
            board_tensor=board_tensor,
            policy_target=policy_target,
            value_target=data['value_target'],
            game_result=data['game_result'],
            move_number=data['move_number'],
            current_player=data['current_player'],
            actual_move=actual_move,
            metadata=data.get('metadata', {})
        )
    
    def save_to_file(self, filepath: str, format: str = 'pickle'):
        """
        保存到文件
        
        Args:
            filepath: 文件路径
            format: 保存格式 ('pickle', 'json')
        """
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @classmethod
    def load_from_file(cls, filepath: str, format: str = 'auto') -> 'TrainingExample':
        """
        从文件加载
        
        Args:
            filepath: 文件路径
            format: 文件格式 ('pickle', 'json', 'auto')
            
        Returns:
            TrainingExample: 训练样本对象
        """
        if format == 'auto':
            # 根据文件扩展名判断格式
            if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            elif filepath.endswith('.json'):
                format = 'json'
            else:
                format = 'pickle'  # 默认使用pickle
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return cls.from_dict(data)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def get_policy_entropy(self) -> float:
        """
        计算策略分布的熵
        
        Returns:
            float: 策略熵值
        """
        # 避免log(0)
        policy = self.policy_target + 1e-8
        policy = policy / policy.sum()  # 归一化
        
        entropy = -np.sum(policy * np.log(policy))
        return entropy
    
    def get_top_policy_moves(self, k: int = 5) -> List[tuple]:
        """
        获取策略概率最高的前k个走法
        
        Args:
            k: 返回的走法数量
            
        Returns:
            List[tuple]: [(索引, 概率), ...]
        """
        top_indices = np.argsort(self.policy_target)[-k:][::-1]
        return [(idx, self.policy_target[idx]) for idx in top_indices]
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"TrainingExample(move={self.move_number}, "
                f"player={self.current_player}, "
                f"value={self.value_target:.3f}, "
                f"result={self.game_result})")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


class TrainingDataset:
    """
    训练数据集类
    
    管理多个训练样本的集合。
    """
    
    def __init__(self, examples: Optional[List[TrainingExample]] = None):
        """
        初始化数据集
        
        Args:
            examples: 训练样本列表
        """
        self.examples = examples or []
        self.metadata = {
            'created_at': None,
            'total_games': 0,
            'total_moves': 0,
            'win_rate': {'red': 0.0, 'black': 0.0, 'draw': 0.0}
        }
        # 如果有初始样本，更新元数据
        if self.examples:
            self._update_metadata()
    
    def add_example(self, example: TrainingExample):
        """
        添加训练样本
        
        Args:
            example: 训练样本
        """
        self.examples.append(example)
        self._update_metadata()
    
    def add_examples(self, examples: List[TrainingExample]):
        """
        批量添加训练样本
        
        Args:
            examples: 训练样本列表
        """
        self.examples.extend(examples)
        self._update_metadata()
    
    def _update_metadata(self):
        """更新元数据统计"""
        if not self.examples:
            return
        
        # 统计游戏数量
        game_ids = set()
        results = []
        
        for example in self.examples:
            game_id = example.metadata.get('game_id', 0)
            game_ids.add(game_id)
            
            # 只统计游戏结束时的结果
            if example.metadata.get('is_final_position', False):
                results.append(example.game_result)
        
        self.metadata['total_games'] = len(game_ids)
        self.metadata['total_moves'] = len(self.examples)
        
        # 计算胜率
        if results:
            red_wins = sum(1 for r in results if r == 1)
            black_wins = sum(1 for r in results if r == -1)
            draws = sum(1 for r in results if r == 0)
            total = len(results)
            
            self.metadata['win_rate'] = {
                'red': red_wins / total,
                'black': black_wins / total,
                'draw': draws / total
            }
    
    def shuffle(self):
        """随机打乱数据集"""
        np.random.shuffle(self.examples)
    
    def split(self, train_ratio: float = 0.8) -> tuple:
        """
        分割数据集
        
        Args:
            train_ratio: 训练集比例
            
        Returns:
            tuple: (训练集, 验证集)
        """
        split_idx = int(len(self.examples) * train_ratio)
        
        train_examples = self.examples[:split_idx]
        val_examples = self.examples[split_idx:]
        
        train_dataset = TrainingDataset(train_examples)
        val_dataset = TrainingDataset(val_examples)
        
        return train_dataset, val_dataset
    
    def get_batch(self, batch_size: int, start_idx: int = 0) -> List[TrainingExample]:
        """
        获取批次数据
        
        Args:
            batch_size: 批次大小
            start_idx: 起始索引
            
        Returns:
            List[TrainingExample]: 批次样本
        """
        end_idx = min(start_idx + batch_size, len(self.examples))
        return self.examples[start_idx:end_idx]
    
    def save_to_file(self, filepath: str, format: str = 'pickle'):
        """
        保存数据集到文件
        
        Args:
            filepath: 文件路径
            format: 保存格式
        """
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        elif format == 'json':
            data = {
                'examples': [example.to_dict() for example in self.examples],
                'metadata': self.metadata
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @classmethod
    def load_from_file(cls, filepath: str, format: str = 'auto') -> 'TrainingDataset':
        """
        从文件加载数据集
        
        Args:
            filepath: 文件路径
            format: 文件格式
            
        Returns:
            TrainingDataset: 数据集对象
        """
        if format == 'auto':
            if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            elif filepath.endswith('.json'):
                format = 'json'
            else:
                format = 'pickle'
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                examples = [TrainingExample.from_dict(ex_data) for ex_data in data['examples']]
                dataset = cls(examples)
                dataset.metadata = data.get('metadata', {})
                return dataset
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not self.examples:
            return {'total_examples': 0}
        
        # 基本统计
        stats = {
            'total_examples': len(self.examples),
            'total_games': self.metadata['total_games'],
            'win_rate': self.metadata['win_rate']
        }
        
        # 价值分布统计
        values = [ex.value_target for ex in self.examples]
        stats['value_stats'] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        
        # 走法编号分布
        move_numbers = [ex.move_number for ex in self.examples]
        stats['move_stats'] = {
            'mean_game_length': np.mean(move_numbers),
            'max_game_length': np.max(move_numbers),
            'min_game_length': np.min(move_numbers)
        }
        
        # 策略熵分布
        entropies = [ex.get_policy_entropy() for ex in self.examples]
        stats['policy_entropy'] = {
            'mean': np.mean(entropies),
            'std': np.std(entropies),
            'min': np.min(entropies),
            'max': np.max(entropies)
        }
        
        return stats
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.examples)
    
    def __getitem__(self, index: int) -> TrainingExample:
        """获取指定索引的样本"""
        return self.examples[index]
    
    def __iter__(self):
        """迭代器"""
        return iter(self.examples)