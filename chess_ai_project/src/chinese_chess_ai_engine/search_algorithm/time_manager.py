"""
搜索时间管理器

实现智能的搜索时间分配和控制机制。
"""

import time
import threading
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
import logging


@dataclass
class TimeAllocation:
    """时间分配数据结构"""
    total_time: float  # 总时间
    move_time: float   # 单步时间
    increment: float   # 增量时间
    moves_to_go: Optional[int] = None  # 剩余步数
    emergency_time: float = 0.1  # 紧急时间保留


class TimeManager:
    """
    搜索时间管理器
    
    智能分配搜索时间，支持动态调整和紧急时间控制。
    """
    
    def __init__(self):
        """初始化时间管理器"""
        self.logger = logging.getLogger(__name__)
        
        # 时间统计
        self.move_times: List[float] = []
        self.search_depths: List[int] = []
        self.nodes_per_second: List[float] = []
        
        # 当前搜索状态
        self.current_search_start: Optional[float] = None
        self.current_allocated_time: Optional[float] = None
        self.is_searching = False
        
        # 线程安全
        self._lock = threading.Lock()
        
        # 回调函数
        self.time_warning_callback: Optional[Callable] = None
        self.time_up_callback: Optional[Callable] = None
    
    def allocate_time(
        self,
        time_allocation: TimeAllocation,
        game_phase: str = 'middle',
        position_complexity: float = 1.0,
        time_pressure: float = 1.0
    ) -> float:
        """
        分配搜索时间
        
        Args:
            time_allocation: 时间分配信息
            game_phase: 游戏阶段 ('opening', 'middle', 'endgame')
            position_complexity: 位置复杂度 (0.5-2.0)
            time_pressure: 时间压力 (0.5-2.0)
            
        Returns:
            float: 分配的搜索时间（秒）
        """
        with self._lock:
            # 基础时间计算
            if time_allocation.moves_to_go:
                # 有剩余步数限制
                base_time = (time_allocation.total_time - time_allocation.emergency_time) / time_allocation.moves_to_go
                base_time += time_allocation.increment
            else:
                # 无限制模式，使用固定时间
                base_time = time_allocation.move_time
            
            # 根据游戏阶段调整
            phase_multiplier = self._get_phase_multiplier(game_phase)
            
            # 根据位置复杂度调整
            complexity_multiplier = max(0.5, min(2.0, position_complexity))
            
            # 根据时间压力调整
            pressure_multiplier = max(0.5, min(2.0, time_pressure))
            
            # 根据历史表现调整
            history_multiplier = self._get_history_multiplier()
            
            # 计算最终时间
            allocated_time = (base_time * 
                            phase_multiplier * 
                            complexity_multiplier * 
                            pressure_multiplier * 
                            history_multiplier)
            
            # 确保不超过可用时间
            max_safe_time = time_allocation.total_time - time_allocation.emergency_time
            allocated_time = min(allocated_time, max_safe_time)
            
            # 确保最小时间
            allocated_time = max(allocated_time, 0.1)
            
            self.logger.info(
                f"时间分配: {allocated_time:.2f}s "
                f"(基础:{base_time:.2f}s, 阶段:{phase_multiplier:.2f}, "
                f"复杂度:{complexity_multiplier:.2f}, 压力:{pressure_multiplier:.2f}, "
                f"历史:{history_multiplier:.2f})"
            )
            
            return allocated_time
    
    def _get_phase_multiplier(self, game_phase: str) -> float:
        """
        获取游戏阶段时间倍数
        
        Args:
            game_phase: 游戏阶段
            
        Returns:
            float: 时间倍数
        """
        multipliers = {
            'opening': 0.8,   # 开局用时较少
            'middle': 1.2,    # 中局用时较多
            'endgame': 1.0    # 残局正常用时
        }
        return multipliers.get(game_phase, 1.0)
    
    def _get_history_multiplier(self) -> float:
        """
        根据历史表现获取时间倍数
        
        Returns:
            float: 历史倍数
        """
        if len(self.move_times) < 3:
            return 1.0
        
        # 计算最近几步的平均用时
        recent_times = self.move_times[-5:]
        avg_time = sum(recent_times) / len(recent_times)
        
        # 计算平均搜索深度
        if self.search_depths:
            recent_depths = self.search_depths[-5:]
            avg_depth = sum(recent_depths) / len(recent_depths)
            
            # 如果搜索深度较浅，可能需要更多时间
            if avg_depth < 8:
                return 1.2
            elif avg_depth > 15:
                return 0.9
        
        # 如果最近用时过短，可能需要更多时间
        if avg_time < 1.0:
            return 1.1
        elif avg_time > 5.0:
            return 0.9
        
        return 1.0
    
    def start_search(self, allocated_time: float) -> 'SearchTimer':
        """
        开始搜索计时
        
        Args:
            allocated_time: 分配的时间
            
        Returns:
            SearchTimer: 搜索计时器
        """
        with self._lock:
            self.current_search_start = time.time()
            self.current_allocated_time = allocated_time
            self.is_searching = True
        
        return SearchTimer(self, allocated_time)
    
    def stop_search(self, nodes_searched: int = 0, max_depth: int = 0):
        """
        停止搜索计时
        
        Args:
            nodes_searched: 搜索的节点数
            max_depth: 最大搜索深度
        """
        with self._lock:
            if not self.is_searching or self.current_search_start is None:
                return
            
            # 计算用时
            elapsed_time = time.time() - self.current_search_start
            
            # 记录统计信息
            self.move_times.append(elapsed_time)
            if max_depth > 0:
                self.search_depths.append(max_depth)
            
            if nodes_searched > 0 and elapsed_time > 0:
                nps = nodes_searched / elapsed_time
                self.nodes_per_second.append(nps)
            
            # 限制历史记录长度
            if len(self.move_times) > 50:
                self.move_times = self.move_times[-30:]
            if len(self.search_depths) > 50:
                self.search_depths = self.search_depths[-30:]
            if len(self.nodes_per_second) > 50:
                self.nodes_per_second = self.nodes_per_second[-30:]
            
            # 重置状态
            self.is_searching = False
            self.current_search_start = None
            self.current_allocated_time = None
            
            self.logger.info(
                f"搜索完成: 用时{elapsed_time:.2f}s, "
                f"节点{nodes_searched}, 深度{max_depth}, "
                f"NPS{nps:.0f}" if nodes_searched > 0 else f"搜索完成: 用时{elapsed_time:.2f}s"
            )
    
    def get_remaining_time(self) -> Optional[float]:
        """
        获取剩余搜索时间
        
        Returns:
            Optional[float]: 剩余时间，如果未在搜索则返回None
        """
        with self._lock:
            if not self.is_searching or self.current_search_start is None or self.current_allocated_time is None:
                return None
            
            elapsed = time.time() - self.current_search_start
            remaining = self.current_allocated_time - elapsed
            return max(0.0, remaining)
    
    def is_time_up(self, safety_margin: float = 0.1) -> bool:
        """
        检查时间是否用完
        
        Args:
            safety_margin: 安全边界时间
            
        Returns:
            bool: 是否时间用完
        """
        remaining = self.get_remaining_time()
        if remaining is None:
            return False
        
        return remaining <= safety_margin
    
    def should_stop_search(self, nodes_searched: int = 0, depth_reached: int = 0) -> bool:
        """
        判断是否应该停止搜索
        
        Args:
            nodes_searched: 已搜索节点数
            depth_reached: 已达到深度
            
        Returns:
            bool: 是否应该停止
        """
        # 时间检查
        if self.is_time_up():
            return True
        
        # 深度检查（如果搜索很深，可以提前停止）
        if depth_reached > 20:
            remaining = self.get_remaining_time()
            if remaining and remaining < 0.5:
                return True
        
        # 节点数检查（如果搜索节点很多但时间不多，可以停止）
        if nodes_searched > 100000:
            remaining = self.get_remaining_time()
            if remaining and remaining < 1.0:
                return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """
        获取时间管理统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            stats = {
                'total_moves': len(self.move_times),
                'is_searching': self.is_searching,
                'current_remaining_time': self.get_remaining_time()
            }
            
            if self.move_times:
                stats.update({
                    'avg_move_time': sum(self.move_times) / len(self.move_times),
                    'min_move_time': min(self.move_times),
                    'max_move_time': max(self.move_times),
                    'total_search_time': sum(self.move_times)
                })
            
            if self.search_depths:
                stats.update({
                    'avg_search_depth': sum(self.search_depths) / len(self.search_depths),
                    'min_search_depth': min(self.search_depths),
                    'max_search_depth': max(self.search_depths)
                })
            
            if self.nodes_per_second:
                stats.update({
                    'avg_nodes_per_second': sum(self.nodes_per_second) / len(self.nodes_per_second),
                    'min_nodes_per_second': min(self.nodes_per_second),
                    'max_nodes_per_second': max(self.nodes_per_second)
                })
            
            return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        with self._lock:
            self.move_times.clear()
            self.search_depths.clear()
            self.nodes_per_second.clear()
    
    def set_callbacks(
        self,
        time_warning_callback: Optional[Callable] = None,
        time_up_callback: Optional[Callable] = None
    ):
        """
        设置时间回调函数
        
        Args:
            time_warning_callback: 时间警告回调
            time_up_callback: 时间用完回调
        """
        self.time_warning_callback = time_warning_callback
        self.time_up_callback = time_up_callback


class SearchTimer:
    """
    搜索计时器
    
    用于监控单次搜索的时间使用。
    """
    
    def __init__(self, time_manager: TimeManager, allocated_time: float):
        """
        初始化搜索计时器
        
        Args:
            time_manager: 时间管理器
            allocated_time: 分配的时间
        """
        self.time_manager = time_manager
        self.allocated_time = allocated_time
        self.start_time = time.time()
        self.warning_sent = False
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_time, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_time(self):
        """监控时间的后台线程"""
        warning_threshold = self.allocated_time * 0.8  # 80%时发出警告
        
        while self.time_manager.is_searching:
            elapsed = time.time() - self.start_time
            
            # 发送时间警告
            if not self.warning_sent and elapsed >= warning_threshold:
                self.warning_sent = True
                if self.time_manager.time_warning_callback:
                    self.time_manager.time_warning_callback(self.allocated_time - elapsed)
            
            # 检查时间是否用完
            if elapsed >= self.allocated_time:
                if self.time_manager.time_up_callback:
                    self.time_manager.time_up_callback()
                break
            
            time.sleep(0.1)  # 每100ms检查一次
    
    def get_elapsed_time(self) -> float:
        """获取已用时间"""
        return time.time() - self.start_time
    
    def get_remaining_time(self) -> float:
        """获取剩余时间"""
        elapsed = self.get_elapsed_time()
        return max(0.0, self.allocated_time - elapsed)
    
    def is_time_up(self, safety_margin: float = 0.1) -> bool:
        """检查时间是否用完"""
        return self.get_remaining_time() <= safety_margin
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        # 计时器会在时间管理器的stop_search中被清理
        pass


class AdaptiveTimeManager(TimeManager):
    """
    自适应时间管理器
    
    根据搜索表现动态调整时间分配策略。
    """
    
    def __init__(self):
        """初始化自适应时间管理器"""
        super().__init__()
        
        # 自适应参数
        self.performance_history: List[Dict] = []
        self.adaptation_rate = 0.1  # 适应速度
        
        # 性能指标权重
        self.time_efficiency_weight = 0.4
        self.search_quality_weight = 0.6
    
    def record_search_performance(
        self,
        time_used: float,
        time_allocated: float,
        nodes_searched: int,
        depth_reached: int,
        move_quality: Optional[float] = None
    ):
        """
        记录搜索性能
        
        Args:
            time_used: 实际用时
            time_allocated: 分配时间
            nodes_searched: 搜索节点数
            depth_reached: 搜索深度
            move_quality: 走法质量评分 (0-1)
        """
        performance = {
            'time_used': time_used,
            'time_allocated': time_allocated,
            'time_efficiency': time_used / time_allocated if time_allocated > 0 else 1.0,
            'nodes_searched': nodes_searched,
            'depth_reached': depth_reached,
            'nodes_per_second': nodes_searched / time_used if time_used > 0 else 0,
            'move_quality': move_quality,
            'timestamp': time.time()
        }
        
        self.performance_history.append(performance)
        
        # 限制历史记录长度
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
    
    def get_adaptive_multiplier(self) -> float:
        """
        获取自适应时间倍数
        
        Returns:
            float: 时间调整倍数
        """
        if len(self.performance_history) < 5:
            return 1.0
        
        recent_performance = self.performance_history[-10:]
        
        # 计算时间效率
        avg_time_efficiency = sum(p['time_efficiency'] for p in recent_performance) / len(recent_performance)
        
        # 计算搜索质量
        avg_depth = sum(p['depth_reached'] for p in recent_performance) / len(recent_performance)
        avg_nps = sum(p['nodes_per_second'] for p in recent_performance) / len(recent_performance)
        
        # 如果有走法质量数据，也考虑进去
        quality_scores = [p['move_quality'] for p in recent_performance if p['move_quality'] is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # 计算调整倍数
        time_factor = 1.0
        
        # 如果时间利用率低，减少时间分配
        if avg_time_efficiency < 0.7:
            time_factor *= 0.9
        elif avg_time_efficiency > 0.95:
            time_factor *= 1.1
        
        # 如果搜索深度浅，增加时间分配
        if avg_depth < 8:
            time_factor *= 1.1
        elif avg_depth > 15:
            time_factor *= 0.95
        
        # 如果走法质量低，增加时间分配
        if avg_quality < 0.4:
            time_factor *= 1.15
        elif avg_quality > 0.8:
            time_factor *= 0.95
        
        # 限制调整范围
        time_factor = max(0.5, min(2.0, time_factor))
        
        return time_factor
    
    def allocate_time(
        self,
        time_allocation: TimeAllocation,
        game_phase: str = 'middle',
        position_complexity: float = 1.0,
        time_pressure: float = 1.0
    ) -> float:
        """
        自适应时间分配
        
        Args:
            time_allocation: 时间分配信息
            game_phase: 游戏阶段
            position_complexity: 位置复杂度
            time_pressure: 时间压力
            
        Returns:
            float: 分配的搜索时间
        """
        # 获取基础时间分配
        base_time = super().allocate_time(time_allocation, game_phase, position_complexity, time_pressure)
        
        # 应用自适应调整
        adaptive_multiplier = self.get_adaptive_multiplier()
        adapted_time = base_time * adaptive_multiplier
        
        # 确保在合理范围内
        max_safe_time = time_allocation.total_time - time_allocation.emergency_time
        adapted_time = min(adapted_time, max_safe_time)
        adapted_time = max(adapted_time, 0.1)
        
        self.logger.info(f"自适应时间调整: {base_time:.2f}s -> {adapted_time:.2f}s (倍数: {adaptive_multiplier:.2f})")
        
        return adapted_time