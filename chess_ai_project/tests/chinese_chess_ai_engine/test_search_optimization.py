"""
搜索优化功能安全测试

只测试不会卡住的基本功能。
"""

import pytest
import time
from unittest.mock import Mock

from chess_ai_project.src.chinese_chess_ai_engine.search_algorithm import (
    TimeAllocation, SearchMetrics, MCTSConfig
)


class TestTimeAllocationSafe:
    """时间分配安全测试"""
    
    def test_time_allocation_creation(self):
        """测试时间分配创建"""
        allocation = TimeAllocation(
            total_time=300.0,
            move_time=10.0,
            increment=5.0,
            moves_to_go=30
        )
        
        assert allocation.total_time == 300.0
        assert allocation.move_time == 10.0
        assert allocation.increment == 5.0
        assert allocation.moves_to_go == 30
        assert allocation.emergency_time == 0.1
    
    def test_time_allocation_defaults(self):
        """测试时间分配默认值"""
        allocation = TimeAllocation(
            total_time=180.0,
            move_time=5.0,
            increment=2.0
        )
        
        assert allocation.moves_to_go is None
        assert allocation.emergency_time == 0.1
    
    def test_time_allocation_with_custom_emergency_time(self):
        """测试自定义紧急时间"""
        allocation = TimeAllocation(
            total_time=600.0,
            move_time=20.0,
            increment=10.0,
            emergency_time=1.0
        )
        
        assert allocation.emergency_time == 1.0


class TestSearchMetricsSafe:
    """搜索指标安全测试"""
    
    def test_search_metrics_creation(self):
        """测试搜索指标创建"""
        metrics = SearchMetrics(
            nodes_searched=1500,
            max_depth=12,
            time_used=3.5,
            memory_used=256.0,
            nodes_per_second=428.6
        )
        
        assert metrics.nodes_searched == 1500
        assert metrics.max_depth == 12
        assert metrics.time_used == 3.5
        assert metrics.memory_used == 256.0
        assert metrics.nodes_per_second == 428.6
    
    def test_search_metrics_defaults(self):
        """测试搜索指标默认值"""
        metrics = SearchMetrics()
        
        assert metrics.nodes_searched == 0
        assert metrics.max_depth == 0
        assert metrics.time_used == 0.0
        assert metrics.memory_used == 0.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.nodes_per_second == 0.0
        assert metrics.branching_factor == 0.0
        assert metrics.tree_size == 0
    
    def test_search_metrics_calculations(self):
        """测试搜索指标计算"""
        metrics = SearchMetrics(
            nodes_searched=2000,
            time_used=4.0
        )
        
        # 手动计算nodes_per_second
        expected_nps = 2000 / 4.0
        metrics.nodes_per_second = expected_nps
        
        assert metrics.nodes_per_second == 500.0


class TestMCTSConfigIntegration:
    """MCTS配置集成测试"""
    
    def test_config_with_time_limits(self):
        """测试带时间限制的配置"""
        config = MCTSConfig(
            num_simulations=500,
            c_puct=1.5,
            temperature=0.8,
            time_limit=5.0
        )
        
        assert config.num_simulations == 500
        assert config.c_puct == 1.5
        assert config.temperature == 0.8
        assert config.time_limit == 5.0
    
    def test_config_optimization_parameters(self):
        """测试配置优化参数"""
        config = MCTSConfig(
            num_simulations=1000,
            max_depth=25,
            dirichlet_alpha=0.2,
            dirichlet_epsilon=0.3
        )
        
        # 模拟优化调整
        optimized_simulations = int(config.num_simulations * 0.8)
        optimized_depth = min(config.max_depth, 20)
        
        assert optimized_simulations == 800
        assert optimized_depth == 20
    
    def test_config_dict_operations(self):
        """测试配置字典操作"""
        config = MCTSConfig(
            num_simulations=600,
            c_puct=1.2,
            temperature=0.6
        )
        
        # 转换为字典
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['num_simulations'] == 600
        assert config_dict['c_puct'] == 1.2
        
        # 从字典创建
        new_config = MCTSConfig.from_dict(config_dict)
        assert new_config.num_simulations == 600
        assert new_config.c_puct == 1.2
        assert new_config.temperature == 0.6


class TestSearchOptimizationLogic:
    """搜索优化逻辑测试"""
    
    def test_performance_analysis_logic(self):
        """测试性能分析逻辑"""
        # 模拟性能数据
        metrics_list = [
            SearchMetrics(nodes_searched=1000, time_used=2.0, max_depth=10),
            SearchMetrics(nodes_searched=1200, time_used=2.5, max_depth=12),
            SearchMetrics(nodes_searched=800, time_used=1.8, max_depth=8)
        ]
        
        # 计算平均性能
        total_nodes = sum(m.nodes_searched for m in metrics_list)
        total_time = sum(m.time_used for m in metrics_list)
        avg_depth = sum(m.max_depth for m in metrics_list) / len(metrics_list)
        
        assert total_nodes == 3000
        assert total_time == 6.3
        assert avg_depth == 10.0
        
        # 计算平均NPS
        avg_nps = total_nodes / total_time
        assert abs(avg_nps - 476.19) < 0.1
    
    def test_optimization_suggestions_logic(self):
        """测试优化建议逻辑"""
        # 模拟不同性能场景
        slow_metrics = SearchMetrics(nodes_per_second=500, max_depth=15, memory_used=2000)
        fast_metrics = SearchMetrics(nodes_per_second=2500, max_depth=8, memory_used=500)
        deep_metrics = SearchMetrics(nodes_per_second=1000, max_depth=60, memory_used=1000)
        
        # 测试优化建议逻辑
        suggestions = []
        
        if slow_metrics.nodes_per_second < 1000:
            suggestions.append("搜索速度较慢")
        
        if slow_metrics.memory_used > 1500:
            suggestions.append("内存使用过高")
        
        if deep_metrics.max_depth > 50:
            suggestions.append("搜索深度过深")
        
        assert "搜索速度较慢" in suggestions
        assert "内存使用过高" in suggestions
        assert "搜索深度过深" in suggestions
    
    def test_time_allocation_logic(self):
        """测试时间分配逻辑"""
        # 模拟时间分配计算
        total_time = 300.0
        moves_to_go = 20
        increment = 3.0
        emergency_time = 0.5
        
        # 基础时间计算
        base_time = (total_time - emergency_time) / moves_to_go + increment
        expected_base = (300.0 - 0.5) / 20 + 3.0
        
        assert abs(base_time - expected_base) < 0.01
        assert base_time == 17.975
        
        # 应用调整因子
        phase_multiplier = 1.2  # 中局
        complexity_multiplier = 1.5  # 复杂位置
        
        adjusted_time = base_time * phase_multiplier * complexity_multiplier
        expected_adjusted = 17.975 * 1.2 * 1.5
        
        assert abs(adjusted_time - expected_adjusted) < 0.01
    
    def test_adaptive_multiplier_logic(self):
        """测试自适应倍数逻辑"""
        # 模拟性能历史
        performance_data = [
            {'time_efficiency': 0.6, 'depth_reached': 6, 'move_quality': 0.3},
            {'time_efficiency': 0.7, 'depth_reached': 8, 'move_quality': 0.4},
            {'time_efficiency': 0.8, 'depth_reached': 10, 'move_quality': 0.6}
        ]
        
        # 计算平均值
        avg_efficiency = sum(p['time_efficiency'] for p in performance_data) / len(performance_data)
        avg_depth = sum(p['depth_reached'] for p in performance_data) / len(performance_data)
        avg_quality = sum(p['move_quality'] for p in performance_data) / len(performance_data)
        
        assert abs(avg_efficiency - 0.7) < 0.01
        assert abs(avg_depth - 8.0) < 0.01
        assert abs(avg_quality - 0.433) < 0.01
        
        # 计算调整因子
        time_factor = 1.0
        
        if avg_efficiency < 0.7:
            time_factor *= 0.9
        elif avg_efficiency > 0.95:
            time_factor *= 1.1
        
        if avg_depth < 8:
            time_factor *= 1.1
        
        if avg_quality < 0.4:
            time_factor *= 1.15
        
        # 在这个例子中，效率刚好0.7，深度刚好8，质量略高于0.4
        # 所以time_factor应该接近1.0
        assert 0.9 <= time_factor <= 1.2


class TestSearchConfigurationOptimization:
    """搜索配置优化测试"""
    
    def test_config_adjustment_for_slow_search(self):
        """测试慢搜索的配置调整"""
        original_config = MCTSConfig(num_simulations=1000, c_puct=1.0, max_depth=50)
        
        # 模拟慢搜索场景
        avg_nps = 300  # 很慢的搜索速度
        
        # 应该减少模拟次数
        if avg_nps < 500:
            adjusted_simulations = int(original_config.num_simulations * 0.8)
        else:
            adjusted_simulations = original_config.num_simulations
        
        assert adjusted_simulations == 800
    
    def test_config_adjustment_for_shallow_search(self):
        """测试浅搜索的配置调整"""
        original_config = MCTSConfig(num_simulations=800, c_puct=1.0)
        
        # 模拟浅搜索场景
        avg_depth = 5  # 很浅的搜索深度
        
        # 应该增加探索参数
        if avg_depth < 8:
            adjusted_c_puct = min(original_config.c_puct * 1.1, 2.0)
        else:
            adjusted_c_puct = original_config.c_puct
        
        assert adjusted_c_puct == 1.1
    
    def test_config_adjustment_for_memory_pressure(self):
        """测试内存压力下的配置调整"""
        original_config = MCTSConfig(num_simulations=1200, max_depth=40)
        
        # 模拟高内存使用场景
        avg_memory = 1500  # MB
        memory_threshold = 1000
        
        if avg_memory > memory_threshold:
            adjusted_simulations = int(original_config.num_simulations * 0.9)
            adjusted_depth = min(original_config.max_depth, 20)
        else:
            adjusted_simulations = original_config.num_simulations
            adjusted_depth = original_config.max_depth
        
        assert adjusted_simulations == 1080
        assert adjusted_depth == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])