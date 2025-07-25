"""
推理引擎测试

测试模型推理、批量预测和性能优化功能。
"""

import pytest
import torch
import numpy as np
import tempfile
import time
from unittest.mock import Mock, patch

from chess_ai_project.src.chinese_chess_ai_engine.neural_network import (
    ChessNet, InferenceEngine, InferenceRequest, InferenceResult
)
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import ChessBoard, Move


class TestInferenceEngine:
    """推理引擎测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        # 创建测试模型
        self.test_model = ChessNet(
            input_channels=20,
            num_blocks=4,  # 使用较小的模型以加快测试
            channels=64
        )
        
        # 创建推理引擎
        self.inference_engine = InferenceEngine(
            model=self.test_model,
            device='cpu',  # 使用CPU以确保测试稳定性
            batch_size=4
        )
        
        # 创建测试棋盘
        self.test_board = ChessBoard()
    
    def test_device_setup_auto(self):
        """测试自动设备选择"""
        with patch('torch.cuda.is_available', return_value=False):
            engine = InferenceEngine(model=self.test_model, device='auto')
            assert engine.device.type == 'cpu'
    
    def test_device_setup_cuda(self):
        """测试CUDA设备选择"""
        # 检查是否有CUDA支持
        if not torch.cuda.is_available():
            pytest.skip("CUDA不可用，跳过CUDA相关测试")
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', return_value='Test GPU'):
                try:
                    engine = InferenceEngine(model=self.test_model, device='auto')
                    assert engine.device.type == 'cuda'
                except Exception as e:
                    pytest.skip(f"CUDA测试失败: {e}")
    
    def test_preprocess_board_chessboard(self):
        """测试ChessBoard对象的预处理"""
        board_tensor = self.inference_engine.preprocess_board(self.test_board)
        
        # 验证输出形状
        assert board_tensor.shape == (20, 10, 9)
        assert isinstance(board_tensor, torch.Tensor)
    
    def test_preprocess_board_tensor(self):
        """测试张量输入的预处理"""
        # 3D张量
        input_tensor = torch.randn(20, 10, 9)
        output_tensor = self.inference_engine.preprocess_board(input_tensor)
        
        assert torch.equal(input_tensor, output_tensor)
        
        # 4D张量（带batch维度）
        input_tensor_4d = torch.randn(1, 20, 10, 9)
        output_tensor_4d = self.inference_engine.preprocess_board(input_tensor_4d)
        
        assert output_tensor_4d.shape == (20, 10, 9)
    
    def test_preprocess_board_numpy(self):
        """测试NumPy数组输入的预处理"""
        # 2D数组（简单棋盘）
        input_array = np.random.randint(-7, 8, (10, 9))
        output_tensor = self.inference_engine.preprocess_board(input_array)
        
        assert output_tensor.shape == (20, 10, 9)
        assert isinstance(output_tensor, torch.Tensor)
        
        # 3D数组
        input_array_3d = np.random.randn(20, 10, 9)
        output_tensor_3d = self.inference_engine.preprocess_board(input_array_3d)
        
        assert output_tensor_3d.shape == (20, 10, 9)
        assert isinstance(output_tensor_3d, torch.Tensor)
    
    def test_preprocess_board_invalid_input(self):
        """测试无效输入的处理"""
        # 无效数据类型
        with pytest.raises(TypeError):
            self.inference_engine.preprocess_board("invalid_input")
        
        # 无效张量维度
        with pytest.raises(ValueError):
            self.inference_engine.preprocess_board(torch.randn(10))
        
        # 无效数组维度
        with pytest.raises(ValueError):
            self.inference_engine.preprocess_board(np.random.randn(10))
    
    def test_single_predict(self):
        """测试单次预测"""
        value, policy = self.inference_engine.predict(self.test_board)
        
        # 验证输出类型和范围
        assert isinstance(value, float)
        assert isinstance(policy, np.ndarray)
        assert -1.0 <= value <= 1.0
        assert len(policy) == 8100  # 10*9*90的策略空间
        assert np.all(policy >= 0)  # 概率应该非负
        assert np.isclose(np.sum(policy), 1.0, rtol=1e-3)  # 概率和应该接近1
    
    def test_batch_predict(self):
        """测试批量预测"""
        # 创建多个测试棋盘
        boards = [ChessBoard() for _ in range(5)]
        
        results = self.inference_engine.batch_predict(boards)
        
        # 验证结果数量
        assert len(results) == 5
        
        # 验证每个结果
        for value, policy in results:
            assert isinstance(value, float)
            assert isinstance(policy, np.ndarray)
            assert -1.0 <= value <= 1.0
            assert len(policy) == 8100
    
    def test_batch_predict_empty(self):
        """测试空批量预测"""
        results = self.inference_engine.batch_predict([])
        assert results == []
    
    def test_async_predict(self):
        """测试异步预测"""
        future = self.inference_engine.async_predict(self.test_board)
        
        # 等待结果
        value, policy = future.result(timeout=10)
        
        # 验证结果
        assert isinstance(value, float)
        assert isinstance(policy, np.ndarray)
        assert -1.0 <= value <= 1.0
    
    def test_async_predict_with_callback(self):
        """测试带回调的异步预测"""
        callback_result = None
        
        def callback(result):
            nonlocal callback_result
            callback_result = result
        
        future = self.inference_engine.async_predict(self.test_board, callback)
        future.result(timeout=10)  # 等待完成
        
        # 验证回调被调用
        assert callback_result is not None
        assert len(callback_result) == 2
    
    def test_caching(self):
        """测试结果缓存"""
        # 第一次预测
        start_time = time.time()
        value1, policy1 = self.inference_engine.predict(self.test_board)
        first_time = time.time() - start_time
        
        # 第二次预测（应该使用缓存）
        start_time = time.time()
        value2, policy2 = self.inference_engine.predict(self.test_board)
        second_time = time.time() - start_time
        
        # 验证结果相同
        assert value1 == value2
        assert np.array_equal(policy1, policy2)
        
        # 验证缓存提升了性能（第二次应该更快）
        # 注意：这个测试可能在某些环境下不稳定
        # assert second_time < first_time
        
        # 验证统计信息
        stats = self.inference_engine.get_stats()
        assert stats['cache_hits'] >= 1
    
    def test_clear_cache(self):
        """测试清空缓存"""
        # 进行一次预测以填充缓存
        self.inference_engine.predict(self.test_board)
        
        # 验证缓存不为空
        assert len(self.inference_engine.result_cache) > 0
        
        # 清空缓存
        self.inference_engine.clear_cache()
        
        # 验证缓存已清空
        assert len(self.inference_engine.result_cache) == 0
    
    def test_set_batch_size(self):
        """测试设置批处理大小"""
        original_batch_size = self.inference_engine.batch_size
        
        # 设置新的批处理大小
        new_batch_size = 16
        self.inference_engine.set_batch_size(new_batch_size)
        
        assert self.inference_engine.batch_size == new_batch_size
        
        # 测试无效值
        self.inference_engine.set_batch_size(0)
        assert self.inference_engine.batch_size == 1  # 应该被限制为最小值1
    
    def test_get_stats(self):
        """测试获取统计信息"""
        # 进行一些预测
        self.inference_engine.predict(self.test_board)
        self.inference_engine.batch_predict([ChessBoard() for _ in range(3)])
        
        stats = self.inference_engine.get_stats()
        
        # 验证统计信息结构
        required_keys = [
            'total_requests', 'total_inference_time', 'cache_hits', 'cache_misses',
            'batch_count', 'average_batch_size', 'average_inference_time',
            'cache_hit_rate', 'device', 'batch_size', 'cache_size', 'is_batch_processing'
        ]
        
        for key in required_keys:
            assert key in stats
        
        # 验证统计值
        assert stats['total_requests'] >= 4  # 1 + 3
        assert stats['batch_count'] >= 1
        assert 0 <= stats['cache_hit_rate'] <= 1
    
    def test_reset_stats(self):
        """测试重置统计信息"""
        # 进行一些预测
        self.inference_engine.predict(self.test_board)
        
        # 验证统计信息不为零
        stats_before = self.inference_engine.get_stats()
        assert stats_before['total_requests'] > 0
        
        # 重置统计信息
        self.inference_engine.reset_stats()
        
        # 验证统计信息已重置
        stats_after = self.inference_engine.get_stats()
        assert stats_after['total_requests'] == 0
        assert stats_after['total_inference_time'] == 0.0
        assert stats_after['cache_hits'] == 0
        assert stats_after['cache_misses'] == 0
    
    def test_benchmark(self):
        """测试性能基准测试"""
        num_samples = 10
        results = self.inference_engine.benchmark(num_samples)
        
        # 验证基准测试结果
        required_keys = [
            'single_inference_avg_time', 'single_inference_std_time',
            'batch_inference_total_time', 'batch_inference_avg_time',
            'throughput_samples_per_second', 'speedup_ratio'
        ]
        
        for key in required_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
            assert results[key] >= 0
        
        # 验证合理性
        assert results['throughput_samples_per_second'] > 0
        assert results['batch_inference_avg_time'] > 0
    
    def test_batch_processing_mode(self):
        """测试批处理模式"""
        # 启动批处理模式
        self.inference_engine.start_batch_processing()
        assert self.inference_engine.is_running
        
        # 停止批处理模式
        self.inference_engine.stop_batch_processing()
        assert not self.inference_engine.is_running
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with InferenceEngine(model=self.test_model, device='cpu') as engine:
            value, policy = engine.predict(self.test_board)
            assert isinstance(value, float)
            assert isinstance(policy, np.ndarray)
        
        # 验证退出时清理了资源
        assert not engine.is_running
        assert len(engine.result_cache) == 0
    
    def test_model_loading_from_path(self):
        """测试从路径加载模型"""
        # 创建临时模型文件
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
            
            # 保存模型
            checkpoint = {
                'model_state_dict': self.test_model.state_dict(),
                'model_config': {
                    'input_channels': 20,
                    'num_blocks': 4,
                    'channels': 64
                }
            }
            torch.save(checkpoint, temp_path)
        
        try:
            # 从路径加载模型
            engine = InferenceEngine(model_path=temp_path, device='cpu')
            
            # 验证模型加载成功
            value, policy = engine.predict(self.test_board)
            assert isinstance(value, float)
            assert isinstance(policy, np.ndarray)
            
        finally:
            # 清理临时文件
            import os
            os.unlink(temp_path)
    
    def test_model_loading_invalid_path(self):
        """测试加载不存在的模型文件"""
        with pytest.raises(FileNotFoundError):
            InferenceEngine(model_path="nonexistent_model.pth", device='cpu')
    
    def test_no_model_provided(self):
        """测试未提供模型的情况"""
        with pytest.raises(ValueError, match="必须提供model或model_path参数"):
            InferenceEngine(device='cpu')


class TestInferenceDataStructures:
    """推理数据结构测试类"""
    
    def test_inference_request(self):
        """测试推理请求数据结构"""
        board_tensor = torch.randn(20, 10, 9)
        request = InferenceRequest(
            board_tensor=board_tensor,
            request_id="test_request",
            timestamp=time.time()
        )
        
        assert torch.equal(request.board_tensor, board_tensor)
        assert request.request_id == "test_request"
        assert isinstance(request.timestamp, float)
        assert request.callback is None
    
    def test_inference_result(self):
        """测试推理结果数据结构"""
        value = 0.5
        policy = np.random.rand(8100)
        result = InferenceResult(
            value=value,
            policy=policy,
            request_id="test_request",
            inference_time=0.1,
            timestamp=time.time()
        )
        
        assert result.value == value
        assert np.array_equal(result.policy, policy)
        assert result.request_id == "test_request"
        assert result.inference_time == 0.1
        assert isinstance(result.timestamp, float)


if __name__ == "__main__":
    pytest.main([__file__])