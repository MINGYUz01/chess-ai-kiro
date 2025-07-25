"""
推理引擎

负责模型的推理、批量预测和性能优化。
"""

import torch
import numpy as np
import time
import threading
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import queue
from dataclasses import dataclass

from .chess_net import ChessNet
from .model_manager import ModelManager
from .feature_encoder import FeatureEncoder
from ..rules_engine import ChessBoard, Move

@dataclass
class InferenceRequest:
    """推理请求数据结构"""
    board_tensor: torch.Tensor
    request_id: str
    timestamp: float
    callback: Optional[callable] = None


@dataclass
class InferenceResult:
    """推理结果数据结构"""
    value: float
    policy: np.ndarray
    request_id: str
    inference_time: float
    timestamp: float


class InferenceEngine:
    """
    推理引擎
    
    提供高效的模型推理、批量预测和性能优化功能。
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[ChessNet] = None,
        device: str = 'auto',
        batch_size: int = 32,
        max_queue_size: int = 1000,
        num_workers: int = 2
    ):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型文件路径
            model: 预加载的模型对象
            device: 计算设备 ('cpu', 'cuda', 'auto')
            batch_size: 批处理大小
            max_queue_size: 最大队列大小
            num_workers: 工作线程数
        """
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = self._setup_device(device)
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        
        # 加载模型
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("必须提供model或model_path参数")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化特征编码器
        self.feature_encoder = FeatureEncoder()
        
        # 批处理队列
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'total_inference_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_count': 0,
            'average_batch_size': 0.0
        }
        self.stats_lock = threading.Lock()
        
        # 批处理线程
        self.batch_thread = None
        self.is_running = False
        
        # 预热模型
        self._warmup_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """
        设置计算设备
        
        Args:
            device: 设备类型
            
        Returns:
            torch.device: PyTorch设备对象
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                self.logger.info("使用CPU进行推理")
        
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> ChessNet:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            ChessNet: 加载的模型
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_config' in checkpoint:
            # 从检查点加载
            config = checkpoint['model_config']
            model = ChessNet(
                input_channels=config['input_channels'],
                num_blocks=config['num_blocks'],
                channels=config['channels']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 直接加载模型状态
            model = ChessNet()
            model.load_state_dict(checkpoint)
        
        self.logger.info(f"模型已加载: {model_path}")
        return model
    
    def _warmup_model(self):
        """预热模型以优化首次推理性能"""
        dummy_input = torch.randn(1, 20, 10, 9).to(self.device)
        
        with torch.no_grad():
            for _ in range(3):  # 预热3次
                _ = self.model(dummy_input)
        
        self.logger.info("模型预热完成")
    
    def preprocess_board(
        self, 
        board_data: Any, 
        legal_moves: Optional[List[Move]] = None,
        move_history: Optional[List[Move]] = None
    ) -> torch.Tensor:
        """
        预处理棋盘数据
        
        Args:
            board_data: 棋盘数据（可以是ChessBoard对象、张量或数组）
            legal_moves: 合法走法列表
            move_history: 走法历史
            
        Returns:
            torch.Tensor: 预处理后的张量 [20, 10, 9]
        """
        if isinstance(board_data, ChessBoard):
            # ChessBoard对象，使用特征编码器
            return self.feature_encoder.encode_board(board_data, legal_moves, move_history)
        
        elif isinstance(board_data, torch.Tensor):
            # 已经是张量格式
            if board_data.dim() == 4:  # [B, C, H, W]
                return board_data.squeeze(0)
            elif board_data.dim() == 3:  # [C, H, W]
                return board_data
            else:
                raise ValueError(f"不支持的张量维度: {board_data.dim()}")
        
        elif isinstance(board_data, np.ndarray):
            # NumPy数组格式
            if board_data.ndim == 2:  # [H, W] - 简单棋盘表示
                # 创建临时ChessBoard对象进行编码
                temp_board = ChessBoard()
                temp_board.board = board_data.copy()
                return self.feature_encoder.encode_board(temp_board, legal_moves, move_history)
            elif board_data.ndim == 3:  # [C, H, W]
                return torch.from_numpy(board_data).float()
            else:
                raise ValueError(f"不支持的数组维度: {board_data.ndim}")
        
        else:
            raise TypeError(f"不支持的数据类型: {type(board_data)}")
    
    def predict(
        self, 
        board_data: Any, 
        legal_moves: Optional[List[Move]] = None,
        move_history: Optional[List[Move]] = None
    ) -> Tuple[float, np.ndarray]:
        """
        单次预测
        
        Args:
            board_data: 棋盘数据
            
        Returns:
            Tuple[float, np.ndarray]: (价值评估, 策略分布)
        """
        start_time = time.time()
        
        # 预处理输入
        board_tensor = self.preprocess_board(board_data, legal_moves, move_history)
        
        # 检查缓存
        cache_key = self._get_cache_key(board_tensor)
        with self.cache_lock:
            if cache_key in self.result_cache:
                result = self.result_cache[cache_key]
                with self.stats_lock:
                    self.stats['cache_hits'] += 1
                return result.value, result.policy
        
        # 执行推理
        board_tensor = board_tensor.unsqueeze(0).to(self.device)  # 添加batch维度
        
        with torch.no_grad():
            value, policy = self.model(board_tensor)
            
            # 转换输出
            value_scalar = value.item()
            policy_array = torch.exp(policy).cpu().numpy().flatten()  # 转换为概率
        
        inference_time = time.time() - start_time
        
        # 更新缓存
        result = InferenceResult(
            value=value_scalar,
            policy=policy_array,
            request_id="",
            inference_time=inference_time,
            timestamp=time.time()
        )
        
        with self.cache_lock:
            if len(self.result_cache) < 10000:  # 限制缓存大小
                self.result_cache[cache_key] = result
        
        # 更新统计
        with self.stats_lock:
            self.stats['total_requests'] += 1
            self.stats['total_inference_time'] += inference_time
            self.stats['cache_misses'] += 1
        
        return value_scalar, policy_array
    
    def batch_predict(
        self, 
        board_data_list: List[Any],
        legal_moves_list: Optional[List[List[Move]]] = None,
        move_histories: Optional[List[List[Move]]] = None
    ) -> List[Tuple[float, np.ndarray]]:
        """
        批量预测
        
        Args:
            board_data_list: 棋盘数据列表
            
        Returns:
            List[Tuple[float, np.ndarray]]: 预测结果列表
        """
        if not board_data_list:
            return []
        
        start_time = time.time()
        
        # 预处理所有输入
        board_tensors = []
        cache_results = []
        uncached_indices = []
        
        for i, board_data in enumerate(board_data_list):
            legal_moves = legal_moves_list[i] if legal_moves_list else None
            move_history = move_histories[i] if move_histories else None
            board_tensor = self.preprocess_board(board_data, legal_moves, move_history)
            cache_key = self._get_cache_key(board_tensor)
            
            with self.cache_lock:
                if cache_key in self.result_cache:
                    result = self.result_cache[cache_key]
                    cache_results.append((i, result.value, result.policy))
                    with self.stats_lock:
                        self.stats['cache_hits'] += 1
                else:
                    board_tensors.append(board_tensor)
                    uncached_indices.append(i)
                    with self.stats_lock:
                        self.stats['cache_misses'] += 1
        
        # 批量推理未缓存的数据
        batch_results = []
        if board_tensors:
            batch_tensor = torch.stack(board_tensors).to(self.device)
            
            with torch.no_grad():
                values, policies = self.model(batch_tensor)
                
                # 转换输出
                values_list = values.cpu().numpy().flatten().tolist()
                policies_list = torch.exp(policies).cpu().numpy()
                
                for j, (value, policy) in enumerate(zip(values_list, policies_list)):
                    original_idx = uncached_indices[j]
                    batch_results.append((original_idx, value, policy.flatten()))
                    
                    # 更新缓存
                    cache_key = self._get_cache_key(board_tensors[j])
                    result = InferenceResult(
                        value=value,
                        policy=policy.flatten(),
                        request_id="",
                        inference_time=0.0,
                        timestamp=time.time()
                    )
                    
                    with self.cache_lock:
                        if len(self.result_cache) < 10000:
                            self.result_cache[cache_key] = result
        
        # 合并结果
        all_results = cache_results + batch_results
        all_results.sort(key=lambda x: x[0])  # 按原始索引排序
        
        final_results = [(value, policy) for _, value, policy in all_results]
        
        # 更新统计
        inference_time = time.time() - start_time
        with self.stats_lock:
            self.stats['total_requests'] += len(board_data_list)
            self.stats['total_inference_time'] += inference_time
            self.stats['batch_count'] += 1
            self.stats['average_batch_size'] = (
                (self.stats['average_batch_size'] * (self.stats['batch_count'] - 1) + len(board_data_list)) 
                / self.stats['batch_count']
            )
        
        return final_results
    
    def async_predict(
        self, 
        board_data: Any, 
        callback: Optional[callable] = None
    ) -> Future[Tuple[float, np.ndarray]]:
        """
        异步预测
        
        Args:
            board_data: 棋盘数据
            callback: 回调函数
            
        Returns:
            Future[Tuple[float, np.ndarray]]: 异步结果
        """
        executor = ThreadPoolExecutor(max_workers=1)
        
        def predict_task():
            result = self.predict(board_data)
            if callback:
                callback(result)
            return result
        
        return executor.submit(predict_task)
    
    def start_batch_processing(self):
        """启动批处理模式"""
        if self.is_running:
            return
        
        self.is_running = True
        self.batch_thread = threading.Thread(target=self._batch_processing_loop)
        self.batch_thread.daemon = True
        self.batch_thread.start()
        
        self.logger.info("批处理模式已启动")
    
    def stop_batch_processing(self):
        """停止批处理模式"""
        self.is_running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
        
        self.logger.info("批处理模式已停止")
    
    def _batch_processing_loop(self):
        """批处理循环"""
        while self.is_running:
            try:
                # 收集批处理请求
                requests = []
                timeout = 0.1  # 100ms超时
                
                # 等待第一个请求
                try:
                    first_request = self.request_queue.get(timeout=timeout)
                    requests.append(first_request)
                except queue.Empty:
                    continue
                
                # 收集更多请求直到批处理大小或超时
                start_time = time.time()
                while (len(requests) < self.batch_size and 
                       time.time() - start_time < timeout):
                    try:
                        request = self.request_queue.get_nowait()
                        requests.append(request)
                    except queue.Empty:
                        break
                
                # 执行批处理
                if requests:
                    self._process_batch(requests)
                    
            except Exception as e:
                self.logger.error(f"批处理循环错误: {e}")
    
    def _process_batch(self, requests: List[InferenceRequest]):
        """处理批处理请求"""
        try:
            board_data_list = [req.board_tensor for req in requests]
            results = self.batch_predict(board_data_list)
            
            # 返回结果
            for request, (value, policy) in zip(requests, results):
                result = InferenceResult(
                    value=value,
                    policy=policy,
                    request_id=request.request_id,
                    inference_time=time.time() - request.timestamp,
                    timestamp=time.time()
                )
                
                if request.callback:
                    request.callback(result)
                    
        except Exception as e:
            self.logger.error(f"批处理执行错误: {e}")
    
    def _get_cache_key(self, board_tensor: torch.Tensor) -> str:
        """
        生成缓存键
        
        Args:
            board_tensor: 棋盘张量
            
        Returns:
            str: 缓存键
        """
        # 使用张量的哈希值作为缓存键
        tensor_bytes = board_tensor.cpu().numpy().tobytes()
        import hashlib
        return hashlib.md5(tensor_bytes).hexdigest()
    
    def set_batch_size(self, batch_size: int):
        """
        设置批处理大小
        
        Args:
            batch_size: 新的批处理大小
        """
        self.batch_size = max(1, batch_size)
        self.logger.info(f"批处理大小已设置为: {self.batch_size}")
    
    def clear_cache(self):
        """清空结果缓存"""
        with self.cache_lock:
            self.result_cache.clear()
        self.logger.info("推理缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.stats_lock:
            stats = self.stats.copy()
        
        # 计算平均推理时间
        if stats['total_requests'] > 0:
            stats['average_inference_time'] = (
                stats['total_inference_time'] / stats['total_requests']
            )
        else:
            stats['average_inference_time'] = 0.0
        
        # 计算缓存命中率
        total_cache_requests = stats['cache_hits'] + stats['cache_misses']
        if total_cache_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_requests
        else:
            stats['cache_hit_rate'] = 0.0
        
        # 添加其他信息
        stats['device'] = str(self.device)
        stats['batch_size'] = self.batch_size
        stats['cache_size'] = len(self.result_cache)
        stats['is_batch_processing'] = self.is_running
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        with self.stats_lock:
            self.stats = {
                'total_requests': 0,
                'total_inference_time': 0.0,
                'cache_hits': 0,
                'cache_misses': 0,
                'batch_count': 0,
                'average_batch_size': 0.0
            }
        self.logger.info("统计信息已重置")
    
    def benchmark(self, num_samples: int = 100) -> Dict[str, float]:
        """
        性能基准测试
        
        Args:
            num_samples: 测试样本数
            
        Returns:
            Dict[str, float]: 基准测试结果
        """
        self.logger.info(f"开始性能基准测试，样本数: {num_samples}")
        
        # 生成测试数据
        test_data = [torch.randn(20, 10, 9) for _ in range(num_samples)]
        
        # 单次推理测试
        single_times = []
        for data in test_data[:min(10, num_samples)]:
            start_time = time.time()
            self.predict(data)
            single_times.append(time.time() - start_time)
        
        # 批量推理测试
        batch_start = time.time()
        self.batch_predict(test_data)
        batch_time = time.time() - batch_start
        
        results = {
            'single_inference_avg_time': np.mean(single_times),
            'single_inference_std_time': np.std(single_times),
            'batch_inference_total_time': batch_time,
            'batch_inference_avg_time': batch_time / num_samples,
            'throughput_samples_per_second': num_samples / batch_time,
            'speedup_ratio': np.mean(single_times) / (batch_time / num_samples)
        }
        
        self.logger.info(f"基准测试完成: {results}")
        return results
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop_batch_processing()
        self.clear_cache()