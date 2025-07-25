"""
模型管理和推理系统演示

展示模型保存、加载、推理等功能的使用方法。
"""

import torch
import numpy as np
import time
import logging
from pathlib import Path

from .neural_network import ChessNet, ModelManager, InferenceEngine
from .rules_engine import ChessBoard, Move

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_model():
    """创建示例模型"""
    logger.info("创建示例神经网络模型...")
    
    model = ChessNet(
        input_channels=20,
        num_blocks=8,
        channels=128
    )
    
    logger.info(f"模型信息: {model.get_model_info()}")
    return model


def demonstrate_model_manager():
    """演示模型管理功能"""
    logger.info("=== 模型管理演示 ===")
    
    # 创建模型管理器
    model_manager = ModelManager("models/demo")
    
    # 创建示例模型
    model = create_sample_model()
    
    # 保存模型
    logger.info("保存模型...")
    metadata = {
        "description": "演示模型",
        "training_epochs": 50,
        "accuracy": 0.85,
        "loss": 0.15,
        "created_by": "demo_script"
    }
    
    model_path = model_manager.save_model(
        model, 
        "demo_v1.0", 
        metadata,
        is_best=True
    )
    logger.info(f"模型已保存到: {model_path}")
    
    # 保存另一个版本
    logger.info("保存另一个版本...")
    metadata_v2 = metadata.copy()
    metadata_v2.update({
        "training_epochs": 100,
        "accuracy": 0.90,
        "loss": 0.10
    })
    
    model_manager.save_model(model, "demo_v2.0", metadata_v2)
    
    # 列出所有模型
    logger.info("列出所有模型版本:")
    models = model_manager.list_models()
    for model_info in models:
        logger.info(f"  版本: {model_info['version']}")
        logger.info(f"    时间: {model_info['timestamp']}")
        logger.info(f"    大小: {model_info.get('file_size_mb', 0):.2f} MB")
        logger.info(f"    最新: {model_info['is_latest']}")
        logger.info(f"    最佳: {model_info['is_best']}")
        if 'metadata' in model_info:
            logger.info(f"    准确率: {model_info['metadata'].get('accuracy', 'N/A')}")
    
    # 加载模型
    logger.info("加载最新模型...")
    loaded_model, loaded_metadata = model_manager.load_model('latest')
    logger.info(f"加载的模型版本: {loaded_metadata.get('version', 'unknown')}")
    logger.info(f"模型准确率: {loaded_metadata.get('accuracy', 'N/A')}")
    
    # 导出ONNX（如果可用）
    try:
        logger.info("导出ONNX模型...")
        onnx_path = model_manager.export_onnx(model, "demo_v1.0")
        logger.info(f"ONNX模型已导出到: {onnx_path}")
    except ImportError:
        logger.warning("ONNX不可用，跳过导出")
    except Exception as e:
        logger.warning(f"ONNX导出失败: {e}")
    
    # 量化模型
    try:
        logger.info("量化模型...")
        quant_path = model_manager.quantize_model(model, "demo_v1.0")
        logger.info(f"量化模型已保存到: {quant_path}")
    except Exception as e:
        logger.warning(f"模型量化失败: {e}")
    
    # 获取存储信息
    storage_info = model_manager.get_storage_info()
    logger.info("存储信息:")
    logger.info(f"  总大小: {storage_info['total_size_mb']:.2f} MB")
    logger.info(f"  文件数量: {storage_info['file_count']}")
    logger.info(f"  版本数量: {storage_info['versions_count']}")
    
    return loaded_model


def demonstrate_inference_engine(model):
    """演示推理引擎功能"""
    logger.info("=== 推理引擎演示 ===")
    
    # 创建推理引擎
    logger.info("创建推理引擎...")
    inference_engine = InferenceEngine(
        model=model,
        device='auto',
        batch_size=8
    )
    
    # 创建测试棋盘
    logger.info("创建测试棋盘...")
    test_board = ChessBoard()
    
    # 单次推理
    logger.info("执行单次推理...")
    start_time = time.time()
    value, policy = inference_engine.predict(test_board)
    inference_time = time.time() - start_time
    
    logger.info(f"推理结果:")
    logger.info(f"  价值评估: {value:.4f}")
    logger.info(f"  策略分布形状: {policy.shape}")
    logger.info(f"  策略分布和: {np.sum(policy):.4f}")
    logger.info(f"  推理时间: {inference_time:.4f}s")
    
    # 批量推理
    logger.info("执行批量推理...")
    test_boards = [ChessBoard() for _ in range(10)]
    
    start_time = time.time()
    batch_results = inference_engine.batch_predict(test_boards)
    batch_time = time.time() - start_time
    
    logger.info(f"批量推理结果:")
    logger.info(f"  批量大小: {len(batch_results)}")
    logger.info(f"  总时间: {batch_time:.4f}s")
    logger.info(f"  平均时间: {batch_time/len(batch_results):.4f}s")
    
    # 测试缓存效果
    logger.info("测试缓存效果...")
    
    # 第一次推理
    start_time = time.time()
    inference_engine.predict(test_board)
    first_time = time.time() - start_time
    
    # 第二次推理（使用缓存）
    start_time = time.time()
    inference_engine.predict(test_board)
    second_time = time.time() - start_time
    
    logger.info(f"缓存效果:")
    logger.info(f"  第一次推理: {first_time:.4f}s")
    logger.info(f"  第二次推理: {second_time:.4f}s")
    logger.info(f"  加速比: {first_time/second_time:.2f}x")
    
    # 异步推理
    logger.info("测试异步推理...")
    
    def async_callback(result):
        logger.info(f"异步推理完成，价值: {result[0]:.4f}")
    
    future = inference_engine.async_predict(test_board, async_callback)
    result = future.result(timeout=10)
    logger.info(f"异步推理结果: 价值={result[0]:.4f}")
    
    # 性能基准测试
    logger.info("执行性能基准测试...")
    benchmark_results = inference_engine.benchmark(num_samples=50)
    
    logger.info("基准测试结果:")
    for key, value in benchmark_results.items():
        logger.info(f"  {key}: {value:.6f}")
    
    # 获取统计信息
    stats = inference_engine.get_stats()
    logger.info("推理引擎统计:")
    logger.info(f"  总请求数: {stats['total_requests']}")
    logger.info(f"  平均推理时间: {stats['average_inference_time']:.4f}s")
    logger.info(f"  缓存命中率: {stats['cache_hit_rate']:.2%}")
    logger.info(f"  缓存大小: {stats['cache_size']}")
    logger.info(f"  设备: {stats['device']}")
    
    return inference_engine


def demonstrate_advanced_features():
    """演示高级功能"""
    logger.info("=== 高级功能演示 ===")
    
    # 创建模型和推理引擎
    model = create_sample_model()
    
    with InferenceEngine(model=model, device='cpu') as engine:
        logger.info("使用上下文管理器创建推理引擎")
        
        # 测试不同输入格式
        logger.info("测试不同输入格式...")
        
        # ChessBoard对象
        board = ChessBoard()
        value1, policy1 = engine.predict(board)
        logger.info(f"ChessBoard输入: 价值={value1:.4f}")
        
        # 张量输入
        tensor_input = torch.randn(20, 10, 9)
        value2, policy2 = engine.predict(tensor_input)
        logger.info(f"张量输入: 价值={value2:.4f}")
        
        # NumPy数组输入
        array_input = np.random.randint(-7, 8, (10, 9))
        value3, policy3 = engine.predict(array_input)
        logger.info(f"NumPy数组输入: 价值={value3:.4f}")
        
        # 测试批处理大小调整
        logger.info("测试批处理大小调整...")
        original_batch_size = engine.batch_size
        engine.set_batch_size(16)
        logger.info(f"批处理大小从 {original_batch_size} 调整为 {engine.batch_size}")
        
        # 测试缓存清理
        logger.info("测试缓存管理...")
        cache_size_before = engine.get_stats()['cache_size']
        engine.clear_cache()
        cache_size_after = engine.get_stats()['cache_size']
        logger.info(f"缓存大小: {cache_size_before} -> {cache_size_after}")
        
        # 测试统计重置
        logger.info("测试统计重置...")
        requests_before = engine.get_stats()['total_requests']
        engine.reset_stats()
        requests_after = engine.get_stats()['total_requests']
        logger.info(f"总请求数: {requests_before} -> {requests_after}")


def main():
    """主函数"""
    logger.info("开始模型管理和推理系统演示")
    
    try:
        # 演示模型管理
        model = demonstrate_model_manager()
        
        # 演示推理引擎
        inference_engine = demonstrate_inference_engine(model)
        
        # 演示高级功能
        demonstrate_advanced_features()
        
        logger.info("演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise
    
    finally:
        # 清理资源
        if 'inference_engine' in locals():
            inference_engine.stop_batch_processing()
            inference_engine.clear_cache()


if __name__ == "__main__":
    main()