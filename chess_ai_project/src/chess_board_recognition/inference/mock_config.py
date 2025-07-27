"""
模拟配置类，用于测试
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class MockInferenceConfig:
    """模拟推理配置类"""
    
    @dataclass
    class Model:
        input_size: Tuple[int, int] = (640, 640)
        use_gpu: bool = False
    
    @dataclass
    class Detection:
        confidence_threshold: float = 0.5
        nms_threshold: float = 0.4
        min_piece_size: int = 10
        max_piece_size: int = 100
    
    @dataclass
    class Preprocessing:
        enhance_contrast: bool = True
        denoise: bool = True
    
    @dataclass
    class Mapping:
        grid_tolerance: float = 10.0
        position_threshold: float = 30.0
        duplicate_threshold: float = 20.0
    
    @dataclass
    class Validation:
        level: str = "standard"
        min_confidence: float = 0.5
        max_piece_deviation: int = 5
    
    @dataclass
    class Postprocessing:
        enable_piece_filtering: bool = True
        enable_position_correction: bool = True
    
    model: Model = Model()
    detection: Detection = Detection()
    preprocessing: Preprocessing = Preprocessing()
    mapping: Mapping = Mapping()
    validation: Validation = Validation()
    postprocessing: Postprocessing = Postprocessing()


# 为了兼容性，创建别名
InferenceConfig = MockInferenceConfig