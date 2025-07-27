"""
棋盘检测器

基于YOLO11的象棋棋盘和棋子检测核心模块。
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics not available, using mock implementation")

try:
    from ..core.interfaces import DetectionResult, ChessboardState
except ImportError:
    # 创建简单的DetectionResult类
    @dataclass
    class DetectionResult:
        image: np.ndarray
        detections: List
        confidence_threshold: float
        processing_time: float
        metadata: Dict = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}

try:
    from ..core.config import InferenceConfig
except ImportError:
    from .mock_config import InferenceConfig


@dataclass
class DetectionBox:
    """检测框数据结构"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[float, float]:
        """获取检测框中心点"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        """获取检测框宽度"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """获取检测框高度"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """获取检测框面积"""
        return self.width * self.height


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        初始化预处理器
        
        Args:
            target_size: 目标图像尺寸 (width, height)
        """
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        预处理图像
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            处理后的图像和缩放信息
        """
        original_height, original_width = image.shape[:2]
        
        # 计算缩放比例，保持宽高比
        scale_w = self.target_size[0] / original_width
        scale_h = self.target_size[1] / original_height
        scale = min(scale_w, scale_h)
        
        # 计算新的尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标尺寸的画布并居中放置图像
        processed = np.full((self.target_size[1], self.target_size[0], 3), 114, dtype=np.uint8)
        
        # 计算偏移量
        offset_x = (self.target_size[0] - new_width) // 2
        offset_y = (self.target_size[1] - new_height) // 2
        
        # 将缩放后的图像放置到画布中心
        processed[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized
        
        # 记录缩放信息
        scale_info = {
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'original_width': original_width,
            'original_height': original_height,
            'new_width': new_width,
            'new_height': new_height
        }
        
        return processed, scale_info
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像
        
        Args:
            image: 输入图像
            
        Returns:
            归一化后的图像
        """
        # 转换为RGB并归一化到[0,1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        normalized = image_rgb.astype(np.float32) / 255.0
        
        return normalized
    
    def denormalize_coordinates(self, boxes: np.ndarray, scale_info: Dict[str, float]) -> np.ndarray:
        """
        将检测框坐标转换回原始图像坐标系
        
        Args:
            boxes: 检测框坐标 (N, 4) [x1, y1, x2, y2]
            scale_info: 缩放信息
            
        Returns:
            原始坐标系下的检测框
        """
        if len(boxes) == 0:
            return boxes
        
        # 保存原始数据类型
        original_dtype = boxes.dtype
        
        # 转换为浮点数以避免类型转换错误
        boxes = boxes.astype(np.float64)
        
        # 减去偏移量
        boxes[:, [0, 2]] -= scale_info['offset_x']  # x坐标
        boxes[:, [1, 3]] -= scale_info['offset_y']  # y坐标
        
        # 缩放回原始尺寸
        boxes /= scale_info['scale']
        
        # 限制在原始图像范围内
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, scale_info['original_width'])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, scale_info['original_height'])
        
        # 转换回原始数据类型
        return boxes.astype(original_dtype)


class ResultProcessor:
    """结果后处理器"""
    
    def __init__(self, class_names: List[str]):
        """
        初始化后处理器
        
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
        self.logger = logging.getLogger(__name__)
    
    def process_detections(self, 
                         results,
                         scale_info: Dict[str, float],
                         confidence_threshold: float = 0.5,
                         nms_threshold: float = 0.4) -> List[DetectionBox]:
        """
        处理YOLO检测结果
        
        Args:
            results: YOLO检测结果
            scale_info: 缩放信息
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测框列表
        """
        detections = []
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]  # 取第一个结果
        
        if result.boxes is None:
            return detections
        
        # 检查是否有检测框（处理Mock对象）
        try:
            if len(result.boxes) == 0:
                return detections
        except TypeError:
            # 处理Mock对象的情况
            pass
        
        # 获取检测框信息
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # 过滤低置信度检测
        valid_indices = confidences >= confidence_threshold
        boxes = boxes[valid_indices]
        confidences = confidences[valid_indices]
        class_ids = class_ids[valid_indices]
        
        if len(boxes) == 0:
            return detections
        
        # 转换回原始坐标系
        original_boxes = self._denormalize_coordinates(boxes, scale_info)
        
        # 创建检测框对象
        for i, (box, conf, cls_id) in enumerate(zip(original_boxes, confidences, class_ids)):
            if cls_id < len(self.class_names):
                detection = DetectionBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=self.class_names[cls_id]
                )
                detections.append(detection)
        
        return detections
    
    def _denormalize_coordinates(self, boxes: np.ndarray, scale_info: Dict[str, float]) -> np.ndarray:
        """将坐标转换回原始图像坐标系"""
        if len(boxes) == 0:
            return boxes
        
        # 转换为浮点数以避免类型转换错误
        boxes = boxes.astype(np.float64)
        
        # 减去偏移量
        boxes[:, [0, 2]] -= scale_info['offset_x']  # x坐标
        boxes[:, [1, 3]] -= scale_info['offset_y']  # y坐标
        
        # 缩放回原始尺寸
        boxes /= scale_info['scale']
        
        # 限制在原始图像范围内
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, scale_info['original_width'])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, scale_info['original_height'])
        
        return boxes
    
    def filter_overlapping_detections(self, 
                                    detections: List[DetectionBox],
                                    iou_threshold: float = 0.5) -> List[DetectionBox]:
        """
        过滤重叠的检测框
        
        Args:
            detections: 检测框列表
            iou_threshold: IoU阈值
            
        Returns:
            过滤后的检测框列表
        """
        if len(detections) <= 1:
            return detections
        
        # 按置信度排序
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        for detection in detections:
            # 检查是否与已选择的检测框重叠
            is_overlapping = False
            for selected in filtered:
                if self._calculate_iou(detection, selected) > iou_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_iou(self, box1: DetectionBox, box2: DetectionBox) -> float:
        """计算两个检测框的IoU"""
        # 计算交集区域
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        return intersection / union if union > 0 else 0.0


class ChessboardDetector:
    """象棋棋盘检测器"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config: Optional[InferenceConfig] = None,
                 device: str = 'auto'):
        """
        初始化检测器
        
        Args:
            model_path: YOLO模型路径，可选
            config: 推理配置
            device: 设备类型 ('auto', 'cpu', 'cuda', 'mps')
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path) if model_path else None
        self.config = config or InferenceConfig()
        
        # 设备选择
        self.device = self._select_device(device)
        
        # 初始化组件
        self.model = None
        
        # 获取输入尺寸
        input_size = getattr(self.config, 'input_size', (640, 640))
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        elif hasattr(self.config, 'model') and hasattr(self.config.model, 'input_size'):
            input_size = self.config.model.input_size
        
        self.preprocessor = ImagePreprocessor(target_size=input_size)
        
        # 象棋棋子类别名称
        self.class_names = [
            'red_king', 'red_advisor', 'red_bishop', 'red_knight', 'red_rook', 'red_cannon', 'red_pawn',
            'black_king', 'black_advisor', 'black_bishop', 'black_knight', 'black_rook', 'black_cannon', 'black_pawn',
            'board', 'selected_piece'
        ]
        
        self.result_processor = ResultProcessor(self.class_names)
        
        # 加载模型（如果提供了路径）
        if self.model_path:
            self._load_model()
        
        self.logger.info(f"棋盘检测器初始化完成，设备: {self.device}")
    
    def load_model(self, model_path: str) -> bool:
        """
        加载YOLO模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        try:
            self.model_path = Path(model_path)
            self._load_model()
            return True
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
    def _select_device(self, device: str) -> str:
        """选择计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            if not self.model_path or not self.model_path.exists():
                self.logger.warning(f"模型文件不存在: {self.model_path}，使用模拟模型")
                self.model = self._create_mock_model()
                return
            
            if YOLO is None:
                self.logger.warning("YOLO不可用，使用模拟模型")
                self.model = self._create_mock_model()
                return
            
            self.logger.info(f"加载YOLO模型: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            
            # 设置设备
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            # 预热模型
            self._warmup_model()
            
            self.logger.info("模型加载成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}，使用模拟模型")
            self.model = self._create_mock_model()
    
    def _warmup_model(self):
        """预热模型"""
        try:
            # 创建虚拟输入进行预热
            input_size = getattr(self.config, 'input_size', (640, 640))
            if isinstance(input_size, int):
                size = input_size
            else:
                size = input_size[0] if input_size else 640
            dummy_input = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            
            self.logger.debug("开始模型预热...")
            _ = self.model(dummy_input, verbose=False)
            self.logger.debug("模型预热完成")
            
        except Exception as e:
            self.logger.warning(f"模型预热失败: {e}")
    
    def detect(self, 
               image: Union[np.ndarray, str, Path],
               confidence_threshold: Optional[float] = None,
               nms_threshold: Optional[float] = None) -> DetectionResult:
        """
        检测棋盘和棋子
        
        Args:
            image: 输入图像（numpy数组或图像路径）
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测结果
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 参数默认值
        conf_thresh = confidence_threshold or getattr(self.config, 'confidence_threshold', 0.5)
        nms_thresh = nms_threshold or getattr(self.config, 'nms_threshold', 0.4)
        
        try:
            # 加载图像
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
                if image is None:
                    raise ValueError(f"无法加载图像: {image}")
            
            original_image = image.copy()
            
            # 预处理
            processed_image, scale_info = self.preprocessor.preprocess(image)
            
            # 推理
            self.logger.debug("开始YOLO推理...")
            results = self.model(
                processed_image,
                conf=conf_thresh,
                iou=nms_thresh,
                verbose=False,
                device=self.device
            )
            
            # 后处理
            detections = self.result_processor.process_detections(
                results, scale_info, conf_thresh, nms_thresh
            )
            
            # 过滤重叠检测
            overlap_threshold = getattr(self.config, 'overlap_threshold', 0.5)
            filtered_detections = self.result_processor.filter_overlapping_detections(
                detections, overlap_threshold
            )
            
            self.logger.debug(f"检测到 {len(filtered_detections)} 个对象")
            
            # 创建检测结果
            detection_result = DetectionResult(
                image=original_image,
                detections=filtered_detections,
                confidence_threshold=conf_thresh,
                processing_time=0.0,  # 这里可以添加时间统计
                metadata={
                    'model_path': str(self.model_path) if self.model_path else None,
                    'device': self.device,
                    'input_size': getattr(self.config, 'input_size', (640, 640)),
                    'scale_info': scale_info
                }
            )
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"检测失败: {e}")
            raise
    
    def detect_batch(self, 
                    images: List[Union[np.ndarray, str, Path]],
                    confidence_threshold: Optional[float] = None,
                    nms_threshold: Optional[float] = None) -> List[DetectionResult]:
        """
        批量检测
        
        Args:
            images: 图像列表
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测结果列表
        """
        results = []
        for image in images:
            try:
                result = self.detect(image, confidence_threshold, nms_threshold)
                results.append(result)
            except Exception as e:
                self.logger.error(f"批量检测中的图像处理失败: {e}")
                # 创建空结果
                empty_result = DetectionResult(
                    image=np.zeros((100, 100, 3), dtype=np.uint8),
                    detections=[],
                    confidence_threshold=confidence_threshold or getattr(self.config, 'confidence_threshold', 0.5),
                    processing_time=0.0,
                    metadata={'error': str(e)}
                )
                results.append(empty_result)
        
        return results
    
    def visualize_detections(self, 
                           detection_result: DetectionResult,
                           show_confidence: bool = True,
                           show_class_names: bool = True) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            detection_result: 检测结果
            show_confidence: 是否显示置信度
            show_class_names: 是否显示类别名称
            
        Returns:
            可视化图像
        """
        image = detection_result.image.copy()
        
        # 定义颜色映射
        colors = {
            'red_king': (0, 0, 255), 'red_advisor': (0, 50, 255), 'red_bishop': (0, 100, 255),
            'red_knight': (0, 150, 255), 'red_rook': (0, 200, 255), 'red_cannon': (0, 255, 255),
            'red_pawn': (0, 255, 200),
            'black_king': (255, 0, 0), 'black_advisor': (255, 50, 0), 'black_bishop': (255, 100, 0),
            'black_knight': (255, 150, 0), 'black_rook': (255, 200, 0), 'black_cannon': (255, 255, 0),
            'black_pawn': (200, 255, 0),
            'board': (128, 128, 128), 'selected_piece': (0, 255, 0)
        }
        
        for detection in detection_result.detections:
            # 获取颜色
            color = colors.get(detection.class_name, (255, 255, 255))
            
            # 绘制检测框
            cv2.rectangle(image, 
                         (int(detection.x1), int(detection.y1)),
                         (int(detection.x2), int(detection.y2)),
                         color, 2)
            
            # 准备标签文本
            label_parts = []
            if show_class_names:
                label_parts.append(detection.class_name)
            if show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # 计算文本尺寸
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 绘制文本背景
                cv2.rectangle(image,
                             (int(detection.x1), int(detection.y1) - text_height - baseline),
                             (int(detection.x1) + text_width, int(detection.y1)),
                             color, -1)
                
                # 绘制文本
                cv2.putText(image, label,
                           (int(detection.x1), int(detection.y1) - baseline),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        if self.model is None:
            return {}
        
        info = {
            'model_path': str(self.model_path) if self.model_path else None,
            'device': self.device,
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'input_size': getattr(self.config, 'input_size', (640, 640)),
            'confidence_threshold': getattr(self.config, 'confidence_threshold', 0.5),
            'nms_threshold': getattr(self.config, 'nms_threshold', 0.4)
        }
        
        # 尝试获取模型的详细信息
        try:
            if hasattr(self.model, 'info'):
                model_info = self.model.info(verbose=False)
                if model_info:
                    info.update(model_info)
        except Exception as e:
            self.logger.debug(f"获取模型详细信息失败: {e}")
        
        return info
    
    def update_config(self, config: InferenceConfig):
        """更新推理配置"""
        self.config = config
        
        # 更新预处理器
        input_size = getattr(config, 'input_size', (640, 640))
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.preprocessor = ImagePreprocessor(target_size=input_size)
        
        self.logger.info("推理配置已更新")
    
    def _create_mock_model(self):
        """创建模拟模型用于测试"""
        class MockModel:
            def __init__(self, detector):
                self.detector = detector
            
            def __call__(self, image, **kwargs):
                # 模拟检测结果
                h, w = image.shape[:2] if isinstance(image, np.ndarray) else (640, 640)
                
                # 模拟棋盘检测
                mock_result = type('Result', (), {
                    'boxes': type('Boxes', (), {
                        'xyxy': type('Tensor', (), {
                            'cpu': lambda: type('CPUTensor', (), {
                                'numpy': lambda: np.array([[w*0.1, h*0.1, w*0.9, h*0.9]])
                            })()
                        })(),
                        'conf': type('Tensor', (), {
                            'cpu': lambda: type('CPUTensor', (), {
                                'numpy': lambda: np.array([0.95])
                            })()
                        })(),
                        'cls': type('Tensor', (), {
                            'cpu': lambda: type('CPUTensor', (), {
                                'numpy': lambda: type('Array', (), {
                                    'astype': lambda dtype: np.array([0])
                                })()
                            })()
                        })()
                    })()
                })()
                
                # 生成一些模拟的棋子位置
                piece_boxes = []
                piece_confs = []
                piece_classes = []
                
                for i in range(10, 90, 10):
                    for j in range(10, 90, 10):
                        if np.random.random() > 0.7:  # 30%概率有棋子
                            x = w * i / 100
                            y = h * j / 100
                            size = min(w, h) * 0.05
                            piece_boxes.append([x-size/2, y-size/2, x+size/2, y+size/2])
                            piece_confs.append(0.8 + np.random.random() * 0.15)
                            piece_classes.append(np.random.randint(1, 15))
                
                if piece_boxes:
                    # 合并棋盘和棋子检测结果
                    all_boxes = np.vstack([
                        np.array([[w*0.1, h*0.1, w*0.9, h*0.9]]),
                        np.array(piece_boxes)
                    ])
                    all_confs = np.concatenate([
                        np.array([0.95]),
                        np.array(piece_confs)
                    ])
                    all_classes = np.concatenate([
                        np.array([0]),
                        np.array(piece_classes)
                    ])
                    
                    mock_result.boxes.xyxy.cpu().numpy = lambda: all_boxes
                    mock_result.boxes.conf.cpu().numpy = lambda: all_confs
                    mock_result.boxes.cls.cpu().numpy().astype = lambda dtype: all_classes
                
                return [mock_result]
            
            def to(self, device):
                return self
            
            def info(self, verbose=False):
                return {'model_type': 'mock', 'parameters': 0}
        
        return MockModel(self)
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.logger.debug("模型资源已释放")