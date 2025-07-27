"""
推理引擎测试

测试棋盘检测和识别功能。
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from chess_ai_project.src.chess_board_recognition.inference import (
    ChessboardDetector, DetectionBox, ImagePreprocessor, BoardMapper,
    ResultProcessor, QualityMetrics, StateValidator, ConfidenceCalculator
)
from chess_ai_project.src.chess_board_recognition.core.interfaces import (
    DetectionResult, ChessboardState, BoardPosition
)
try:
    from chess_ai_project.src.chess_board_recognition.core.config import InferenceConfig
except ImportError:
    from chess_ai_project.src.chess_board_recognition.inference.mock_config import InferenceConfig


class TestDetectionBox:
    """测试检测框类"""
    
    def test_detection_box_creation(self):
        """测试检测框创建"""
        box = DetectionBox(
            x1=10.0, y1=20.0, x2=50.0, y2=60.0,
            confidence=0.8, class_id=1, class_name='red_king'
        )
        
        assert box.x1 == 10.0
        assert box.y1 == 20.0
        assert box.x2 == 50.0
        assert box.y2 == 60.0
        assert box.confidence == 0.8
        assert box.class_id == 1
        assert box.class_name == 'red_king'
    
    def test_detection_box_properties(self):
        """测试检测框属性"""
        box = DetectionBox(
            x1=10.0, y1=20.0, x2=50.0, y2=60.0,
            confidence=0.8, class_id=1, class_name='red_king'
        )
        
        # 测试中心点
        center = box.center
        assert center == (30.0, 40.0)
        
        # 测试宽度和高度
        assert box.width == 40.0
        assert box.height == 40.0
        
        # 测试面积
        assert box.area == 1600.0


class TestImagePreprocessor:
    """测试图像预处理器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.preprocessor = ImagePreprocessor(target_size=(640, 640))
    
    def test_preprocess_image(self):
        """测试图像预处理"""
        # 创建测试图像
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed_image, scale_info = self.preprocessor.preprocess(image)
        
        # 检查输出尺寸
        assert processed_image.shape == (640, 640, 3)
        
        # 检查缩放信息
        assert 'scale' in scale_info
        assert 'offset_x' in scale_info
        assert 'offset_y' in scale_info
        assert 'original_width' in scale_info
        assert 'original_height' in scale_info
        
        assert scale_info['original_width'] == 640
        assert scale_info['original_height'] == 480
    
    def test_normalize_image(self):
        """测试图像归一化"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        normalized = self.preprocessor.normalize(image)
        
        # 检查数据类型和范围
        assert normalized.dtype == np.float32
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0
        assert normalized.shape == (100, 100, 3)
    
    def test_denormalize_coordinates(self):
        """测试坐标反归一化"""
        # 创建测试检测框
        boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        
        scale_info = {
            'scale': 0.5,
            'offset_x': 50,
            'offset_y': 25,
            'original_width': 800,
            'original_height': 600
        }
        
        denormalized = self.preprocessor.denormalize_coordinates(boxes, scale_info)
        
        # 检查结果
        assert denormalized.shape == boxes.shape
        assert denormalized.dtype == boxes.dtype


class TestBoardMapper:
    """测试棋盘映射器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.board_mapper = BoardMapper()
    
    def test_map_detections_to_board(self):
        """测试检测结果映射"""
        # 创建模拟检测结果
        detections = [
            DetectionBox(100, 100, 200, 200, 0.9, 0, 'board'),
            DetectionBox(150, 150, 160, 160, 0.8, 1, 'red_king'),
            DetectionBox(170, 150, 180, 160, 0.7, 2, 'red_advisor')
        ]
        
        image_shape = (600, 800)  # height, width
        
        chessboard_state = self.board_mapper.map_detections_to_board(detections, image_shape)
        
        # 检查结果
        assert isinstance(chessboard_state, ChessboardState)
        assert chessboard_state.board_matrix.shape == (10, 9)
        assert chessboard_state.board_corners is not None
        assert len(chessboard_state.grid_points) == 10
        assert len(chessboard_state.grid_points[0]) == 9
    
    def test_visualize_grid(self):
        """测试网格可视化"""
        # 创建模拟棋盘状态
        board_matrix = np.zeros((10, 9), dtype=int)
        board_matrix[0, 4] = -1  # 黑将
        board_matrix[9, 4] = 1   # 红帅
        
        grid_points = []
        for row in range(10):
            row_points = []
            for col in range(9):
                point = BoardPosition(
                    x=col * 50 + 50, y=row * 50 + 50,
                    row=row, col=col,
                    occupied=(board_matrix[row, col] != 0),
                    piece_type='red_king' if board_matrix[row, col] == 1 else 
                              ('black_king' if board_matrix[row, col] == -1 else None),
                    confidence=0.8 if board_matrix[row, col] != 0 else 0.0
                )
                row_points.append(point)
            grid_points.append(row_points)
        
        chessboard_state = ChessboardState(
            board_matrix=board_matrix,
            board_corners=((0, 0), (400, 0), (0, 450), (400, 450)),
            grid_points=grid_points,
            selected_position=None,
            validation_result={},
            detection_count={'board': 1, 'pieces': 2},
            confidence_stats={'mean_confidence': 0.8}
        )
        
        # 创建测试图像
        image = np.zeros((500, 450, 3), dtype=np.uint8)
        
        vis_image = self.board_mapper.visualize_grid(image, chessboard_state)
        
        # 检查可视化结果
        assert vis_image.shape == image.shape
        assert vis_image.dtype == image.dtype
    
    def test_export_board_state(self):
        """测试棋盘状态导出"""
        # 创建简单的棋盘状态
        board_matrix = np.array([[1, 0], [0, -1]])
        
        chessboard_state = ChessboardState(
            board_matrix=board_matrix,
            board_corners=((0, 0), (100, 0), (0, 100), (100, 100)),
            grid_points=[],
            selected_position=(0, 0),
            validation_result={'valid': True},
            detection_count={'pieces': 2},
            confidence_stats={'mean_confidence': 0.8}
        )
        
        exported = self.board_mapper.export_board_state(chessboard_state)
        
        # 检查导出结果
        assert 'board_matrix' in exported
        assert 'selected_position' in exported
        assert 'validation_result' in exported
        assert 'detection_count' in exported
        assert 'confidence_stats' in exported
        assert 'piece_positions' in exported
        
        assert exported['board_matrix'] == board_matrix.tolist()
        assert exported['selected_position'] == (0, 0)


class TestStateValidator:
    """测试状态验证器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.validator = StateValidator()
    
    def test_validate_piece_counts(self):
        """测试棋子数量验证"""
        # 创建标准开局的网格点
        grid_points = []
        for row in range(10):
            row_points = []
            for col in range(9):
                point = BoardPosition(x=col*50, y=row*50, row=row, col=col)
                row_points.append(point)
            grid_points.append(row_points)
        
        # 设置一些棋子
        grid_points[0][4].occupied = True
        grid_points[0][4].piece_type = 'black_king'
        grid_points[9][4].occupied = True
        grid_points[9][4].piece_type = 'red_king'
        
        chessboard_state = ChessboardState(
            board_matrix=np.zeros((10, 9)),
            board_corners=(),
            grid_points=grid_points,
            selected_position=None,
            validation_result={},
            detection_count={},
            confidence_stats={}
        )
        
        result = self.validator.validate_piece_counts(chessboard_state)
        
        # 检查验证结果
        assert 'valid' in result
        assert 'piece_counts' in result
        assert 'missing_pieces' in result
        assert 'extra_pieces' in result
        
        # 应该有很多缺失的棋子
        assert len(result['missing_pieces']) > 0
    
    def test_validate_piece_positions(self):
        """测试棋子位置验证"""
        # 创建网格点
        grid_points = []
        for row in range(10):
            row_points = []
            for col in range(9):
                point = BoardPosition(x=col*50, y=row*50, row=row, col=col)
                row_points.append(point)
            grid_points.append(row_points)
        
        # 在错误位置放置红帅（应该在九宫格内）
        grid_points[5][5].occupied = True
        grid_points[5][5].piece_type = 'red_king'
        
        chessboard_state = ChessboardState(
            board_matrix=np.zeros((10, 9)),
            board_corners=(),
            grid_points=grid_points,
            selected_position=None,
            validation_result={},
            detection_count={},
            confidence_stats={}
        )
        
        result = self.validator.validate_piece_positions(chessboard_state)
        
        # 应该检测到位置错误
        assert not result['valid']
        assert len(result['errors']) > 0
        assert len(result['invalid_positions']) > 0


class TestConfidenceCalculator:
    """测试置信度计算器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.calculator = ConfidenceCalculator()
    
    def test_calculate_overall_confidence(self):
        """测试整体置信度计算"""
        # 创建网格点
        grid_points = []
        for row in range(10):
            row_points = []
            for col in range(9):
                point = BoardPosition(
                    x=col*50, y=row*50, row=row, col=col,
                    occupied=(row < 2),  # 前两行有棋子
                    confidence=0.8 if row < 2 else 0.0
                )
                row_points.append(point)
            grid_points.append(row_points)
        
        chessboard_state = ChessboardState(
            board_matrix=np.zeros((10, 9)),
            board_corners=(),
            grid_points=grid_points,
            selected_position=None,
            validation_result={},
            detection_count={},
            confidence_stats={}
        )
        
        confidence = self.calculator.calculate_overall_confidence(chessboard_state)
        
        # 应该接近0.8
        assert 0.7 <= confidence <= 0.9
    
    def test_calculate_detection_completeness(self):
        """测试检测完整性计算"""
        # 创建网格点，模拟检测到16个棋子（一半）
        grid_points = []
        piece_count = 0
        for row in range(10):
            row_points = []
            for col in range(9):
                occupied = piece_count < 16
                point = BoardPosition(
                    x=col*50, y=row*50, row=row, col=col,
                    occupied=occupied
                )
                if occupied:
                    piece_count += 1
                row_points.append(point)
            grid_points.append(row_points)
        
        chessboard_state = ChessboardState(
            board_matrix=np.zeros((10, 9)),
            board_corners=(),
            grid_points=grid_points,
            selected_position=None,
            validation_result={},
            detection_count={},
            confidence_stats={}
        )
        
        completeness = self.calculator.calculate_detection_completeness(chessboard_state)
        
        # 应该是0.5（16/32）
        assert abs(completeness - 0.5) < 0.1


class TestResultProcessor:
    """测试结果处理器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.processor = ResultProcessor()
    
    @patch('chess_ai_project.src.chess_board_recognition.inference.result_processor.BoardMapper')
    def test_process_detection_result(self, mock_board_mapper_class):
        """测试检测结果处理"""
        # 模拟BoardMapper
        mock_board_mapper = Mock()
        mock_board_mapper_class.return_value = mock_board_mapper
        
        # 创建模拟棋盘状态
        mock_chessboard_state = ChessboardState(
            board_matrix=np.zeros((10, 9)),
            board_corners=(),
            grid_points=[],
            selected_position=None,
            validation_result={},
            detection_count={'pieces': 5},
            confidence_stats={'mean_confidence': 0.8}
        )
        
        mock_board_mapper.map_detections_to_board.return_value = mock_chessboard_state
        
        # 创建检测结果
        detection_result = DetectionResult(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            detections=[],
            confidence_threshold=0.5,
            processing_time=0.1
        )
        
        # 处理结果
        processed_state = self.processor.process_detection_result(detection_result)
        
        # 检查处理结果
        assert isinstance(processed_state, ChessboardState)
        assert 'quality_metrics' in processed_state.confidence_stats
        assert 'processing_time' in processed_state.confidence_stats
    
    def test_get_processing_statistics(self):
        """测试处理统计信息"""
        # 初始状态下应该没有统计信息
        stats = self.processor.get_processing_statistics()
        assert stats == {}
        
        # 添加一些历史状态
        for i in range(3):
            mock_state = ChessboardState(
                board_matrix=np.zeros((10, 9)),
                board_corners=(),
                grid_points=[],
                selected_position=None,
                validation_result={},
                detection_count={},
                confidence_stats={
                    'quality_metrics': {
                        'quality_grade': 'A',
                        'overall_confidence': 0.9
                    },
                    'processing_time': 0.1
                }
            )
            self.processor._update_history(mock_state)
        
        stats = self.processor.get_processing_statistics()
        
        # 检查统计信息
        assert 'total_processed' in stats
        assert 'recent_average_confidence' in stats
        assert 'recent_average_processing_time' in stats
        assert 'quality_grade_distribution' in stats
        assert 'stability_trend' in stats
        
        assert stats['total_processed'] == 3
    
    def test_clear_history(self):
        """测试清空历史记录"""
        # 添加一些历史状态
        mock_state = ChessboardState(
            board_matrix=np.zeros((10, 9)),
            board_corners=(),
            grid_points=[],
            selected_position=None,
            validation_result={},
            detection_count={},
            confidence_stats={}
        )
        
        self.processor._update_history(mock_state)
        assert len(self.processor.history_states) == 1
        
        # 清空历史
        self.processor.clear_history()
        assert len(self.processor.history_states) == 0


@patch('chess_ai_project.src.chess_board_recognition.inference.chessboard_detector.YOLO')
class TestChessboardDetector:
    """测试棋盘检测器"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建临时模型文件
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        self.temp_model_file.close()
        
        self.config = InferenceConfig()
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)
    
    def test_detector_initialization(self, mock_yolo_class):
        """测试检测器初始化"""
        # 模拟YOLO模型
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        
        detector = ChessboardDetector(
            model_path=self.temp_model_file.name,
            config=self.config
        )
        
        assert detector.model_path == Path(self.temp_model_file.name)
        assert detector.config == self.config
        assert detector.device in ['cpu', 'cuda', 'mps']
        assert len(detector.class_names) > 0
    
    def test_detect_image(self, mock_yolo_class):
        """测试图像检测"""
        # 模拟YOLO模型和结果
        mock_model = Mock()
        mock_result = Mock()
        mock_boxes = Mock()
        
        # 模拟检测框数据
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
        mock_boxes.cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([0])
        
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model
        
        detector = ChessboardDetector(
            model_path=self.temp_model_file.name,
            config=self.config
        )
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 执行检测
        detection_result = detector.detect(test_image)
        
        # 检查结果
        assert isinstance(detection_result, DetectionResult)
        assert detection_result.image.shape == test_image.shape
        assert len(detection_result.detections) >= 0
    
    def test_detect_batch(self, mock_yolo_class):
        """测试批量检测"""
        # 模拟YOLO模型
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None  # 无检测结果
        mock_model.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model
        
        detector = ChessboardDetector(
            model_path=self.temp_model_file.name,
            config=self.config
        )
        
        # 创建测试图像列表
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ]
        
        # 执行批量检测
        results = detector.detect_batch(images)
        
        # 检查结果
        assert len(results) == 2
        assert all(isinstance(r, DetectionResult) for r in results)
    
    def test_get_model_info(self, mock_yolo_class):
        """测试获取模型信息"""
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        
        detector = ChessboardDetector(
            model_path=self.temp_model_file.name,
            config=self.config
        )
        
        info = detector.get_model_info()
        
        # 检查信息内容
        assert 'model_path' in info
        assert 'device' in info
        assert 'class_names' in info
        assert 'num_classes' in info
        assert 'input_size' in info
        assert 'confidence_threshold' in info
        assert 'nms_threshold' in info


if __name__ == '__main__':
    pytest.main([__file__])