"""
棋盘映射器

将检测结果转换为象棋棋局矩阵的核心模块。
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging

try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

from .chessboard_detector import DetectionBox

try:
    from ..core.interfaces import ChessboardState, BoardPosition
except ImportError:
    # 创建简单的接口类
    @dataclass
    class BoardPosition:
        x: float
        y: float
        row: int
        col: int
        occupied: bool = False
        piece_type: Optional[str] = None
        confidence: float = 0.0
    
    @dataclass
    class ChessboardState:
        board_matrix: np.ndarray
        board_corners: Tuple
        grid_points: List[List[BoardPosition]]
        selected_position: Optional[Tuple[int, int]]
        validation_result: Dict
        detection_count: Dict
        confidence_stats: Dict


@dataclass
class GridPoint:
    """棋盘网格点"""
    x: float
    y: float
    row: int
    col: int
    occupied: bool = False
    piece_type: Optional[str] = None
    confidence: float = 0.0


class BoardGeometryAnalyzer:
    """棋盘几何分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_board_corners(self, board_detection: DetectionBox) -> Tuple[Tuple[float, float], ...]:
        """
        检测棋盘四个角点
        
        Args:
            board_detection: 棋盘检测框
            
        Returns:
            四个角点坐标 (top_left, top_right, bottom_left, bottom_right)
        """
        # 基于检测框估算角点
        x1, y1, x2, y2 = board_detection.x1, board_detection.y1, board_detection.x2, board_detection.y2
        
        # 添加一些边距以确保包含完整棋盘
        margin_x = (x2 - x1) * 0.05
        margin_y = (y2 - y1) * 0.05
        
        top_left = (x1 - margin_x, y1 - margin_y)
        top_right = (x2 + margin_x, y1 - margin_y)
        bottom_left = (x1 - margin_x, y2 + margin_y)
        bottom_right = (x2 + margin_x, y2 + margin_y)
        
        return top_left, top_right, bottom_left, bottom_right
    
    def generate_grid_points(self, corners: Tuple[Tuple[float, float], ...]) -> List[List[GridPoint]]:
        """
        生成棋盘网格点
        
        Args:
            corners: 棋盘四个角点
            
        Returns:
            10x9的网格点矩阵
        """
        top_left, top_right, bottom_left, bottom_right = corners
        
        grid_points = []
        
        # 象棋棋盘是10行9列的交叉点
        for row in range(10):
            row_points = []
            for col in range(9):
                # 双线性插值计算网格点坐标
                row_ratio = row / 9.0  # 行比例
                col_ratio = col / 8.0  # 列比例
                
                # 上边界插值
                top_x = top_left[0] + (top_right[0] - top_left[0]) * col_ratio
                top_y = top_left[1] + (top_right[1] - top_left[1]) * col_ratio
                
                # 下边界插值
                bottom_x = bottom_left[0] + (bottom_right[0] - bottom_left[0]) * col_ratio
                bottom_y = bottom_left[1] + (bottom_right[1] - bottom_left[1]) * col_ratio
                
                # 垂直插值
                x = top_x + (bottom_x - top_x) * row_ratio
                y = top_y + (bottom_y - top_y) * row_ratio
                
                grid_point = GridPoint(x=x, y=y, row=row, col=col)
                row_points.append(grid_point)
            
            grid_points.append(row_points)
        
        return grid_points
    
    def refine_grid_with_detections(self, 
                                   grid_points: List[List[GridPoint]], 
                                   piece_detections: List[DetectionBox]) -> List[List[GridPoint]]:
        """
        使用棋子检测结果优化网格点位置
        
        Args:
            grid_points: 初始网格点
            piece_detections: 棋子检测结果
            
        Returns:
            优化后的网格点
        """
        if not piece_detections:
            return grid_points
        
        # 提取棋子中心点
        piece_centers = [(det.center[0], det.center[1]) for det in piece_detections]
        
        # 对每个网格点，找到最近的棋子并调整位置
        for row in range(len(grid_points)):
            for col in range(len(grid_points[row])):
                grid_point = grid_points[row][col]
                
                # 计算到所有棋子中心的距离
                distances = [
                    np.sqrt((grid_point.x - px)**2 + (grid_point.y - py)**2)
                    for px, py in piece_centers
                ]
                
                if distances:
                    min_distance = min(distances)
                    # 如果最近的棋子距离合理，微调网格点位置
                    if min_distance < 50:  # 阈值可调
                        closest_idx = distances.index(min_distance)
                        closest_center = piece_centers[closest_idx]
                        
                        # 轻微调整网格点位置
                        adjustment_factor = 0.3
                        grid_point.x += (closest_center[0] - grid_point.x) * adjustment_factor
                        grid_point.y += (closest_center[1] - grid_point.y) * adjustment_factor
        
        return grid_points


class PieceMapper:
    """棋子映射器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 棋子类型到数值的映射
        self.piece_mapping = {
            'red_king': 1, 'red_advisor': 2, 'red_bishop': 3, 'red_knight': 4,
            'red_rook': 5, 'red_cannon': 6, 'red_pawn': 7,
            'black_king': -1, 'black_advisor': -2, 'black_bishop': -3, 'black_knight': -4,
            'black_rook': -5, 'black_cannon': -6, 'black_pawn': -7,
            'selected_piece': 99  # 特殊标记
        }
    
    def map_pieces_to_grid(self, 
                          grid_points: List[List[GridPoint]], 
                          piece_detections: List[DetectionBox],
                          distance_threshold: float = 30.0) -> List[List[GridPoint]]:
        """
        将棋子映射到网格点
        
        Args:
            grid_points: 网格点矩阵
            piece_detections: 棋子检测结果
            distance_threshold: 距离阈值
            
        Returns:
            映射后的网格点矩阵
        """
        # 重置所有网格点的占用状态
        for row in grid_points:
            for point in row:
                point.occupied = False
                point.piece_type = None
                point.confidence = 0.0
        
        # 为每个检测到的棋子找到最近的网格点
        for detection in piece_detections:
            if detection.class_name == 'board':
                continue  # 跳过棋盘检测
            
            best_point = None
            min_distance = float('inf')
            
            # 遍历所有网格点找到最近的
            for row in grid_points:
                for point in row:
                    distance = np.sqrt(
                        (point.x - detection.center[0])**2 + 
                        (point.y - detection.center[1])**2
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_point = point
            
            # 如果距离在阈值内，则映射棋子
            if best_point and min_distance <= distance_threshold:
                # 如果该位置已有棋子，选择置信度更高的
                if not best_point.occupied or detection.confidence > best_point.confidence:
                    best_point.occupied = True
                    best_point.piece_type = detection.class_name
                    best_point.confidence = detection.confidence
                    
                    self.logger.debug(
                        f"映射棋子 {detection.class_name} 到位置 ({best_point.row}, {best_point.col}), "
                        f"距离: {min_distance:.1f}, 置信度: {detection.confidence:.3f}"
                    )
            else:
                self.logger.warning(
                    f"棋子 {detection.class_name} 无法映射到网格，最小距离: {min_distance:.1f}"
                )
        
        return grid_points
    
    def grid_to_matrix(self, grid_points: List[List[GridPoint]]) -> np.ndarray:
        """
        将网格点转换为棋局矩阵
        
        Args:
            grid_points: 网格点矩阵
            
        Returns:
            10x9的棋局矩阵
        """
        matrix = np.zeros((10, 9), dtype=int)
        
        for row in range(len(grid_points)):
            for col in range(len(grid_points[row])):
                point = grid_points[row][col]
                if point.occupied and point.piece_type:
                    piece_value = self.piece_mapping.get(point.piece_type, 0)
                    matrix[row][col] = piece_value
        
        return matrix
    
    def validate_piece_positions(self, matrix: np.ndarray) -> Dict[str, any]:
        """
        验证棋子位置的合理性
        
        Args:
            matrix: 棋局矩阵
            
        Returns:
            验证结果
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'piece_counts': {}
        }
        
        # 统计各类棋子数量
        unique, counts = np.unique(matrix[matrix != 0], return_counts=True)
        piece_counts = dict(zip(unique, counts))
        validation_result['piece_counts'] = piece_counts
        
        # 预期的棋子数量
        expected_counts = {
            1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 5,  # 红方
            -1: 1, -2: 2, -3: 2, -4: 2, -5: 2, -6: 2, -7: 5  # 黑方
        }
        
        # 检查棋子数量
        for piece_type, expected_count in expected_counts.items():
            actual_count = piece_counts.get(piece_type, 0)
            if actual_count > expected_count:
                validation_result['errors'].append(
                    f"棋子类型 {piece_type} 数量过多: {actual_count} > {expected_count}"
                )
                validation_result['valid'] = False
            elif actual_count < expected_count:
                validation_result['warnings'].append(
                    f"棋子类型 {piece_type} 数量不足: {actual_count} < {expected_count}"
                )
        
        # 检查王的位置（应该在九宫格内）
        red_king_positions = np.where(matrix == 1)
        black_king_positions = np.where(matrix == -1)
        
        if len(red_king_positions[0]) > 0:
            king_row, king_col = red_king_positions[0][0], red_king_positions[1][0]
            if not (7 <= king_row <= 9 and 3 <= king_col <= 5):
                validation_result['errors'].append(f"红方帅位置不正确: ({king_row}, {king_col})")
                validation_result['valid'] = False
        
        if len(black_king_positions[0]) > 0:
            king_row, king_col = black_king_positions[0][0], black_king_positions[1][0]
            if not (0 <= king_row <= 2 and 3 <= king_col <= 5):
                validation_result['errors'].append(f"黑方将位置不正确: ({king_row}, {king_col})")
                validation_result['valid'] = False
        
        return validation_result


class BoardMapper:
    """棋盘映射器主类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.geometry_analyzer = BoardGeometryAnalyzer()
        self.piece_mapper = PieceMapper()
    
    def map_detections_to_board(self, 
                               detections: List[DetectionBox],
                               image_shape: Tuple[int, int]) -> ChessboardState:
        """
        将检测结果映射为棋盘状态
        
        Args:
            detections: 检测结果列表
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            棋盘状态
        """
        try:
            # 分离棋盘和棋子检测
            board_detections = [d for d in detections if d.class_name == 'board']
            piece_detections = [d for d in detections if d.class_name != 'board']
            
            self.logger.debug(f"检测到 {len(board_detections)} 个棋盘, {len(piece_detections)} 个棋子")
            
            # 如果没有检测到棋盘，使用整个图像作为棋盘区域
            if not board_detections:
                self.logger.warning("未检测到棋盘，使用整个图像区域")
                board_detection = DetectionBox(
                    x1=0, y1=0, x2=image_shape[1], y2=image_shape[0],
                    confidence=0.5, class_id=-1, class_name='board'
                )
            else:
                # 使用置信度最高的棋盘检测
                board_detection = max(board_detections, key=lambda x: x.confidence)
            
            # 分析棋盘几何结构
            corners = self.geometry_analyzer.detect_board_corners(board_detection)
            grid_points = self.geometry_analyzer.generate_grid_points(corners)
            
            # 使用棋子检测优化网格
            if piece_detections:
                grid_points = self.geometry_analyzer.refine_grid_with_detections(
                    grid_points, piece_detections
                )
            
            # 映射棋子到网格
            grid_points = self.piece_mapper.map_pieces_to_grid(grid_points, piece_detections)
            
            # 转换为矩阵
            board_matrix = self.piece_mapper.grid_to_matrix(grid_points)
            
            # 验证棋局
            validation_result = self.piece_mapper.validate_piece_positions(board_matrix)
            
            # 检测选中的棋子
            selected_pieces = [d for d in detections if d.class_name == 'selected_piece']
            selected_position = None
            if selected_pieces:
                selected_piece = max(selected_pieces, key=lambda x: x.confidence)
                # 找到选中棋子对应的网格位置
                selected_position = self._find_grid_position(selected_piece, grid_points)
            
            # 创建棋盘状态
            chessboard_state = ChessboardState(
                board_matrix=board_matrix,
                board_corners=corners,
                grid_points=[[BoardPosition(p.x, p.y, p.row, p.col, p.occupied, p.piece_type, p.confidence) 
                            for p in row] for row in grid_points],
                selected_position=selected_position,
                validation_result=validation_result,
                detection_count={
                    'board': len(board_detections),
                    'pieces': len(piece_detections),
                    'selected': len(selected_pieces)
                },
                confidence_stats={
                    'mean_confidence': np.mean([d.confidence for d in detections]) if detections else 0.0,
                    'min_confidence': min([d.confidence for d in detections]) if detections else 0.0,
                    'max_confidence': max([d.confidence for d in detections]) if detections else 0.0
                }
            )
            
            self.logger.info(f"棋盘映射完成，检测到 {np.count_nonzero(board_matrix)} 个棋子")
            
            return chessboard_state
            
        except Exception as e:
            self.logger.error(f"棋盘映射失败: {e}")
            raise
    
    def _find_grid_position(self, 
                           detection: DetectionBox, 
                           grid_points: List[List[GridPoint]]) -> Optional[Tuple[int, int]]:
        """
        找到检测框对应的网格位置
        
        Args:
            detection: 检测框
            grid_points: 网格点矩阵
            
        Returns:
            网格位置 (row, col) 或 None
        """
        min_distance = float('inf')
        best_position = None
        
        for row in range(len(grid_points)):
            for col in range(len(grid_points[row])):
                point = grid_points[row][col]
                distance = np.sqrt(
                    (point.x - detection.center[0])**2 + 
                    (point.y - detection.center[1])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_position = (row, col)
        
        return best_position if min_distance < 50 else None
    
    def visualize_grid(self, 
                      image: np.ndarray, 
                      chessboard_state: ChessboardState) -> np.ndarray:
        """
        可视化棋盘网格和映射结果
        
        Args:
            image: 原始图像
            chessboard_state: 棋盘状态
            
        Returns:
            可视化图像
        """
        vis_image = image.copy()
        
        # 绘制棋盘角点
        if chessboard_state.board_corners:
            for corner in chessboard_state.board_corners:
                cv2.circle(vis_image, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)
        
        # 绘制网格点
        if chessboard_state.grid_points:
            for row in chessboard_state.grid_points:
                for point in row:
                    color = (0, 0, 255) if point.occupied else (255, 0, 0)
                    cv2.circle(vis_image, (int(point.x), int(point.y)), 3, color, -1)
                    
                    # 如果有棋子，显示类型
                    if point.occupied and point.piece_type:
                        cv2.putText(vis_image, point.piece_type[:3], 
                                  (int(point.x) + 5, int(point.y) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 绘制网格线
        if chessboard_state.grid_points:
            grid = chessboard_state.grid_points
            
            # 绘制水平线
            for row in range(len(grid)):
                for col in range(len(grid[row]) - 1):
                    pt1 = (int(grid[row][col].x), int(grid[row][col].y))
                    pt2 = (int(grid[row][col + 1].x), int(grid[row][col + 1].y))
                    cv2.line(vis_image, pt1, pt2, (128, 128, 128), 1)
            
            # 绘制垂直线
            for col in range(len(grid[0])):
                for row in range(len(grid) - 1):
                    pt1 = (int(grid[row][col].x), int(grid[row][col].y))
                    pt2 = (int(grid[row + 1][col].x), int(grid[row + 1][col].y))
                    cv2.line(vis_image, pt1, pt2, (128, 128, 128), 1)
        
        # 标记选中位置
        if chessboard_state.selected_position:
            row, col = chessboard_state.selected_position
            if (0 <= row < len(chessboard_state.grid_points) and 
                0 <= col < len(chessboard_state.grid_points[0])):
                point = chessboard_state.grid_points[row][col]
                cv2.circle(vis_image, (int(point.x), int(point.y)), 10, (0, 255, 0), 3)
        
        return vis_image
    
    def export_board_state(self, chessboard_state: ChessboardState) -> Dict[str, any]:
        """
        导出棋盘状态为字典格式
        
        Args:
            chessboard_state: 棋盘状态
            
        Returns:
            棋盘状态字典
        """
        return {
            'board_matrix': chessboard_state.board_matrix.tolist(),
            'selected_position': chessboard_state.selected_position,
            'validation_result': chessboard_state.validation_result,
            'detection_count': chessboard_state.detection_count,
            'confidence_stats': chessboard_state.confidence_stats,
            'piece_positions': self._extract_piece_positions(chessboard_state)
        }
    
    def _extract_piece_positions(self, chessboard_state: ChessboardState) -> Dict[str, List[Tuple[int, int]]]:
        """提取各类棋子的位置"""
        piece_positions = {}
        
        for row in range(chessboard_state.board_matrix.shape[0]):
            for col in range(chessboard_state.board_matrix.shape[1]):
                piece_value = chessboard_state.board_matrix[row, col]
                if piece_value != 0:
                    # 根据数值确定棋子类型
                    piece_type = self._value_to_piece_type(piece_value)
                    if piece_type not in piece_positions:
                        piece_positions[piece_type] = []
                    piece_positions[piece_type].append((row, col))
        
        return piece_positions
    
    def _value_to_piece_type(self, value: int) -> str:
        """将数值转换为棋子类型名称"""
        value_to_name = {
            1: 'red_king', 2: 'red_advisor', 3: 'red_bishop', 4: 'red_knight',
            5: 'red_rook', 6: 'red_cannon', 7: 'red_pawn',
            -1: 'black_king', -2: 'black_advisor', -3: 'black_bishop', -4: 'black_knight',
            -5: 'black_rook', -6: 'black_cannon', -7: 'black_pawn'
        }
        return value_to_name.get(value, f'unknown_{value}')