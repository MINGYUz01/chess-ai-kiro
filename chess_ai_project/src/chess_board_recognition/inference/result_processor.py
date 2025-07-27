"""
结果处理器

处理和验证棋盘检测结果，提供置信度计算和质量评估功能。
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging
from collections import Counter
import time

from .chessboard_detector import DetectionBox
from .board_mapper import BoardMapper
from ..core.interfaces import ChessboardState, DetectionResult


@dataclass
class QualityMetrics:
    """质量评估指标"""
    overall_confidence: float
    detection_completeness: float  # 检测完整性
    position_accuracy: float       # 位置准确性
    board_stability: float         # 棋盘稳定性
    piece_consistency: float       # 棋子一致性
    validation_score: float        # 验证分数
    quality_grade: str            # 质量等级 (A/B/C/D/F)
    
    def to_dict(self) -> Dict[str, any]:
        """转换为字典"""
        return {
            'overall_confidence': self.overall_confidence,
            'detection_completeness': self.detection_completeness,
            'position_accuracy': self.position_accuracy,
            'board_stability': self.board_stability,
            'piece_consistency': self.piece_consistency,
            'validation_score': self.validation_score,
            'quality_grade': self.quality_grade
        }


class StateValidator:
    """棋局状态验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 标准象棋棋子数量
        self.standard_piece_counts = {
            'red_king': 1, 'red_advisor': 2, 'red_bishop': 2, 'red_knight': 2,
            'red_rook': 2, 'red_cannon': 2, 'red_pawn': 5,
            'black_king': 1, 'black_advisor': 2, 'black_bishop': 2, 'black_knight': 2,
            'black_rook': 2, 'black_cannon': 2, 'black_pawn': 5
        }
        
        # 棋子初始位置（用于验证开局）
        self.initial_positions = {
            'red_king': [(9, 4)],
            'red_advisor': [(9, 3), (9, 5)],
            'red_bishop': [(9, 2), (9, 6)],
            'red_knight': [(9, 1), (9, 7)],
            'red_rook': [(9, 0), (9, 8)],
            'red_cannon': [(7, 1), (7, 7)],
            'red_pawn': [(6, 0), (6, 2), (6, 4), (6, 6), (6, 8)],
            'black_king': [(0, 4)],
            'black_advisor': [(0, 3), (0, 5)],
            'black_bishop': [(0, 2), (0, 6)],
            'black_knight': [(0, 1), (0, 7)],
            'black_rook': [(0, 0), (0, 8)],
            'black_cannon': [(2, 1), (2, 7)],
            'black_pawn': [(3, 0), (3, 2), (3, 4), (3, 6), (3, 8)]
        }
    
    def validate_piece_counts(self, chessboard_state: ChessboardState) -> Dict[str, any]:
        """
        验证棋子数量
        
        Args:
            chessboard_state: 棋盘状态
            
        Returns:
            验证结果
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'piece_counts': {},
            'missing_pieces': {},
            'extra_pieces': {}
        }
        
        # 统计实际棋子数量
        actual_counts = {}
        for row in chessboard_state.grid_points:
            for point in row:
                if point.occupied and point.piece_type:
                    piece_type = point.piece_type
                    actual_counts[piece_type] = actual_counts.get(piece_type, 0) + 1
        
        result['piece_counts'] = actual_counts
        
        # 检查每种棋子的数量
        for piece_type, expected_count in self.standard_piece_counts.items():
            actual_count = actual_counts.get(piece_type, 0)
            
            if actual_count < expected_count:
                missing = expected_count - actual_count
                result['missing_pieces'][piece_type] = missing
                result['warnings'].append(f"{piece_type} 缺少 {missing} 个")
            elif actual_count > expected_count:
                extra = actual_count - expected_count
                result['extra_pieces'][piece_type] = extra
                result['errors'].append(f"{piece_type} 多出 {extra} 个")
                result['valid'] = False
        
        return result
    
    def validate_piece_positions(self, chessboard_state: ChessboardState) -> Dict[str, any]:
        """
        验证棋子位置合理性
        
        Args:
            chessboard_state: 棋盘状态
            
        Returns:
            验证结果
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'invalid_positions': []
        }
        
        for row in range(len(chessboard_state.grid_points)):
            for col in range(len(chessboard_state.grid_points[row])):
                point = chessboard_state.grid_points[row][col]
                
                if point.occupied and point.piece_type:
                    # 验证王的位置（必须在九宫格内）
                    if point.piece_type == 'red_king':
                        if not (7 <= row <= 9 and 3 <= col <= 5):
                            result['errors'].append(f"红方帅位置错误: ({row}, {col})")
                            result['invalid_positions'].append((row, col, point.piece_type))
                            result['valid'] = False
                    
                    elif point.piece_type == 'black_king':
                        if not (0 <= row <= 2 and 3 <= col <= 5):
                            result['errors'].append(f"黑方将位置错误: ({row}, {col})")
                            result['invalid_positions'].append((row, col, point.piece_type))
                            result['valid'] = False
                    
                    # 验证士的位置（必须在九宫格内）
                    elif point.piece_type in ['red_advisor', 'black_advisor']:
                        if point.piece_type == 'red_advisor':
                            valid_positions = [(7, 3), (7, 5), (8, 4), (9, 3), (9, 5)]
                        else:
                            valid_positions = [(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)]
                        
                        if (row, col) not in valid_positions:
                            result['warnings'].append(f"{point.piece_type} 位置可能错误: ({row}, {col})")
                    
                    # 验证相/象的位置
                    elif point.piece_type in ['red_bishop', 'black_bishop']:
                        if point.piece_type == 'red_bishop':
                            valid_positions = [(5, 2), (5, 6), (7, 0), (7, 4), (7, 8), (9, 2), (9, 6)]
                        else:
                            valid_positions = [(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)]
                        
                        if (row, col) not in valid_positions:
                            result['warnings'].append(f"{point.piece_type} 位置可能错误: ({row}, {col})")
                    
                    # 验证兵/卒不能后退
                    elif point.piece_type == 'red_pawn':
                        if row > 6:  # 红兵不应该在己方区域后退
                            result['warnings'].append(f"红兵位置异常: ({row}, {col})")
                    
                    elif point.piece_type == 'black_pawn':
                        if row < 3:  # 黑卒不应该在己方区域后退
                            result['warnings'].append(f"黑卒位置异常: ({row}, {col})")
        
        return result
    
    def validate_board_integrity(self, chessboard_state: ChessboardState) -> Dict[str, any]:
        """
        验证棋盘完整性
        
        Args:
            chessboard_state: 棋盘状态
            
        Returns:
            验证结果
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'board_detected': False,
            'grid_completeness': 0.0
        }
        
        # 检查是否检测到棋盘
        if chessboard_state.detection_count.get('board', 0) > 0:
            result['board_detected'] = True
        else:
            result['warnings'].append("未检测到棋盘边界")
        
        # 检查网格点完整性
        total_points = 10 * 9
        valid_points = 0
        
        for row in chessboard_state.grid_points:
            for point in row:
                if point.x > 0 and point.y > 0:  # 有效坐标
                    valid_points += 1
        
        result['grid_completeness'] = valid_points / total_points
        
        if result['grid_completeness'] < 0.9:
            result['warnings'].append(f"网格完整性不足: {result['grid_completeness']:.1%}")
        
        return result


class ConfidenceCalculator:
    """置信度计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_overall_confidence(self, chessboard_state: ChessboardState) -> float:
        """
        计算整体置信度
        
        Args:
            chessboard_state: 棋盘状态
            
        Returns:
            整体置信度 (0-1)
        """
        if not chessboard_state.grid_points:
            return 0.0
        
        # 收集所有有效检测的置信度
        confidences = []
        for row in chessboard_state.grid_points:
            for point in row:
                if point.occupied and point.confidence > 0:
                    confidences.append(point.confidence)
        
        if not confidences:
            return 0.0
        
        # 使用加权平均，给高置信度更多权重
        weights = np.array(confidences)
        weighted_confidence = np.average(confidences, weights=weights)
        
        return float(weighted_confidence)
    
    def calculate_detection_completeness(self, chessboard_state: ChessboardState) -> float:
        """
        计算检测完整性
        
        Args:
            chessboard_state: 棋盘状态
            
        Returns:
            完整性分数 (0-1)
        """
        # 统计检测到的棋子数量
        detected_pieces = 0
        for row in chessboard_state.grid_points:
            for point in row:
                if point.occupied:
                    detected_pieces += 1
        
        # 标准象棋总共32个棋子
        expected_pieces = 32
        completeness = min(detected_pieces / expected_pieces, 1.0)
        
        return completeness
    
    def calculate_position_accuracy(self, chessboard_state: ChessboardState) -> float:
        """
        计算位置准确性
        
        Args:
            chessboard_state: 棋盘状态
            
        Returns:
            准确性分数 (0-1)
        """
        # 基于网格点的位置偏差计算准确性
        total_points = 0
        accurate_points = 0
        
        for row in chessboard_state.grid_points:
            for point in row:
                if point.occupied:
                    total_points += 1
                    # 这里可以添加更复杂的位置准确性计算
                    # 目前简单地基于置信度判断
                    if point.confidence > 0.7:
                        accurate_points += 1
        
        if total_points == 0:
            return 0.0
        
        return accurate_points / total_points


class ResultProcessor:
    """结果处理器主类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.board_mapper = BoardMapper()
        self.state_validator = StateValidator()
        self.confidence_calculator = ConfidenceCalculator()
        
        # 历史状态用于稳定性分析
        self.history_states = []
        self.max_history_size = 10
    
    def process_detection_result(self, 
                                detection_result: DetectionResult,
                                enable_validation: bool = True) -> ChessboardState:
        """
        处理检测结果
        
        Args:
            detection_result: 原始检测结果
            enable_validation: 是否启用验证
            
        Returns:
            处理后的棋盘状态
        """
        start_time = time.time()
        
        try:
            # 映射检测结果到棋盘状态
            image_shape = detection_result.image.shape[:2]  # (height, width)
            chessboard_state = self.board_mapper.map_detections_to_board(
                detection_result.detections, image_shape
            )
            
            # 如果启用验证，进行状态验证
            if enable_validation:
                validation_results = self._validate_chessboard_state(chessboard_state)
                chessboard_state.validation_result.update(validation_results)
            
            # 计算质量指标
            quality_metrics = self._calculate_quality_metrics(chessboard_state)
            chessboard_state.confidence_stats['quality_metrics'] = quality_metrics.to_dict()
            
            # 更新历史状态
            self._update_history(chessboard_state)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            chessboard_state.confidence_stats['processing_time'] = processing_time
            
            self.logger.info(f"结果处理完成，用时: {processing_time:.3f}s, "
                           f"质量等级: {quality_metrics.quality_grade}")
            
            return chessboard_state
            
        except Exception as e:
            self.logger.error(f"结果处理失败: {e}")
            raise
    
    def _validate_chessboard_state(self, chessboard_state: ChessboardState) -> Dict[str, any]:
        """验证棋盘状态"""
        validation_results = {}
        
        # 验证棋子数量
        piece_count_result = self.state_validator.validate_piece_counts(chessboard_state)
        validation_results['piece_counts'] = piece_count_result
        
        # 验证棋子位置
        position_result = self.state_validator.validate_piece_positions(chessboard_state)
        validation_results['positions'] = position_result
        
        # 验证棋盘完整性
        integrity_result = self.state_validator.validate_board_integrity(chessboard_state)
        validation_results['board_integrity'] = integrity_result
        
        # 综合验证结果
        overall_valid = (piece_count_result['valid'] and 
                        position_result['valid'] and 
                        integrity_result['valid'])
        
        validation_results['overall_valid'] = overall_valid
        validation_results['total_errors'] = (len(piece_count_result['errors']) + 
                                            len(position_result['errors']) + 
                                            len(integrity_result['errors']))
        validation_results['total_warnings'] = (len(piece_count_result['warnings']) + 
                                               len(position_result['warnings']) + 
                                               len(integrity_result['warnings']))
        
        return validation_results
    
    def _calculate_quality_metrics(self, chessboard_state: ChessboardState) -> QualityMetrics:
        """计算质量指标"""
        # 计算各项指标
        overall_confidence = self.confidence_calculator.calculate_overall_confidence(chessboard_state)
        detection_completeness = self.confidence_calculator.calculate_detection_completeness(chessboard_state)
        position_accuracy = self.confidence_calculator.calculate_position_accuracy(chessboard_state)
        
        # 计算棋盘稳定性（基于历史状态）
        board_stability = self._calculate_board_stability(chessboard_state)
        
        # 计算棋子一致性
        piece_consistency = self._calculate_piece_consistency(chessboard_state)
        
        # 计算验证分数
        validation_score = self._calculate_validation_score(chessboard_state)
        
        # 计算综合质量等级
        quality_grade = self._calculate_quality_grade(
            overall_confidence, detection_completeness, position_accuracy,
            board_stability, piece_consistency, validation_score
        )
        
        return QualityMetrics(
            overall_confidence=overall_confidence,
            detection_completeness=detection_completeness,
            position_accuracy=position_accuracy,
            board_stability=board_stability,
            piece_consistency=piece_consistency,
            validation_score=validation_score,
            quality_grade=quality_grade
        )
    
    def _calculate_board_stability(self, chessboard_state: ChessboardState) -> float:
        """计算棋盘稳定性"""
        if len(self.history_states) < 2:
            return 1.0  # 没有历史数据时认为稳定
        
        # 比较当前状态与最近的历史状态
        current_matrix = chessboard_state.board_matrix
        last_matrix = self.history_states[-1].board_matrix
        
        # 计算矩阵差异
        differences = np.sum(current_matrix != last_matrix)
        total_positions = current_matrix.size
        
        stability = 1.0 - (differences / total_positions)
        return max(0.0, stability)
    
    def _calculate_piece_consistency(self, chessboard_state: ChessboardState) -> float:
        """计算棋子一致性"""
        # 检查同类棋子的置信度一致性
        piece_confidences = {}
        
        for row in chessboard_state.grid_points:
            for point in row:
                if point.occupied and point.piece_type:
                    piece_type = point.piece_type
                    if piece_type not in piece_confidences:
                        piece_confidences[piece_type] = []
                    piece_confidences[piece_type].append(point.confidence)
        
        if not piece_confidences:
            return 0.0
        
        # 计算每种棋子置信度的标准差
        consistency_scores = []
        for piece_type, confidences in piece_confidences.items():
            if len(confidences) > 1:
                std_dev = np.std(confidences)
                consistency = 1.0 - min(std_dev, 1.0)  # 标准差越小，一致性越高
                consistency_scores.append(consistency)
            else:
                consistency_scores.append(1.0)  # 单个棋子认为完全一致
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_validation_score(self, chessboard_state: ChessboardState) -> float:
        """计算验证分数"""
        validation_result = chessboard_state.validation_result
        
        if not validation_result:
            return 0.0
        
        # 基于错误和警告数量计算分数
        total_errors = validation_result.get('total_errors', 0)
        total_warnings = validation_result.get('total_warnings', 0)
        
        # 错误权重更高
        penalty = total_errors * 0.2 + total_warnings * 0.1
        score = max(0.0, 1.0 - penalty)
        
        return score
    
    def _calculate_quality_grade(self, *scores) -> str:
        """计算质量等级"""
        average_score = np.mean(scores)
        
        if average_score >= 0.9:
            return 'A'
        elif average_score >= 0.8:
            return 'B'
        elif average_score >= 0.7:
            return 'C'
        elif average_score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _update_history(self, chessboard_state: ChessboardState):
        """更新历史状态"""
        self.history_states.append(chessboard_state)
        
        # 限制历史记录大小
        if len(self.history_states) > self.max_history_size:
            self.history_states.pop(0)
    
    def get_processing_statistics(self) -> Dict[str, any]:
        """获取处理统计信息"""
        if not self.history_states:
            return {}
        
        # 统计最近的处理结果
        recent_states = self.history_states[-5:]  # 最近5个状态
        
        quality_grades = []
        confidence_scores = []
        processing_times = []
        
        for state in recent_states:
            quality_metrics = state.confidence_stats.get('quality_metrics', {})
            quality_grades.append(quality_metrics.get('quality_grade', 'F'))
            confidence_scores.append(quality_metrics.get('overall_confidence', 0.0))
            processing_times.append(state.confidence_stats.get('processing_time', 0.0))
        
        # 统计质量等级分布
        grade_counts = Counter(quality_grades)
        
        return {
            'total_processed': len(self.history_states),
            'recent_average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'recent_average_processing_time': np.mean(processing_times) if processing_times else 0.0,
            'quality_grade_distribution': dict(grade_counts),
            'stability_trend': self._calculate_stability_trend()
        }
    
    def _calculate_stability_trend(self) -> str:
        """计算稳定性趋势"""
        if len(self.history_states) < 3:
            return 'insufficient_data'
        
        # 计算最近几个状态的稳定性变化
        recent_stabilities = []
        for i in range(1, min(5, len(self.history_states))):
            state = self.history_states[-i]
            quality_metrics = state.confidence_stats.get('quality_metrics', {})
            stability = quality_metrics.get('board_stability', 0.0)
            recent_stabilities.append(stability)
        
        if len(recent_stabilities) < 2:
            return 'stable'
        
        # 计算趋势
        trend = np.polyfit(range(len(recent_stabilities)), recent_stabilities, 1)[0]
        
        if trend > 0.05:
            return 'improving'
        elif trend < -0.05:
            return 'declining'
        else:
            return 'stable'
    
    def export_processing_report(self, filepath: str):
        """导出处理报告"""
        report = {
            'timestamp': time.time(),
            'statistics': self.get_processing_statistics(),
            'recent_states': []
        }
        
        # 添加最近状态的详细信息
        for state in self.history_states[-3:]:  # 最近3个状态
            state_info = {
                'board_matrix': state.board_matrix.tolist(),
                'detection_count': state.detection_count,
                'confidence_stats': state.confidence_stats,
                'validation_result': state.validation_result
            }
            report['recent_states'].append(state_info)
        
        # 保存报告
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"处理报告已保存到: {filepath}")
    
    def clear_history(self):
        """清空历史记录"""
        self.history_states.clear()
        self.logger.info("历史记录已清空")