"""
游戏接口测试

测试游戏接口和会话管理功能。
"""

import pytest
import tempfile
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from chess_ai_project.src.chinese_chess_ai_engine.inference_interface import (
    GameInterface, GameSession, GameConfig, MoveRecord,
    GameState, GameResult, PlayerType,
    GameInterfaceError, SessionNotFoundError, InvalidMoveError, GameStateError
)
from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import ChessBoard, Move


class TestGameConfig:
    """测试游戏配置类"""
    
    def test_game_config_creation(self):
        """测试游戏配置创建"""
        config = GameConfig(
            red_player_type=PlayerType.HUMAN,
            black_player_type=PlayerType.AI,
            ai_difficulty=7,
            time_limit_per_move=30.0,
            total_time_limit=1800.0
        )
        
        assert config.red_player_type == PlayerType.HUMAN
        assert config.black_player_type == PlayerType.AI
        assert config.ai_difficulty == 7
        assert config.time_limit_per_move == 30.0
        assert config.total_time_limit == 1800.0
        assert config.ai_config is not None
        assert config.ai_config.difficulty_level == 7
    
    def test_game_config_defaults(self):
        """测试游戏配置默认值"""
        config = GameConfig()
        
        assert config.red_player_type == PlayerType.HUMAN
        assert config.black_player_type == PlayerType.AI
        assert config.ai_difficulty == 5
        assert config.time_limit_per_move is None
        assert config.total_time_limit is None
        assert config.allow_undo == True
        assert config.max_moves == 300
        assert config.draw_by_repetition == True
        assert config.save_game_record == True
        assert config.record_analysis == False


class TestMoveRecord:
    """测试走法记录类"""
    
    def test_move_record_creation(self):
        """测试走法记录创建"""
        move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        timestamp = datetime.now()
        
        record = MoveRecord(
            move=move,
            player=1,
            timestamp=timestamp,
            time_used=2.5,
            comment="测试走法"
        )
        
        assert record.move == move
        assert record.player == 1
        assert record.timestamp == timestamp
        assert record.time_used == 2.5
        assert record.comment == "测试走法"
        assert record.analysis is None
    
    def test_move_record_serialization(self):
        """测试走法记录序列化"""
        move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        timestamp = datetime.now()
        
        record = MoveRecord(
            move=move,
            player=1,
            timestamp=timestamp,
            time_used=2.5
        )
        
        # 转换为字典
        data = record.to_dict()
        assert 'move' in data
        assert 'player' in data
        assert 'timestamp' in data
        assert 'time_used' in data
        
        # 从字典重建
        rebuilt_record = MoveRecord.from_dict(data)
        assert rebuilt_record.move.from_pos == move.from_pos
        assert rebuilt_record.move.to_pos == move.to_pos
        assert rebuilt_record.player == record.player
        assert rebuilt_record.time_used == record.time_used


class TestGameSession:
    """测试游戏会话类"""
    
    def test_game_session_creation(self):
        """测试游戏会话创建"""
        config = GameConfig()
        session = GameSession(config=config)
        
        assert session.session_id is not None
        assert len(session.session_id) > 0
        assert session.config == config
        assert isinstance(session.board, ChessBoard)
        assert session.state == GameState.WAITING
        assert session.result == GameResult.ONGOING
        assert session.current_player == 1
        assert session.total_moves == 0
        assert len(session.move_history) == 0
        assert len(session.board_history) == 1  # 初始棋局
        assert session.created_at is not None
    
    def test_game_session_serialization(self):
        """测试游戏会话序列化"""
        config = GameConfig()
        session = GameSession(config=config)
        
        # 转换为字典
        data = session.to_dict()
        assert 'session_id' in data
        assert 'config' in data
        assert 'board' in data
        assert 'state' in data
        assert 'result' in data
        
        # 验证配置序列化
        config_data = data['config']
        assert config_data['red_player_type'] == PlayerType.HUMAN.value
        assert config_data['black_player_type'] == PlayerType.AI.value


class TestGameInterface:
    """测试游戏接口类"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.game_interface = GameInterface(
            ai_model_path="test_model.pth",
            save_directory=self.temp_dir
        )
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_game_interface_initialization(self):
        """测试游戏接口初始化"""
        assert self.game_interface.ai_model_path == "test_model.pth"
        assert self.game_interface.save_directory == Path(self.temp_dir)
        assert self.game_interface.current_session is None
        assert len(self.game_interface.sessions) == 0
    
    def test_create_session(self):
        """测试创建会话"""
        config = GameConfig(ai_difficulty=3)
        session_id = self.game_interface.create_session(config)
        
        assert session_id is not None
        assert session_id in self.game_interface.sessions
        assert self.game_interface.current_session is not None
        assert self.game_interface.current_session.session_id == session_id
        assert self.game_interface.current_session.config.ai_difficulty == 3
    
    def test_create_session_with_defaults(self):
        """测试使用默认配置创建会话"""
        session_id = self.game_interface.create_session()
        
        assert session_id is not None
        session = self.game_interface.sessions[session_id]
        assert session.config.ai_difficulty == 5  # 默认难度
    
    def test_start_game(self):
        """测试开始游戏"""
        session_id = self.game_interface.create_session()
        
        # 开始游戏
        success = self.game_interface.start_game(session_id)
        assert success == True
        
        session = self.game_interface.sessions[session_id]
        assert session.state == GameState.PLAYING
        assert session.start_time is not None
        assert session.last_move_time is not None
    
    def test_start_game_invalid_state(self):
        """测试在错误状态下开始游戏"""
        session_id = self.game_interface.create_session()
        
        # 先开始游戏
        self.game_interface.start_game(session_id)
        
        # 再次尝试开始游戏应该失败
        success = self.game_interface.start_game(session_id)
        assert success == False
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.game_interface.ChessAI')
    def test_make_move_success(self, mock_chess_ai_class):
        """测试成功执行走法"""
        # 模拟ChessAI
        mock_ai = Mock()
        mock_chess_ai_class.return_value = mock_ai
        
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        session = self.game_interface.sessions[session_id]
        
        # 模拟合法走法
        test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        with patch.object(session.board, 'get_legal_moves') as mock_get_legal_moves:
            mock_get_legal_moves.return_value = [test_move]
            
            with patch.object(session.board, 'make_move') as mock_make_move:
                # 执行走法
                success, error = self.game_interface.make_move(test_move, session_id)
                
                assert success == True
                assert error == ""
                assert len(session.move_history) == 1
                assert session.total_moves == 1
                assert session.current_player == -1  # 切换到黑方
                mock_make_move.assert_called_once_with(test_move)
    
    def test_make_move_illegal(self):
        """测试执行非法走法"""
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        session = self.game_interface.sessions[session_id]
        
        # 模拟空的合法走法列表
        with patch.object(session.board, 'get_legal_moves') as mock_get_legal_moves:
            mock_get_legal_moves.return_value = []
            
            test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
            success, error = self.game_interface.make_move(test_move, session_id)
            
            assert success == False
            assert "非法走法" in error
            assert len(session.move_history) == 0
    
    def test_make_move_wrong_player(self):
        """测试错误玩家执行走法"""
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        session = self.game_interface.sessions[session_id]
        # 设置会话为黑方回合，但棋盘仍然是红方回合
        session.current_player = -1
        # 棋盘的current_player保持为1（红方）
        
        test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        success, error = self.game_interface.make_move(test_move, session_id)
        
        assert success == False
        assert "不是当前玩家的回合" in error
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.game_interface.ChessAI')
    def test_get_ai_move(self, mock_chess_ai_class):
        """测试获取AI走法"""
        # 模拟ChessAI
        mock_ai = Mock()
        test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        mock_ai.get_best_move.return_value = test_move
        mock_chess_ai_class.return_value = mock_ai
        
        # 创建AI vs AI的会话
        config = GameConfig(
            red_player_type=PlayerType.AI,
            black_player_type=PlayerType.AI
        )
        session_id = self.game_interface.create_session(config)
        self.game_interface.start_game(session_id)
        
        # 获取AI走法
        ai_move = self.game_interface.get_ai_move(session_id)
        
        assert ai_move == test_move
        mock_ai.get_best_move.assert_called_once()
    
    def test_get_ai_move_human_player(self):
        """测试人类玩家时获取AI走法"""
        # 创建人类 vs AI的会话
        config = GameConfig(
            red_player_type=PlayerType.HUMAN,  # 红方是人类
            black_player_type=PlayerType.AI
        )
        session_id = self.game_interface.create_session(config)
        self.game_interface.start_game(session_id)
        
        # 当前是红方（人类）回合，不应该返回AI走法
        ai_move = self.game_interface.get_ai_move(session_id)
        assert ai_move is None
    
    def test_undo_move(self):
        """测试撤销走法"""
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        session = self.game_interface.sessions[session_id]
        
        # 先执行一个走法
        test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        with patch.object(session.board, 'get_legal_moves') as mock_get_legal_moves:
            mock_get_legal_moves.return_value = [test_move]
            
            with patch.object(session.board, 'make_move'):
                self.game_interface.make_move(test_move, session_id)
        
        # 撤销走法
        with patch.object(session.board, 'undo_move') as mock_undo_move:
            success = self.game_interface.undo_move(session_id)
            
            assert success == True
            assert len(session.move_history) == 0
            assert session.total_moves == 0
            assert session.current_player == 1  # 恢复到红方
            mock_undo_move.assert_called_once()
    
    def test_undo_move_not_allowed(self):
        """测试不允许撤销时的情况"""
        config = GameConfig(allow_undo=False)
        session_id = self.game_interface.create_session(config)
        
        success = self.game_interface.undo_move(session_id)
        assert success == False
    
    def test_undo_move_no_history(self):
        """测试没有历史记录时撤销走法"""
        session_id = self.game_interface.create_session()
        
        success = self.game_interface.undo_move(session_id)
        assert success == False
    
    def test_pause_and_resume_game(self):
        """测试暂停和恢复游戏"""
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        session = self.game_interface.sessions[session_id]
        
        # 暂停游戏
        success = self.game_interface.pause_game(session_id)
        assert success == True
        assert session.state == GameState.PAUSED
        
        # 恢复游戏
        success = self.game_interface.resume_game(session_id)
        assert success == True
        assert session.state == GameState.PLAYING
    
    def test_resign_game(self):
        """测试认输"""
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        session = self.game_interface.sessions[session_id]
        
        # 红方认输
        success = self.game_interface.resign_game(1, session_id)
        assert success == True
        assert session.state == GameState.FINISHED
        assert session.result == GameResult.BLACK_WIN
        assert session.finished_at is not None
    
    def test_get_game_status(self):
        """测试获取游戏状态"""
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        status = self.game_interface.get_game_status(session_id)
        
        assert 'session_id' in status
        assert 'state' in status
        assert 'result' in status
        assert 'current_player' in status
        assert 'board_fen' in status
        assert 'board_matrix' in status
        assert 'legal_moves' in status
        assert 'total_moves' in status
        assert 'can_undo' in status
        
        assert status['session_id'] == session_id
        assert status['state'] == GameState.PLAYING.value
        assert status['result'] == GameResult.ONGOING.value
        assert status['current_player'] == 1
        assert status['total_moves'] == 0
    
    def test_get_move_history(self):
        """测试获取走法历史"""
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        # 初始状态下应该没有走法历史
        history = self.game_interface.get_move_history(session_id)
        assert len(history) == 0
        
        # 执行一个走法后再检查
        session = self.game_interface.sessions[session_id]
        test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        
        with patch.object(session.board, 'get_legal_moves') as mock_get_legal_moves:
            mock_get_legal_moves.return_value = [test_move]
            
            with patch.object(session.board, 'make_move'):
                self.game_interface.make_move(test_move, session_id)
        
        history = self.game_interface.get_move_history(session_id)
        assert len(history) == 1
        assert 'move_number' in history[0]
        assert 'move' in history[0]
        assert 'player' in history[0]
        assert 'time_used' in history[0]
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.game_interface.ChessAI')
    def test_analyze_position(self, mock_chess_ai_class):
        """测试位置分析"""
        # 模拟ChessAI和分析结果
        mock_ai = Mock()
        mock_analysis = Mock()
        mock_analysis.evaluation = 0.5
        mock_ai.analyze_position.return_value = mock_analysis
        mock_chess_ai_class.return_value = mock_ai
        
        session_id = self.game_interface.create_session()
        
        result = self.game_interface.analyze_position(session_id)
        
        assert result == mock_analysis
        mock_ai.analyze_position.assert_called_once()
    
    def test_save_and_load_session(self):
        """测试保存和加载会话"""
        # 创建会话
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        # 保存会话
        success = self.game_interface.save_session(session_id)
        assert success == True
        
        # 检查文件是否存在
        save_files = list(Path(self.temp_dir).glob("game_*.json"))
        assert len(save_files) > 0
        
        # 加载会话
        loaded_session_id = self.game_interface.load_session(str(save_files[0]))
        assert loaded_session_id is not None
        assert loaded_session_id in self.game_interface.sessions
    
    def test_list_sessions(self):
        """测试列出会话"""
        # 创建多个会话
        session_id1 = self.game_interface.create_session()
        session_id2 = self.game_interface.create_session()
        
        sessions_info = self.game_interface.list_sessions()
        
        assert len(sessions_info) == 2
        session_ids = [info['session_id'] for info in sessions_info]
        assert session_id1 in session_ids
        assert session_id2 in session_ids
    
    def test_delete_session(self):
        """测试删除会话"""
        session_id = self.game_interface.create_session()
        
        # 确认会话存在
        assert session_id in self.game_interface.sessions
        assert self.game_interface.current_session.session_id == session_id
        
        # 删除会话
        success = self.game_interface.delete_session(session_id)
        assert success == True
        assert session_id not in self.game_interface.sessions
        assert self.game_interface.current_session is None
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        # 创建会话
        session_id = self.game_interface.create_session()
        
        # 获取单个会话统计
        stats = self.game_interface.get_statistics(session_id)
        assert 'session_id' in stats
        assert 'state' in stats
        assert 'total_moves' in stats
        
        # 获取全局统计
        global_stats = self.game_interface.get_statistics()
        assert 'total_sessions' in global_stats
        assert 'finished_sessions' in global_stats
        assert 'active_sessions' in global_stats
        assert global_stats['total_sessions'] == 1
    
    def test_session_not_found_error(self):
        """测试会话不存在的错误处理"""
        with pytest.raises(ValueError, match="会话不存在"):
            self.game_interface.get_game_status("nonexistent_session")
    
    def test_no_current_session_error(self):
        """测试没有当前会话的错误处理"""
        with pytest.raises(ValueError, match="没有当前会话"):
            self.game_interface.get_game_status()
    
    def test_game_end_conditions(self):
        """测试游戏结束条件检查"""
        session_id = self.game_interface.create_session()
        self.game_interface.start_game(session_id)
        
        session = self.game_interface.sessions[session_id]
        
        # 模拟将死状态
        with patch.object(session.board, 'is_checkmate') as mock_checkmate:
            mock_checkmate.return_value = True
            
            # 执行一个走法来触发游戏结束检查
            test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
            with patch.object(session.board, 'get_legal_moves') as mock_get_legal_moves:
                mock_get_legal_moves.return_value = [test_move]
                
                with patch.object(session.board, 'make_move'):
                    self.game_interface.make_move(test_move, session_id)
            
            # 检查游戏是否结束
            assert session.state == GameState.FINISHED
            # 执行走法后current_player已经切换到黑方(-1)，所以如果黑方被将死，红方获胜
            assert session.result == GameResult.RED_WIN


if __name__ == '__main__':
    pytest.main([__file__])