"""
API服务器测试

测试RESTful API接口的功能。
"""

import pytest
import asyncio
import json
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from pathlib import Path

from chess_ai_project.src.chinese_chess_ai_engine.inference_interface import (
    APIServer, create_api_server, PlayerType
)


class TestAPIServer:
    """测试API服务器类"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建API服务器
        self.api_server = create_api_server(
            ai_model_path="test_model.pth",
            save_directory=self.temp_dir,
            api_keys=None  # 不启用认证
        )
        
        # 创建测试客户端
        self.client = TestClient(self.api_server.app)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "status" in data["data"]
        assert data["data"]["status"] == "healthy"
    
    def test_create_session(self):
        """测试创建会话"""
        request_data = {
            "red_player_type": "human",
            "black_player_type": "ai",
            "ai_difficulty": 5
        }
        
        response = self.client.post("/sessions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "session_id" in data["data"]
        assert len(data["data"]["session_id"]) > 0
    
    def test_create_session_with_validation_error(self):
        """测试创建会话时的验证错误"""
        request_data = {
            "ai_difficulty": 15  # 超出范围
        }
        
        response = self.client.post("/sessions", json=request_data)
        
        assert response.status_code == 422  # 验证错误
    
    def test_list_sessions(self):
        """测试获取会话列表"""
        # 先创建一个会话
        self.client.post("/sessions", json={})
        
        response = self.client.get("/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "sessions" in data["data"]
        assert "count" in data["data"]
        assert data["data"]["count"] >= 1
    
    def test_get_session_status(self):
        """测试获取会话状态"""
        # 先创建一个会话
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        
        response = self.client.get(f"/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "session_id" in data["data"]
        assert data["data"]["session_id"] == session_id
    
    def test_get_nonexistent_session(self):
        """测试获取不存在的会话"""
        response = self.client.get("/sessions/nonexistent")
        
        assert response.status_code == 404
    
    def test_delete_session(self):
        """测试删除会话"""
        # 先创建一个会话
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        
        # 删除会话
        response = self.client.delete(f"/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        
        # 验证会话已被删除
        get_response = self.client.get(f"/sessions/{session_id}")
        assert get_response.status_code == 404
    
    def test_start_game(self):
        """测试开始游戏"""
        # 先创建一个会话
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        
        response = self.client.post(f"/sessions/{session_id}/start")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.game_interface.ChessAI')
    def test_make_move(self, mock_chess_ai_class):
        """测试执行走法"""
        # 模拟ChessAI
        mock_ai = Mock()
        mock_chess_ai_class.return_value = mock_ai
        
        # 先创建并开始游戏
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        self.client.post(f"/sessions/{session_id}/start")
        
        # 模拟合法走法
        with patch.object(self.api_server.game_interface, 'make_move') as mock_make_move:
            mock_make_move.return_value = (True, "")
            
            with patch.object(self.api_server.game_interface, 'get_game_status') as mock_get_status:
                mock_get_status.return_value = {
                    "session_id": session_id,
                    "state": "playing",
                    "current_player": -1
                }
                
                request_data = {"move": "a0b1"}
                response = self.client.post(f"/sessions/{session_id}/moves", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] == True
    
    def test_make_invalid_move(self):
        """测试执行非法走法"""
        # 先创建并开始游戏
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        self.client.post(f"/sessions/{session_id}/start")
        
        # 模拟非法走法
        with patch.object(self.api_server.game_interface, 'make_move') as mock_make_move:
            mock_make_move.return_value = (False, "非法走法")
            
            request_data = {"move": "invalid"}
            response = self.client.post(f"/sessions/{session_id}/moves", json=request_data)
            
            assert response.status_code == 400
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.game_interface.ChessAI')
    def test_get_ai_move(self, mock_chess_ai_class):
        """测试获取AI走法"""
        # 模拟ChessAI
        mock_ai = Mock()
        mock_chess_ai_class.return_value = mock_ai
        
        # 先创建并开始游戏
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        self.client.post(f"/sessions/{session_id}/start")
        
        # 模拟AI走法
        from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import Move
        test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        
        with patch.object(self.api_server.game_interface, 'get_ai_move') as mock_get_ai_move:
            mock_get_ai_move.return_value = test_move
            
            response = self.client.get(f"/sessions/{session_id}/ai-move")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "move" in data["data"]
    
    def test_get_ai_move_failure(self):
        """测试获取AI走法失败"""
        # 先创建并开始游戏
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        self.client.post(f"/sessions/{session_id}/start")
        
        # 模拟AI走法失败
        with patch.object(self.api_server.game_interface, 'get_ai_move') as mock_get_ai_move:
            mock_get_ai_move.return_value = None
            
            response = self.client.get(f"/sessions/{session_id}/ai-move")
            
            assert response.status_code == 400
    
    def test_undo_move(self):
        """测试撤销走法"""
        # 先创建并开始游戏
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        self.client.post(f"/sessions/{session_id}/start")
        
        # 模拟撤销成功
        with patch.object(self.api_server.game_interface, 'undo_move') as mock_undo_move:
            mock_undo_move.return_value = True
            
            with patch.object(self.api_server.game_interface, 'get_game_status') as mock_get_status:
                mock_get_status.return_value = {
                    "session_id": session_id,
                    "state": "playing"
                }
                
                response = self.client.post(f"/sessions/{session_id}/undo")
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] == True
    
    def test_pause_and_resume_game(self):
        """测试暂停和恢复游戏"""
        # 先创建并开始游戏
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        self.client.post(f"/sessions/{session_id}/start")
        
        # 暂停游戏
        with patch.object(self.api_server.game_interface, 'pause_game') as mock_pause:
            mock_pause.return_value = True
            
            response = self.client.post(f"/sessions/{session_id}/pause")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
        
        # 恢复游戏
        with patch.object(self.api_server.game_interface, 'resume_game') as mock_resume:
            mock_resume.return_value = True
            
            response = self.client.post(f"/sessions/{session_id}/resume")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    def test_resign_game(self):
        """测试认输"""
        # 先创建并开始游戏
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        self.client.post(f"/sessions/{session_id}/start")
        
        # 认输
        with patch.object(self.api_server.game_interface, 'resign_game') as mock_resign:
            mock_resign.return_value = True
            
            response = self.client.post(f"/sessions/{session_id}/resign?player=1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.game_interface.ChessAI')
    def test_analyze_position(self, mock_chess_ai_class):
        """测试位置分析"""
        # 模拟ChessAI
        mock_ai = Mock()
        mock_chess_ai_class.return_value = mock_ai
        
        # 先创建并开始游戏
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        self.client.post(f"/sessions/{session_id}/start")
        
        # 模拟分析结果
        from chess_ai_project.src.chinese_chess_ai_engine.rules_engine import Move
        test_move = Move(from_pos=(0, 4), to_pos=(1, 4), piece=1)
        
        mock_analysis = Mock()
        mock_analysis.best_move = test_move
        mock_analysis.evaluation = 0.5
        mock_analysis.win_probability = (0.6, 0.4)
        mock_analysis.principal_variation = [test_move]
        mock_analysis.top_moves = [(test_move, 0.8)]
        mock_analysis.search_depth = 5
        mock_analysis.nodes_searched = 1000
        mock_analysis.time_used = 2.0
        mock_analysis.metadata = {}
        
        with patch.object(self.api_server.game_interface, 'analyze_position') as mock_analyze:
            mock_analyze.return_value = mock_analysis
            
            request_data = {"depth": 5}
            response = self.client.post(f"/sessions/{session_id}/analyze", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "best_move" in data["data"]
            assert "evaluation" in data["data"]
    
    def test_get_move_history(self):
        """测试获取走法历史"""
        # 先创建并开始游戏
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        self.client.post(f"/sessions/{session_id}/start")
        
        # 模拟走法历史
        mock_history = [
            {
                "move_number": 1,
                "move": "a0b1",
                "player": 1,
                "time_used": 2.0
            }
        ]
        
        with patch.object(self.api_server.game_interface, 'get_move_history') as mock_get_history:
            mock_get_history.return_value = mock_history
            
            response = self.client.get(f"/sessions/{session_id}/history")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "moves" in data["data"]
            assert "count" in data["data"]
    
    def test_get_global_statistics(self):
        """测试获取全局统计信息"""
        mock_stats = {
            "total_sessions": 5,
            "finished_sessions": 3,
            "active_sessions": 2
        }
        
        with patch.object(self.api_server.game_interface, 'get_statistics') as mock_get_stats:
            mock_get_stats.return_value = mock_stats
            
            response = self.client.get("/statistics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["data"] == mock_stats
    
    def test_get_session_statistics(self):
        """测试获取会话统计信息"""
        # 先创建一个会话
        create_response = self.client.post("/sessions", json={})
        session_id = create_response.json()["data"]["session_id"]
        
        mock_stats = {
            "session_id": session_id,
            "total_moves": 10,
            "game_duration": 300.0
        }
        
        with patch.object(self.api_server.game_interface, 'get_statistics') as mock_get_stats:
            mock_get_stats.return_value = mock_stats
            
            response = self.client.get(f"/sessions/{session_id}/statistics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["data"] == mock_stats


class TestAPIServerWithAuth:
    """测试带认证的API服务器"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建带认证的API服务器
        self.api_keys = ["test-key-123", "another-key-456"]
        self.api_server = create_api_server(
            ai_model_path="test_model.pth",
            save_directory=self.temp_dir,
            api_keys=self.api_keys
        )
        
        # 创建测试客户端
        self.client = TestClient(self.api_server.app)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_health_check_no_auth_required(self):
        """测试健康检查不需要认证"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_create_session_without_auth(self):
        """测试没有认证时创建会话"""
        request_data = {
            "red_player_type": "human",
            "black_player_type": "ai"
        }
        
        response = self.client.post("/sessions", json=request_data)
        
        assert response.status_code == 401
    
    def test_create_session_with_invalid_auth(self):
        """测试无效认证时创建会话"""
        request_data = {
            "red_player_type": "human",
            "black_player_type": "ai"
        }
        
        headers = {"Authorization": "Bearer invalid-key"}
        response = self.client.post("/sessions", json=request_data, headers=headers)
        
        assert response.status_code == 401
    
    def test_create_session_with_valid_auth(self):
        """测试有效认证时创建会话"""
        request_data = {
            "red_player_type": "human",
            "black_player_type": "ai"
        }
        
        headers = {"Authorization": f"Bearer {self.api_keys[0]}"}
        response = self.client.post("/sessions", json=request_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "session_id" in data["data"]


class TestRateLimiter:
    """测试限流器"""
    
    def test_rate_limiter_allows_requests_within_limit(self):
        """测试限流器允许限制内的请求"""
        from chess_ai_project.src.chinese_chess_ai_engine.inference_interface.api_server import RateLimiter
        
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        client_id = "test_client"
        
        # 前5个请求应该被允许
        for i in range(5):
            assert limiter.is_allowed(client_id) == True
        
        # 第6个请求应该被拒绝
        assert limiter.is_allowed(client_id) == False
    
    def test_rate_limiter_resets_after_window(self):
        """测试限流器在时间窗口后重置"""
        from chess_ai_project.src.chinese_chess_ai_engine.inference_interface.api_server import RateLimiter
        
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        client_id = "test_client"
        
        # 用完限额
        assert limiter.is_allowed(client_id) == True
        assert limiter.is_allowed(client_id) == True
        assert limiter.is_allowed(client_id) == False
        
        # 等待时间窗口过期
        import time
        time.sleep(1.1)
        
        # 现在应该可以再次请求
        assert limiter.is_allowed(client_id) == True


class TestAPIServerIntegration:
    """API服务器集成测试"""
    
    def setup_method(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建API服务器
        self.api_server = create_api_server(
            ai_model_path="test_model.pth",
            save_directory=self.temp_dir
        )
        
        # 创建测试客户端
        self.client = TestClient(self.api_server.app)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('chess_ai_project.src.chinese_chess_ai_engine.inference_interface.game_interface.ChessAI')
    def test_complete_game_flow(self, mock_chess_ai_class):
        """测试完整的游戏流程"""
        # 模拟ChessAI
        mock_ai = Mock()
        mock_chess_ai_class.return_value = mock_ai
        
        # 1. 创建会话
        create_response = self.client.post("/sessions", json={
            "red_player_type": "human",
            "black_player_type": "ai",
            "ai_difficulty": 5
        })
        assert create_response.status_code == 200
        session_id = create_response.json()["data"]["session_id"]
        
        # 2. 开始游戏
        start_response = self.client.post(f"/sessions/{session_id}/start")
        assert start_response.status_code == 200
        
        # 3. 获取游戏状态
        status_response = self.client.get(f"/sessions/{session_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()["data"]
        assert status_data["state"] == "playing"
        
        # 4. 模拟执行走法
        with patch.object(self.api_server.game_interface, 'make_move') as mock_make_move:
            mock_make_move.return_value = (True, "")
            
            with patch.object(self.api_server.game_interface, 'get_game_status') as mock_get_status:
                mock_get_status.return_value = {
                    "session_id": session_id,
                    "state": "playing",
                    "current_player": -1,
                    "total_moves": 1
                }
                
                move_response = self.client.post(f"/sessions/{session_id}/moves", json={
                    "move": "a0b1"
                })
                assert move_response.status_code == 200
        
        # 5. 获取走法历史
        with patch.object(self.api_server.game_interface, 'get_move_history') as mock_get_history:
            mock_get_history.return_value = [
                {"move_number": 1, "move": "a0b1", "player": 1, "time_used": 1.0}
            ]
            
            history_response = self.client.get(f"/sessions/{session_id}/history")
            assert history_response.status_code == 200
            history_data = history_response.json()["data"]
            assert history_data["count"] == 1
        
        # 6. 删除会话
        delete_response = self.client.delete(f"/sessions/{session_id}")
        assert delete_response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__])