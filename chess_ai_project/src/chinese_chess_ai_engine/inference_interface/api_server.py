"""
象棋AI API服务器

提供RESTful API接口，支持异步请求处理、身份验证、限流等功能。
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .game_interface import (
    GameInterface, GameConfig, GameState, GameResult, PlayerType,
    GameInterfaceError, SessionNotFoundError, InvalidMoveError, GameStateError
)
from .chess_ai import AnalysisResult
from ..rules_engine import Move


# ==================== 数据模型 ====================

class APIResponse(BaseModel):
    """API响应基础模型"""
    success: bool = True
    message: str = ""
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = False
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class CreateSessionRequest(BaseModel):
    """创建会话请求模型"""
    red_player_type: PlayerType = PlayerType.HUMAN
    black_player_type: PlayerType = PlayerType.AI
    ai_difficulty: int = Field(default=5, ge=1, le=10)
    time_limit_per_move: Optional[float] = Field(default=None, gt=0)
    total_time_limit: Optional[float] = Field(default=None, gt=0)
    allow_undo: bool = True
    max_moves: int = Field(default=300, gt=0)
    draw_by_repetition: bool = True
    save_game_record: bool = True
    record_analysis: bool = False


class MakeMoveRequest(BaseModel):
    """执行走法请求模型"""
    move: Union[str, Dict[str, Any]]  # 坐标记法字符串或Move字典
    analysis: bool = False


class AnalysisRequest(BaseModel):
    """位置分析请求模型"""
    depth: Optional[int] = Field(default=None, ge=1, le=20)
    time_limit: Optional[float] = Field(default=None, gt=0)


class SessionResponse(BaseModel):
    """会话响应模型"""
    session_id: str
    state: str
    result: str
    current_player: int
    board_fen: str
    board_matrix: List[List[int]]
    legal_moves: List[str]
    total_moves: int
    red_time_used: float
    black_time_used: float
    can_undo: bool
    last_move: Optional[str]
    repetition_count: int


class MoveHistoryResponse(BaseModel):
    """走法历史响应模型"""
    moves: List[Dict[str, Any]]
    total_count: int


class AnalysisResponse(BaseModel):
    """分析结果响应模型"""
    best_move: Optional[str]
    evaluation: float
    win_probability: tuple[float, float]
    principal_variation: List[str]
    top_moves: List[tuple[str, float]]
    search_depth: int
    nodes_searched: int
    time_used: float
    metadata: Dict[str, Any]


# ==================== 中间件和依赖 ====================

class RateLimiter:
    """简单的内存限流器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        
        # 清理过期记录
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
        else:
            self.requests[client_id] = []
        
        # 检查限制
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # 记录请求
        self.requests[client_id].append(now)
        return True


class APIKeyAuth:
    """API密钥认证"""
    
    def __init__(self, api_keys: Optional[List[str]] = None):
        self.api_keys = set(api_keys or [])
        self.security = HTTPBearer(auto_error=False)
    
    async def __call__(self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
        """验证API密钥"""
        # 如果没有配置API密钥，则跳过验证
        if not self.api_keys:
            return None
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="需要API密钥认证",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if credentials.credentials not in self.api_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的API密钥",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return credentials.credentials


# ==================== API服务器类 ====================

class APIServer:
    """象棋AI API服务器"""
    
    def __init__(self, 
                 ai_model_path: Optional[str] = None,
                 save_directory: Optional[str] = None,
                 api_keys: Optional[List[str]] = None,
                 rate_limit_requests: int = 100,
                 rate_limit_window: int = 60,
                 cors_origins: Optional[List[str]] = None,
                 trusted_hosts: Optional[List[str]] = None):
        """
        初始化API服务器
        
        Args:
            ai_model_path: AI模型路径
            save_directory: 游戏记录保存目录
            api_keys: API密钥列表，如果为空则不启用认证
            rate_limit_requests: 限流请求数
            rate_limit_window: 限流时间窗口（秒）
            cors_origins: CORS允许的源
            trusted_hosts: 信任的主机列表
        """
        self.logger = logging.getLogger(__name__)
        
        # 游戏接口
        self.game_interface = GameInterface(ai_model_path, save_directory)
        
        # 认证和限流
        self.api_auth = APIKeyAuth(api_keys)
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        
        # 配置
        self.cors_origins = cors_origins or ["*"]
        self.trusted_hosts = trusted_hosts or ["*"]
        
        # 创建FastAPI应用
        self.app = self._create_app()
        
        self.logger.info("API服务器初始化完成")
    
    def _create_app(self) -> FastAPI:
        """创建FastAPI应用"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """应用生命周期管理"""
            self.logger.info("API服务器启动")
            yield
            self.logger.info("API服务器关闭")
        
        app = FastAPI(
            title="象棋AI API",
            description="中国象棋AI引擎的RESTful API接口",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            lifespan=lifespan
        )
        
        # 添加中间件
        self._add_middlewares(app)
        
        # 添加路由
        self._add_routes(app)
        
        # 添加异常处理器
        self._add_exception_handlers(app)
        
        return app
    
    def _add_middlewares(self, app: FastAPI):
        """添加中间件"""
        
        # CORS中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 信任主机中间件
        if self.trusted_hosts != ["*"]:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.trusted_hosts
            )
        
        # GZip压缩中间件
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # 请求处理时间中间件
        @app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.perf_counter()
            response = await call_next(request)
            process_time = time.perf_counter() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        
        # 限流中间件
        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            
            if not self.rate_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content=ErrorResponse(
                        error_code="RATE_LIMIT_EXCEEDED",
                        error_message="请求频率过高，请稍后再试"
                    ).dict()
                )
            
            response = await call_next(request)
            return response
    
    def _add_routes(self, app: FastAPI):
        """添加API路由"""
        
        # ==================== 健康检查 ====================
        
        @app.get("/health", response_model=APIResponse, tags=["健康检查"])
        async def health_check():
            """健康检查接口"""
            return APIResponse(
                message="服务正常运行",
                data={
                    "status": "healthy",
                    "timestamp": datetime.now(),
                    "version": "1.0.0"
                }
            )
        
        # ==================== 会话管理 ====================
        
        @app.post("/sessions", response_model=APIResponse, tags=["会话管理"])
        async def create_session(
            request: CreateSessionRequest,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """创建新的游戏会话"""
            try:
                config = GameConfig(
                    red_player_type=request.red_player_type,
                    black_player_type=request.black_player_type,
                    ai_difficulty=request.ai_difficulty,
                    time_limit_per_move=request.time_limit_per_move,
                    total_time_limit=request.total_time_limit,
                    allow_undo=request.allow_undo,
                    max_moves=request.max_moves,
                    draw_by_repetition=request.draw_by_repetition,
                    save_game_record=request.save_game_record,
                    record_analysis=request.record_analysis
                )
                
                session_id = self.game_interface.create_session(config)
                
                return APIResponse(
                    message="会话创建成功",
                    data={"session_id": session_id}
                )
                
            except Exception as e:
                self.logger.error(f"创建会话失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"创建会话失败: {str(e)}"
                )
        
        @app.get("/sessions", response_model=APIResponse, tags=["会话管理"])
        async def list_sessions(api_key: Optional[str] = Depends(self.api_auth)):
            """获取所有会话列表"""
            try:
                sessions = self.game_interface.list_sessions()
                return APIResponse(
                    message="获取会话列表成功",
                    data={"sessions": sessions, "count": len(sessions)}
                )
            except Exception as e:
                self.logger.error(f"获取会话列表失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取会话列表失败: {str(e)}"
                )
        
        @app.get("/sessions/{session_id}", response_model=APIResponse, tags=["会话管理"])
        async def get_session_status(
            session_id: str,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """获取会话状态"""
            try:
                status_info = self.game_interface.get_game_status(session_id)
                return APIResponse(
                    message="获取会话状态成功",
                    data=status_info
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"获取会话状态失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取会话状态失败: {str(e)}"
                )
        
        @app.delete("/sessions/{session_id}", response_model=APIResponse, tags=["会话管理"])
        async def delete_session(
            session_id: str,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """删除会话"""
            try:
                success = self.game_interface.delete_session(session_id)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="会话不存在"
                    )
                
                return APIResponse(message="会话删除成功")
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"删除会话失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"删除会话失败: {str(e)}"
                )
        
        # ==================== 游戏控制 ====================
        
        @app.post("/sessions/{session_id}/start", response_model=APIResponse, tags=["游戏控制"])
        async def start_game(
            session_id: str,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """开始游戏"""
            try:
                success = self.game_interface.start_game(session_id)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="无法开始游戏，请检查游戏状态"
                    )
                
                return APIResponse(message="游戏开始成功")
                
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"开始游戏失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"开始游戏失败: {str(e)}"
                )
        
        @app.post("/sessions/{session_id}/moves", response_model=APIResponse, tags=["游戏控制"])
        async def make_move(
            session_id: str,
            request: MakeMoveRequest,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """执行走法"""
            try:
                success, error = self.game_interface.make_move(
                    request.move, session_id, request.analysis
                )
                
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=error
                    )
                
                # 获取更新后的游戏状态
                status_info = self.game_interface.get_game_status(session_id)
                
                return APIResponse(
                    message="走法执行成功",
                    data=status_info
                )
                
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"执行走法失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"执行走法失败: {str(e)}"
                )
        
        @app.get("/sessions/{session_id}/ai-move", response_model=APIResponse, tags=["游戏控制"])
        async def get_ai_move(
            session_id: str,
            time_limit: Optional[float] = None,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """获取AI走法"""
            try:
                ai_move = self.game_interface.get_ai_move(session_id, time_limit)
                
                if ai_move is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="无法获取AI走法，请检查游戏状态和玩家类型"
                    )
                
                return APIResponse(
                    message="获取AI走法成功",
                    data={"move": ai_move.to_coordinate_notation()}
                )
                
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"获取AI走法失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取AI走法失败: {str(e)}"
                )
        
        @app.post("/sessions/{session_id}/undo", response_model=APIResponse, tags=["游戏控制"])
        async def undo_move(
            session_id: str,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """撤销走法"""
            try:
                success = self.game_interface.undo_move(session_id)
                
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="无法撤销走法，请检查游戏状态和配置"
                    )
                
                # 获取更新后的游戏状态
                status_info = self.game_interface.get_game_status(session_id)
                
                return APIResponse(
                    message="撤销走法成功",
                    data=status_info
                )
                
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"撤销走法失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"撤销走法失败: {str(e)}"
                )
        
        @app.post("/sessions/{session_id}/pause", response_model=APIResponse, tags=["游戏控制"])
        async def pause_game(
            session_id: str,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """暂停游戏"""
            try:
                success = self.game_interface.pause_game(session_id)
                
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="无法暂停游戏，请检查游戏状态"
                    )
                
                return APIResponse(message="游戏暂停成功")
                
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"暂停游戏失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"暂停游戏失败: {str(e)}"
                )
        
        @app.post("/sessions/{session_id}/resume", response_model=APIResponse, tags=["游戏控制"])
        async def resume_game(
            session_id: str,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """恢复游戏"""
            try:
                success = self.game_interface.resume_game(session_id)
                
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="无法恢复游戏，请检查游戏状态"
                    )
                
                return APIResponse(message="游戏恢复成功")
                
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"恢复游戏失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"恢复游戏失败: {str(e)}"
                )
        
        @app.post("/sessions/{session_id}/resign", response_model=APIResponse, tags=["游戏控制"])
        async def resign_game(
            session_id: str,
            player: int,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """认输"""
            try:
                success = self.game_interface.resign_game(player, session_id)
                
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="无法认输，请检查游戏状态"
                    )
                
                return APIResponse(message="认输成功")
                
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"认输失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"认输失败: {str(e)}"
                )
        
        # ==================== 分析功能 ====================
        
        @app.post("/sessions/{session_id}/analyze", response_model=APIResponse, tags=["分析功能"])
        async def analyze_position(
            session_id: str,
            request: AnalysisRequest,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """分析当前位置"""
            try:
                analysis = self.game_interface.analyze_position(
                    session_id, request.depth
                )
                
                if analysis is None:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="位置分析失败"
                    )
                
                return APIResponse(
                    message="位置分析成功",
                    data={
                        "best_move": analysis.best_move.to_coordinate_notation() if analysis.best_move else None,
                        "evaluation": analysis.evaluation,
                        "win_probability": analysis.win_probability,
                        "principal_variation": [move.to_coordinate_notation() for move in analysis.principal_variation],
                        "top_moves": [(move.to_coordinate_notation(), score) for move, score in analysis.top_moves],
                        "search_depth": analysis.search_depth,
                        "nodes_searched": analysis.nodes_searched,
                        "time_used": analysis.time_used,
                        "metadata": analysis.metadata
                    }
                )
                
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"位置分析失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"位置分析失败: {str(e)}"
                )
        
        # ==================== 历史记录 ====================
        
        @app.get("/sessions/{session_id}/history", response_model=APIResponse, tags=["历史记录"])
        async def get_move_history(
            session_id: str,
            include_analysis: bool = False,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """获取走法历史"""
            try:
                history = self.game_interface.get_move_history(
                    session_id, include_analysis
                )
                
                return APIResponse(
                    message="获取走法历史成功",
                    data={"moves": history, "count": len(history)}
                )
                
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"获取走法历史失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取走法历史失败: {str(e)}"
                )
        
        # ==================== 统计信息 ====================
        
        @app.get("/statistics", response_model=APIResponse, tags=["统计信息"])
        async def get_global_statistics(api_key: Optional[str] = Depends(self.api_auth)):
            """获取全局统计信息"""
            try:
                stats = self.game_interface.get_statistics()
                return APIResponse(
                    message="获取统计信息成功",
                    data=stats
                )
            except Exception as e:
                self.logger.error(f"获取统计信息失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取统计信息失败: {str(e)}"
                )
        
        @app.get("/sessions/{session_id}/statistics", response_model=APIResponse, tags=["统计信息"])
        async def get_session_statistics(
            session_id: str,
            api_key: Optional[str] = Depends(self.api_auth)
        ):
            """获取会话统计信息"""
            try:
                stats = self.game_interface.get_statistics(session_id)
                return APIResponse(
                    message="获取会话统计信息成功",
                    data=stats
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                self.logger.error(f"获取会话统计信息失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"获取会话统计信息失败: {str(e)}"
                )
    
    def _add_exception_handlers(self, app: FastAPI):
        """添加异常处理器"""
        
        @app.exception_handler(GameInterfaceError)
        async def game_interface_error_handler(request: Request, exc: GameInterfaceError):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error_code="GAME_INTERFACE_ERROR",
                    error_message=str(exc)
                ).dict()
            )
        
        @app.exception_handler(SessionNotFoundError)
        async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ErrorResponse(
                    error_code="SESSION_NOT_FOUND",
                    error_message=str(exc)
                ).dict()
            )
        
        @app.exception_handler(InvalidMoveError)
        async def invalid_move_handler(request: Request, exc: InvalidMoveError):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error_code="INVALID_MOVE",
                    error_message=str(exc)
                ).dict()
            )
        
        @app.exception_handler(GameStateError)
        async def game_state_error_handler(request: Request, exc: GameStateError):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    error_code="GAME_STATE_ERROR",
                    error_message=str(exc)
                ).dict()
            )
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """运行API服务器"""
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# ==================== 工厂函数 ====================

def create_api_server(
    ai_model_path: Optional[str] = None,
    save_directory: Optional[str] = None,
    api_keys: Optional[List[str]] = None,
    **kwargs
) -> APIServer:
    """创建API服务器实例"""
    return APIServer(
        ai_model_path=ai_model_path,
        save_directory=save_directory,
        api_keys=api_keys,
        **kwargs
    )


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="象棋AI API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--model-path", help="AI模型路径")
    parser.add_argument("--save-dir", help="游戏记录保存目录")
    parser.add_argument("--api-keys", nargs="*", help="API密钥列表")
    parser.add_argument("--reload", action="store_true", help="启用自动重载")
    parser.add_argument("--log-level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建服务器
    server = create_api_server(
        ai_model_path=args.model_path,
        save_directory=args.save_dir,
        api_keys=args.api_keys
    )
    
    # 运行服务器
    server.run(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )