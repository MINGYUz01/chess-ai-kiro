"""
异常定义

定义象棋AI引擎的各种异常类型。
"""


class ChessAIError(Exception):
    """
    象棋AI引擎基础异常
    
    所有象棋AI相关异常的基类。
    """
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class InvalidMoveError(ChessAIError):
    """
    非法走法异常
    
    当尝试执行非法走法时抛出。
    """
    
    def __init__(self, move_str: str, reason: str = ""):
        message = f"非法走法: {move_str}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "INVALID_MOVE")
        self.move_str = move_str
        self.reason = reason


class ModelLoadError(ChessAIError):
    """
    模型加载异常
    
    当模型加载失败时抛出。
    """
    
    def __init__(self, model_path: str, reason: str = ""):
        message = f"模型加载失败: {model_path}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "MODEL_LOAD_ERROR")
        self.model_path = model_path
        self.reason = reason


class SearchTimeoutError(ChessAIError):
    """
    搜索超时异常
    
    当搜索超过时间限制时抛出。
    """
    
    def __init__(self, time_limit: float, actual_time: float = None):
        message = f"搜索超时: 时间限制 {time_limit:.2f}秒"
        if actual_time:
            message += f", 实际用时 {actual_time:.2f}秒"
        super().__init__(message, "SEARCH_TIMEOUT")
        self.time_limit = time_limit
        self.actual_time = actual_time


class TrainingError(ChessAIError):
    """
    训练过程异常
    
    当训练过程中出现错误时抛出。
    """
    
    def __init__(self, stage: str, reason: str = ""):
        message = f"训练错误 - {stage}"
        if reason:
            message += f": {reason}"
        super().__init__(message, "TRAINING_ERROR")
        self.stage = stage
        self.reason = reason


class ConfigurationError(ChessAIError):
    """
    配置错误异常
    
    当配置参数无效时抛出。
    """
    
    def __init__(self, config_name: str, reason: str = ""):
        message = f"配置错误 - {config_name}"
        if reason:
            message += f": {reason}"
        super().__init__(message, "CONFIG_ERROR")
        self.config_name = config_name
        self.reason = reason


class GameStateError(ChessAIError):
    """
    游戏状态异常
    
    当游戏状态无效或不一致时抛出。
    """
    
    def __init__(self, state_description: str, reason: str = ""):
        message = f"游戏状态错误: {state_description}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "GAME_STATE_ERROR")
        self.state_description = state_description
        self.reason = reason


class NetworkError(ChessAIError):
    """
    网络相关异常
    
    当神经网络操作失败时抛出。
    """
    
    def __init__(self, operation: str, reason: str = ""):
        message = f"网络错误 - {operation}"
        if reason:
            message += f": {reason}"
        super().__init__(message, "NETWORK_ERROR")
        self.operation = operation
        self.reason = reason


class DataError(ChessAIError):
    """
    数据相关异常
    
    当数据处理出现错误时抛出。
    """
    
    def __init__(self, data_type: str, reason: str = ""):
        message = f"数据错误 - {data_type}"
        if reason:
            message += f": {reason}"
        super().__init__(message, "DATA_ERROR")
        self.data_type = data_type
        self.reason = reason


class ResourceError(ChessAIError):
    """
    资源相关异常
    
    当系统资源不足或访问失败时抛出。
    """
    
    def __init__(self, resource_type: str, reason: str = ""):
        message = f"资源错误 - {resource_type}"
        if reason:
            message += f": {reason}"
        super().__init__(message, "RESOURCE_ERROR")
        self.resource_type = resource_type
        self.reason = reason


class APIError(ChessAIError):
    """
    API相关异常
    
    当API调用失败时抛出。
    """
    
    def __init__(self, endpoint: str, status_code: int = None, reason: str = ""):
        message = f"API错误 - {endpoint}"
        if status_code:
            message += f" (状态码: {status_code})"
        if reason:
            message += f": {reason}"
        super().__init__(message, "API_ERROR")
        self.endpoint = endpoint
        self.status_code = status_code
        self.reason = reason