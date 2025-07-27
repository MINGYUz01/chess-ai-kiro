# 象棋AI API服务器使用指南

## 概述

象棋AI API服务器提供了完整的RESTful API接口，支持游戏会话管理、走法执行、AI分析等功能。服务器基于FastAPI构建，支持异步处理、身份验证、限流等企业级特性。

## 快速开始

### 启动服务器

```bash
# 基本启动
python -m chess_ai_project.src.chinese_chess_ai_engine.inference_interface.api_server

# 指定参数启动
python -m chess_ai_project.src.chinese_chess_ai_engine.inference_interface.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model-path models/chess_model.pth \
    --save-dir data/sessions \
    --api-keys key1 key2 \
    --reload
```

### 使用代码启动

```python
from chess_ai_project.src.chinese_chess_ai_engine.inference_interface import create_api_server

# 创建服务器
server = create_api_server(
    ai_model_path="models/chess_model.pth",
    save_directory="data/sessions",
    api_keys=["your-api-key-here"],  # 可选，不设置则不启用认证
    rate_limit_requests=100,         # 每分钟最大请求数
    rate_limit_window=60,           # 限流时间窗口（秒）
    cors_origins=["*"],             # CORS允许的源
    trusted_hosts=["*"]             # 信任的主机
)

# 运行服务器
server.run(host="0.0.0.0", port=8000, reload=True)
```

## API接口文档

服务器启动后，可以通过以下URL访问API文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 主要功能

### 1. 健康检查

```http
GET /health
```

检查服务器运行状态。

**响应示例**:
```json
{
  "success": true,
  "message": "服务正常运行",
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00",
    "version": "1.0.0"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### 2. 会话管理

#### 创建会话

```http
POST /sessions
Content-Type: application/json
Authorization: Bearer your-api-key  # 如果启用了认证

{
  "red_player_type": "human",
  "black_player_type": "ai",
  "ai_difficulty": 5,
  "time_limit_per_move": 30.0,
  "total_time_limit": 1800.0,
  "allow_undo": true,
  "max_moves": 300,
  "draw_by_repetition": true,
  "save_game_record": true,
  "record_analysis": false
}
```

#### 获取会话列表

```http
GET /sessions
Authorization: Bearer your-api-key
```

#### 获取会话状态

```http
GET /sessions/{session_id}
Authorization: Bearer your-api-key
```

#### 删除会话

```http
DELETE /sessions/{session_id}
Authorization: Bearer your-api-key
```

### 3. 游戏控制

#### 开始游戏

```http
POST /sessions/{session_id}/start
Authorization: Bearer your-api-key
```

#### 执行走法

```http
POST /sessions/{session_id}/moves
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "move": "a0b1",  # 坐标记法
  "analysis": false
}
```

#### 获取AI走法

```http
GET /sessions/{session_id}/ai-move?time_limit=5.0
Authorization: Bearer your-api-key
```

#### 撤销走法

```http
POST /sessions/{session_id}/undo
Authorization: Bearer your-api-key
```

#### 暂停/恢复游戏

```http
POST /sessions/{session_id}/pause
POST /sessions/{session_id}/resume
Authorization: Bearer your-api-key
```

#### 认输

```http
POST /sessions/{session_id}/resign?player=1
Authorization: Bearer your-api-key
```

### 4. 分析功能

#### 位置分析

```http
POST /sessions/{session_id}/analyze
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "depth": 10,
  "time_limit": 5.0
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "位置分析成功",
  "data": {
    "best_move": "a0b1",
    "evaluation": 0.25,
    "win_probability": [0.6, 0.4],
    "principal_variation": ["a0b1", "b9c7", "c0d2"],
    "top_moves": [["a0b1", 0.8], ["c0d2", 0.6]],
    "search_depth": 10,
    "nodes_searched": 15000,
    "time_used": 3.2,
    "metadata": {
      "difficulty_level": 5,
      "temperature": 0.1
    }
  }
}
```

### 5. 历史记录

#### 获取走法历史

```http
GET /sessions/{session_id}/history?include_analysis=false
Authorization: Bearer your-api-key
```

### 6. 统计信息

#### 全局统计

```http
GET /statistics
Authorization: Bearer your-api-key
```

#### 会话统计

```http
GET /sessions/{session_id}/statistics
Authorization: Bearer your-api-key
```

## 客户端示例

### Python客户端

```python
import requests
import json

class ChessAIClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def create_session(self, config=None):
        """创建游戏会话"""
        config = config or {
            "red_player_type": "human",
            "black_player_type": "ai",
            "ai_difficulty": 5
        }
        
        response = requests.post(
            f"{self.base_url}/sessions",
            json=config,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["data"]["session_id"]
    
    def start_game(self, session_id):
        """开始游戏"""
        response = requests.post(
            f"{self.base_url}/sessions/{session_id}/start",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def make_move(self, session_id, move):
        """执行走法"""
        response = requests.post(
            f"{self.base_url}/sessions/{session_id}/moves",
            json={"move": move},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_ai_move(self, session_id):
        """获取AI走法"""
        response = requests.get(
            f"{self.base_url}/sessions/{session_id}/ai-move",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["data"]["move"]
    
    def analyze_position(self, session_id, depth=None):
        """分析位置"""
        data = {}
        if depth:
            data["depth"] = depth
        
        response = requests.post(
            f"{self.base_url}/sessions/{session_id}/analyze",
            json=data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["data"]
    
    def get_game_status(self, session_id):
        """获取游戏状态"""
        response = requests.get(
            f"{self.base_url}/sessions/{session_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["data"]

# 使用示例
client = ChessAIClient(api_key="your-api-key")

# 创建会话并开始游戏
session_id = client.create_session()
client.start_game(session_id)

# 执行走法
client.make_move(session_id, "a0b1")

# 获取AI走法
ai_move = client.get_ai_move(session_id)
print(f"AI走法: {ai_move}")

# 分析位置
analysis = client.analyze_position(session_id, depth=8)
print(f"最佳走法: {analysis['best_move']}")
print(f"评估值: {analysis['evaluation']}")
```

### JavaScript客户端

```javascript
class ChessAIClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json'
        };
        if (apiKey) {
            this.headers['Authorization'] = `Bearer ${apiKey}`;
        }
    }
    
    async createSession(config = {}) {
        const defaultConfig = {
            red_player_type: 'human',
            black_player_type: 'ai',
            ai_difficulty: 5
        };
        
        const response = await fetch(`${this.baseUrl}/sessions`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({...defaultConfig, ...config})
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error_message || 'Request failed');
        return data.data.session_id;
    }
    
    async startGame(sessionId) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/start`, {
            method: 'POST',
            headers: this.headers
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error_message || 'Request failed');
        return data;
    }
    
    async makeMove(sessionId, move) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/moves`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({move})
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error_message || 'Request failed');
        return data;
    }
    
    async getAIMove(sessionId) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/ai-move`, {
            headers: this.headers
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error_message || 'Request failed');
        return data.data.move;
    }
    
    async analyzePosition(sessionId, depth = null) {
        const body = {};
        if (depth) body.depth = depth;
        
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/analyze`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(body)
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error_message || 'Request failed');
        return data.data;
    }
    
    async getGameStatus(sessionId) {
        const response = await fetch(`${this.baseUrl}/sessions/${sessionId}`, {
            headers: this.headers
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error_message || 'Request failed');
        return data.data;
    }
}

// 使用示例
const client = new ChessAIClient('http://localhost:8000', 'your-api-key');

async function playGame() {
    try {
        // 创建会话并开始游戏
        const sessionId = await client.createSession();
        await client.startGame(sessionId);
        
        // 执行走法
        await client.makeMove(sessionId, 'a0b1');
        
        // 获取AI走法
        const aiMove = await client.getAIMove(sessionId);
        console.log(`AI走法: ${aiMove}`);
        
        // 分析位置
        const analysis = await client.analyzePosition(sessionId, 8);
        console.log(`最佳走法: ${analysis.best_move}`);
        console.log(`评估值: ${analysis.evaluation}`);
        
    } catch (error) {
        console.error('游戏出错:', error.message);
    }
}

playGame();
```

## 错误处理

API使用标准的HTTP状态码和统一的错误响应格式：

```json
{
  "success": false,
  "error_code": "INVALID_MOVE",
  "error_message": "非法走法",
  "details": {
    "move": "invalid_move",
    "legal_moves": ["a0b1", "c0d2"]
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### 常见错误码

- `RATE_LIMIT_EXCEEDED`: 请求频率过高
- `SESSION_NOT_FOUND`: 会话不存在
- `INVALID_MOVE`: 非法走法
- `GAME_STATE_ERROR`: 游戏状态错误
- `GAME_INTERFACE_ERROR`: 游戏接口错误

## 安全特性

### 1. API密钥认证

```python
# 启用API密钥认证
server = create_api_server(
    api_keys=["key1", "key2", "key3"]
)
```

### 2. 限流保护

```python
# 配置限流参数
server = create_api_server(
    rate_limit_requests=100,  # 每分钟最大请求数
    rate_limit_window=60      # 时间窗口（秒）
)
```

### 3. CORS配置

```python
# 配置CORS
server = create_api_server(
    cors_origins=["https://yourdomain.com", "https://app.yourdomain.com"]
)
```

### 4. 信任主机

```python
# 配置信任主机
server = create_api_server(
    trusted_hosts=["yourdomain.com", "*.yourdomain.com"]
)
```

## 性能优化

### 1. 异步处理

API服务器使用FastAPI的异步特性，支持高并发请求处理。

### 2. 响应压缩

自动启用GZip压缩，减少网络传输量。

### 3. 请求时间监控

每个响应都包含`X-Process-Time`头，显示处理时间。

### 4. 连接池

建议客户端使用连接池来提高性能：

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

## 部署建议

### 1. 生产环境配置

```bash
# 使用Gunicorn部署
pip install gunicorn
gunicorn chess_ai_project.src.chinese_chess_ai_engine.inference_interface.api_server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile -
```

### 2. Docker部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "chess_ai_project.src.chinese_chess_ai_engine.inference_interface.api_server:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### 3. 反向代理配置

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 监控和日志

### 1. 日志配置

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
```

### 2. 健康检查

定期调用`/health`端点检查服务器状态。

### 3. 性能监控

监控`X-Process-Time`头来跟踪API响应时间。

## 故障排除

### 1. 常见问题

**问题**: 服务器启动失败
**解决**: 检查端口是否被占用，模型文件是否存在

**问题**: API请求超时
**解决**: 增加客户端超时时间，检查服务器负载

**问题**: 认证失败
**解决**: 检查API密钥是否正确，确认请求头格式

### 2. 调试模式

```python
# 启用调试模式
server.run(debug=True, reload=True)
```

### 3. 日志级别

```bash
# 设置详细日志
python api_server.py --log-level debug
```