"""
中国象棋神经网络模型

实现基于ResNet的双头网络架构，包含价值网络和策略网络。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class ResidualBlock(nn.Module):
    """
    残差块
    
    实现ResNet的基本残差连接结构。
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        """
        初始化残差块
        
        Args:
            channels: 通道数
            kernel_size: 卷积核大小
            dropout: Dropout概率
        """
        super(ResidualBlock, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 输出张量
        """
        residual = x
        
        # 第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # 第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        out += residual
        out = F.relu(out)
        
        return out


class AttentionModule(nn.Module):
    """
    注意力机制模块
    
    实现自注意力机制来增强位置理解能力。
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        """
        初始化注意力模块
        
        Args:
            channels: 输入通道数
            num_heads: 注意力头数
        """
        super(AttentionModule, self).__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels必须能被num_heads整除"
        
        # 查询、键、值的线性变换
        self.query_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.key_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.value_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
        # 输出投影
        self.output_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.LayerNorm(channels)
        
        # 位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, channels, 10, 9))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 输出张量
        """
        batch_size, channels, height, width = x.shape
        residual = x
        
        # 添加位置编码
        x = x + self.position_encoding
        
        # 生成查询、键、值
        query = self.query_conv(x)  # [B, C, H, W]
        key = self.key_conv(x)      # [B, C, H, W]
        value = self.value_conv(x)  # [B, C, H, W]
        
        # 重塑为多头注意力格式
        query = query.view(batch_size, self.num_heads, self.head_dim, height * width)
        key = key.view(batch_size, self.num_heads, self.head_dim, height * width)
        value = value.view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query.transpose(-2, -1), key)  # [B, H, HW, HW]
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重
        attended_values = torch.matmul(value, attention_weights.transpose(-2, -1))  # [B, H, D, HW]
        
        # 重塑回原始格式
        attended_values = attended_values.view(batch_size, channels, height, width)
        
        # 输出投影
        output = self.output_conv(attended_values)
        
        # 残差连接和层归一化
        output = output + residual
        output = output.permute(0, 2, 3, 1)  # [B, H, W, C]
        output = self.norm(output)
        output = output.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return output


class ChessNet(nn.Module):
    """
    中国象棋神经网络
    
    基于ResNet架构的双头网络，包含价值网络和策略网络。
    """
    
    def __init__(
        self,
        input_channels: int = 20,
        num_blocks: int = 20,
        channels: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化网络
        
        Args:
            input_channels: 输入通道数（棋盘特征维度，默认20通道）
            num_blocks: 残差块数量
            channels: 网络通道数
            num_attention_heads: 注意力头数
            dropout: Dropout概率
        """
        super(ChessNet, self).__init__()
        
        self.input_channels = input_channels
        self.num_blocks = num_blocks
        self.channels = channels
        
        # 输入卷积层
        self.input_conv = nn.Conv2d(input_channels, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels, dropout=dropout) for _ in range(num_blocks)
        ])
        
        # 注意力模块（在中间层添加）
        self.attention_modules = nn.ModuleList([
            AttentionModule(channels, num_attention_heads) 
            for _ in range(num_blocks // 4)  # 每4个残差块添加一个注意力模块
        ])
        
        # 价值网络头
        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(dropout)
        
        # 策略网络头
        self.policy_conv = nn.Conv2d(channels, 8, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_fc = nn.Linear(8 * 10 * 9, 10 * 9 * 90)  # 90个可能的移动方向
        self.policy_dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_channels, 10, 9] (默认20通道)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (价值输出, 策略输出)
                - 价值输出: [batch_size, 1] 范围[-1, 1]
                - 策略输出: [batch_size, 8100] 所有可能走法的概率分布
        """
        # 输入层
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        # 残差块和注意力模块
        attention_idx = 0
        for i, block in enumerate(self.residual_blocks):
            x = block(x)
            
            # 每4个残差块后添加注意力
            if (i + 1) % 4 == 0 and attention_idx < len(self.attention_modules):
                x = self.attention_modules[attention_idx](x)
                attention_idx += 1
        
        # 价值网络头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.reshape(value.size(0), -1)  # 展平
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_dropout(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)  # 输出范围[-1, 1]
        
        # 策略网络头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.reshape(policy.size(0), -1)  # 展平
        policy = self.policy_dropout(policy)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)  # 对数概率分布
        
        return value, policy
    
    def predict_value(self, board_tensor: torch.Tensor) -> float:
        """
        预测棋局价值
        
        Args:
            board_tensor: 棋盘张量 [input_channels, 10, 9] (默认20通道)
            
        Returns:
            float: 价值评估 [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)  # 添加batch维度
            
            value, _ = self.forward(board_tensor)
            return value.item()
    
    def predict_policy(self, board_tensor: torch.Tensor) -> np.ndarray:
        """
        预测策略分布
        
        Args:
            board_tensor: 棋盘张量 [input_channels, 10, 9] (默认20通道)
            
        Returns:
            np.ndarray: 策略概率分布 [8100]
        """
        self.eval()
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)  # 添加batch维度
            
            _, policy = self.forward(board_tensor)
            policy = torch.exp(policy)  # 转换为概率
            return policy.cpu().numpy().flatten()
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            metadata: 元数据信息
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_channels': self.input_channels,
                'num_blocks': self.num_blocks,
                'channels': self.channels,
            },
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        加载模型检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            Dict[str, Any]: 元数据信息
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('metadata', {})
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_channels': self.input_channels,
            'num_blocks': self.num_blocks,
            'channels': self.channels,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
        }