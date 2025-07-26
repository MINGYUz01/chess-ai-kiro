"""
神经网络训练器

实现AlphaZero风格的神经网络训练功能。
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# 可选的TensorBoard支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from .training_config import TrainingConfig
from .training_example import TrainingExample, TrainingDataset
from ..neural_network import ChessNet


class ChessTrainingDataset(Dataset):
    """
    象棋训练数据集
    
    将TrainingDataset包装为PyTorch Dataset。
    """
    
    def __init__(self, training_dataset: TrainingDataset):
        """
        初始化数据集
        
        Args:
            training_dataset: 训练数据集
        """
        self.training_dataset = training_dataset
        self.examples = training_dataset.examples
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (棋盘特征, 策略目标, 价值目标)
        """
        example = self.examples[idx]
        
        # 确保张量在正确的设备上
        board_tensor = example.board_tensor.clone().detach()
        policy_target = torch.from_numpy(example.policy_target).clone().detach()
        value_target = torch.tensor(example.value_target, dtype=torch.float32)
        
        return board_tensor, policy_target, value_target


class EarlyStopping:
    """
    早停机制
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        """
        初始化早停机制
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            mode: 监控模式 ('min' 或 'max')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前分数
            
        Returns:
            bool: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """检查当前分数是否更好"""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class ModelEMA:
    """
    模型指数移动平均
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        初始化EMA
        
        Args:
            model: 原始模型
            decay: 衰减率
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化shadow参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """
    神经网络训练器
    
    实现AlphaZero风格的神经网络训练。
    """
    
    def __init__(
        self,
        model: ChessNet,
        config: TrainingConfig,
        checkpoint_dir: Optional[str] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            config: 训练配置
            checkpoint_dir: 检查点保存目录
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('./checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 设置设备
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.scaler = torch.amp.GradScaler(enabled=config.mixed_precision)
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        ) if config.early_stopping else None
        
        # 模型EMA
        self.model_ema = ModelEMA(model, config.model_ema_decay) if config.model_ema else None
        
        # 编译模型
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # TensorBoard
        self.writer = None
        if config.tensorboard_log and TENSORBOARD_AVAILABLE:
            log_dir = self.checkpoint_dir / 'tensorboard'
            self.writer = SummaryWriter(log_dir=str(log_dir))
        elif config.tensorboard_log and not TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard不可用，跳过日志记录")
        
        # 保存配置
        config_path = self.checkpoint_dir / 'config.json'
        config.save_to_file(str(config_path))
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        params = self.config.get_optimizer_params()
        
        if self.config.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), **params)
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(self.model.parameters(), **params)
        elif self.config.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), **params)
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        if self.config.lr_scheduler == 'constant':
            return None
        
        params = self.config.get_scheduler_params()
        
        if self.config.lr_scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **params)
        elif self.config.lr_scheduler == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, **params)
        elif self.config.lr_scheduler == 'exponential':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, **params)
        elif self.config.lr_scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **params)
        elif self.config.lr_scheduler == 'linear':
            return optim.lr_scheduler.LinearLR(self.optimizer, **params)
        else:
            raise ValueError(f"不支持的学习率调度器: {self.config.lr_scheduler}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            Dict[str, float]: 训练指标
        """
        self.model.train()
        
        # 统计指标
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_samples = 0
        
        # 进度统计
        batch_count = len(train_loader)
        start_time = time.time()
        
        for batch_idx, (board_tensors, policy_targets, value_targets) in enumerate(train_loader):
            # 移动到设备
            board_tensors = board_tensors.to(self.device, non_blocking=True)
            policy_targets = policy_targets.to(self.device, non_blocking=True)
            value_targets = value_targets.to(self.device, non_blocking=True)
            
            # 前向传播
            with torch.autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
                value_pred, policy_pred = self.model(board_tensors)
                
                # 计算损失
                value_loss = F.mse_loss(value_pred.squeeze(), value_targets)
                policy_loss = F.cross_entropy(policy_pred, policy_targets)
                
                # 总损失
                loss = (self.config.value_loss_weight * value_loss + 
                       self.config.policy_loss_weight * policy_loss)
                
                # 梯度累积
                loss = loss / self.config.accumulate_grad_batches
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度累积和更新
            if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
                # 梯度裁剪
                if self.config.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                # 优化器步骤
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # 更新EMA
                if self.model_ema:
                    self.model_ema.update()
                
                self.global_step += 1
            
            # 统计
            batch_size = board_tensors.size(0)
            total_loss += loss.item() * batch_size * self.config.accumulate_grad_batches
            total_value_loss += value_loss.item() * batch_size
            total_policy_loss += policy_loss.item() * batch_size
            total_samples += batch_size
            
            # 日志记录
            if (batch_idx + 1) % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                samples_per_sec = total_samples / elapsed_time
                
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{batch_count}] "
                    f"Loss: {loss.item():.6f} "
                    f"Value: {value_loss.item():.6f} "
                    f"Policy: {policy_loss.item():.6f} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f} "
                    f"Speed: {samples_per_sec:.1f} samples/s"
                )
                
                # TensorBoard记录
                if self.writer:
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/ValueLoss', value_loss.item(), self.global_step)
                    self.writer.add_scalar('Train/PolicyLoss', policy_loss.item(), self.global_step)
                    self.writer.add_scalar('Train/LearningRate', 
                                         self.optimizer.param_groups[0]['lr'], self.global_step)
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        avg_value_loss = total_value_loss / total_samples
        avg_policy_loss = total_policy_loss / total_samples
        
        # 学习率调度器步骤
        if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'value_loss': avg_value_loss,
            'policy_loss': avg_policy_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch
            
        Returns:
            Dict[str, float]: 验证指标
        """
        self.model.eval()
        
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_samples = 0
        
        # 计算准确率
        value_mae = 0.0
        policy_accuracy = 0.0
        
        with torch.no_grad():
            for board_tensors, policy_targets, value_targets in val_loader:
                # 移动到设备
                board_tensors = board_tensors.to(self.device, non_blocking=True)
                policy_targets = policy_targets.to(self.device, non_blocking=True)
                value_targets = value_targets.to(self.device, non_blocking=True)
                
                # 前向传播
                with torch.autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
                    value_pred, policy_pred = self.model(board_tensors)
                    
                    # 计算损失
                    value_loss = F.mse_loss(value_pred.squeeze(), value_targets)
                    policy_loss = F.cross_entropy(policy_pred, policy_targets)
                    
                    # 总损失
                    loss = (self.config.value_loss_weight * value_loss + 
                           self.config.policy_loss_weight * policy_loss)
                
                # 统计
                batch_size = board_tensors.size(0)
                total_loss += loss.item() * batch_size
                total_value_loss += value_loss.item() * batch_size
                total_policy_loss += policy_loss.item() * batch_size
                total_samples += batch_size
                
                # 计算准确率
                value_mae += F.l1_loss(value_pred.squeeze(), value_targets).item() * batch_size
                
                # 策略准确率（top-1）
                policy_pred_labels = torch.argmax(policy_pred, dim=1)
                policy_target_labels = torch.argmax(policy_targets, dim=1)
                policy_accuracy += (policy_pred_labels == policy_target_labels).float().sum().item()
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        avg_value_loss = total_value_loss / total_samples
        avg_policy_loss = total_policy_loss / total_samples
        avg_value_mae = value_mae / total_samples
        avg_policy_accuracy = policy_accuracy / total_samples
        
        # 学习率调度器步骤（对于ReduceLROnPlateau）
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        
        # TensorBoard记录
        if self.writer:
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
            self.writer.add_scalar('Val/ValueLoss', avg_value_loss, epoch)
            self.writer.add_scalar('Val/PolicyLoss', avg_policy_loss, epoch)
            self.writer.add_scalar('Val/ValueMAE', avg_value_mae, epoch)
            self.writer.add_scalar('Val/PolicyAccuracy', avg_policy_accuracy, epoch)
        
        return {
            'loss': avg_loss,
            'value_loss': avg_value_loss,
            'policy_loss': avg_policy_loss,
            'value_mae': avg_value_mae,
            'policy_accuracy': avg_policy_accuracy
        }
    
    def train(
        self,
        training_dataset: TrainingDataset,
        validation_dataset: Optional[TrainingDataset] = None
    ) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            training_dataset: 训练数据集
            validation_dataset: 验证数据集
            
        Returns:
            Dict[str, List[float]]: 训练历史
        """
        self.logger.info(f"开始训练，共 {self.config.num_epochs} 个epoch")
        self.logger.info(f"训练样本数: {len(training_dataset)}")
        if validation_dataset:
            self.logger.info(f"验证样本数: {len(validation_dataset)}")
        
        # 创建数据加载器
        train_dataset = ChessTrainingDataset(training_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=True
        )
        
        val_loader = None
        if validation_dataset:
            val_dataset = ChessTrainingDataset(validation_dataset)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                prefetch_factor=self.config.prefetch_factor
            )
        
        # 训练历史
        history = defaultdict(list)
        
        # 训练循环
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = {}
            if val_loader and (epoch + 1) % self.config.validation_interval == 0:
                val_metrics = self.validate(val_loader, epoch)
            
            # 记录历史
            for key, value in train_metrics.items():
                history[f'train_{key}'].append(value)
            
            for key, value in val_metrics.items():
                history[f'val_{key}'].append(value)
            
            # 保存最佳模型
            current_loss = val_metrics.get('loss', train_metrics['loss'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(epoch, {'loss': current_loss}, is_best=True)
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, train_metrics)
            
            # 早停检查
            if self.early_stopping and val_metrics:
                if self.early_stopping(val_metrics['loss']):
                    self.logger.info(f"早停触发，在epoch {epoch}")
                    break
            
            # 记录epoch信息
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch} 完成 ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.6f}"
                + (f", Val Loss: {val_metrics['loss']:.6f}" if val_metrics else "")
            )
        
        # 保存最终模型
        self.save_checkpoint(self.current_epoch, train_metrics, is_final=True)
        
        # 关闭TensorBoard
        if self.writer:
            self.writer.close()
        
        self.logger.info("训练完成")
        return dict(history)
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_final: bool = False
    ):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            metrics: 当前指标
            is_best: 是否是最佳模型
            is_final: 是否是最终模型
        """
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'global_step': self.global_step,
            'config': self.config.to_dict(),
            'metrics': metrics
        }
        
        # 添加EMA状态
        if self.model_ema:
            checkpoint['model_ema_state_dict'] = self.model_ema.shadow
        
        # 保存检查点
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"保存最佳模型到 {checkpoint_path}")
        
        if is_final:
            checkpoint_path = self.checkpoint_dir / 'final_model.pth'
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"保存最终模型到 {checkpoint_path}")
        
        # 定期检查点
        if not is_best and not is_final:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            
            # 清理旧检查点
            self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            Dict[str, Any]: 检查点信息
        """
        self.logger.info(f"加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载scaler状态
        if checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 加载EMA状态
        if self.model_ema and checkpoint.get('model_ema_state_dict'):
            self.model_ema.shadow = checkpoint['model_ema_state_dict']
        
        # 恢复训练状态
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        self.logger.info(f"检查点加载完成，从epoch {self.current_epoch} 继续训练")
        
        return checkpoint
    
    def _cleanup_checkpoints(self):
        """清理旧的检查点文件"""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoint_files) > self.config.max_checkpoints:
            # 按修改时间排序，删除最旧的
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            
            for old_checkpoint in checkpoint_files[:-self.config.max_checkpoints]:
                old_checkpoint.unlink()
                self.logger.debug(f"删除旧检查点: {old_checkpoint}")
    
    def get_model_for_inference(self) -> ChessNet:
        """
        获取用于推理的模型
        
        Returns:
            ChessNet: 推理模型
        """
        if self.model_ema:
            # 使用EMA模型
            self.model_ema.apply_shadow()
            model = self.model
            self.model_ema.restore()
            return model
        else:
            return self.model
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'writer') and self.writer:
            self.writer.close()