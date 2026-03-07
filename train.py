# -*- coding: utf-8 -*-
"""
HemoSparse 统一训练主流程
- 支持三组对照模型（SNN, DenseSNN, ANN）
- RTX 4070 原生优化（显存自适应, AMP混合精度）
- 稀疏性监控与日志持久化
"""

import os
import sys
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from spikingjelly.activation_based import functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, NUM_EPOCHS, SNN_LR, ANN_LR, WEIGHT_DECAY, MOMENTUM,
    RESULTS_DIR, CHECKPOINT_DIR, USE_AMP, set_seed
)
from data.dataloader import get_blood_mnist_loaders
from models.snn_model import SNN
from models.dense_snn_model import DenseSNN
from models.ann_model import ANN
from models.sparsity_hooks import SparsityMonitor

class Trainer:
    def __init__(self, model_type='snn', T=20, batch_size=None):
        self.model_type = model_type.lower()
        self.T = T
        set_seed()
        
        # 1. 准备数据
        mode = 'ann' if self.model_type == 'ann' else 'snn'
        self.train_loader, self.val_loader, self.test_loader, self.info = \
            get_blood_mnist_loaders(batch_size=batch_size, T=T, mode=mode, augment=True)
            
        # 2. 初始化模型
        if self.model_type == 'snn':
            self.model = SNN(in_channels=3, num_classes=8, T=T).to(DEVICE)
            self.optimizer = optim.SGD(self.model.parameters(), lr=SNN_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            self.is_snn = True
        elif self.model_type == 'densesnn':
            self.model = DenseSNN(in_channels=3, num_classes=8, T=T).to(DEVICE)
            self.optimizer = optim.SGD(self.model.parameters(), lr=SNN_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            self.is_snn = True
        elif self.model_type == 'ann':
            self.model = ANN(in_channels=3, num_classes=8).to(DEVICE)
            self.optimizer = optim.Adam(self.model.parameters(), lr=ANN_LR, weight_decay=WEIGHT_DECAY)
            self.is_snn = False
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        # 3. 损失函数与混合精度
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled=USE_AMP)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS)
        
        # 4. 稀疏性监控 (只针对 SNN)
        self.sparsity_monitor = SparsityMonitor(self.model) if self.is_snn else None
        
        # 5. 日志与最佳指标
        self.history = []
        self.best_acc = 0.0
        self.start_time = time.time()
        
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        batch_sparsities = []
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).squeeze()
            
            self.optimizer.zero_grad()
            
            # AMP 混合精度前向传播
            with autocast(device_type='cuda', enabled=USE_AMP):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 重置 SNN 状态
            if self.is_snn:
                functional.reset_net(self.model)
                if self.sparsity_monitor:
                    stats = self.sparsity_monitor.get_sparsity_stats()
                    batch_sparsities.append(stats['global_sparsity'])
                    
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        avg_sparsity = sum(batch_sparsities)/len(batch_sparsities) if batch_sparsities else 0.0
        
        # 释放显存
        torch.cuda.empty_cache()
        return epoch_loss, epoch_acc, avg_sparsity
        
    def _evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        batch_sparsities = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).squeeze()
                
                with autocast(device_type='cuda', enabled=USE_AMP):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                if self.is_snn:
                    functional.reset_net(self.model)
                    if self.sparsity_monitor:
                        stats = self.sparsity_monitor.get_sparsity_stats()
                        batch_sparsities.append(stats['global_sparsity'])
                        
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        epoch_loss = total_loss / len(loader)
        epoch_acc = 100. * correct / total
        avg_sparsity = sum(batch_sparsities)/len(batch_sparsities) if batch_sparsities else 0.0
        
        return epoch_loss, epoch_acc, avg_sparsity

    def train(self):
        print(f"\n[{self.model_type.upper()}] 开始训练 | Device: {DEVICE} | AMP: {USE_AMP}")
        
        for epoch in range(NUM_EPOCHS):
            # 训练
            train_loss, train_acc, train_sp = self._train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_sp = self._evaluate(self.val_loader)
            
            # 学习率调整
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            
            # 记录最佳
            is_best = False
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                is_best = True
                self._save_checkpoint(is_best=True)
                
            # 打印日志
            log_str = (f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                       f"LR: {lr:.4f} | "
                       f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% ")
            if self.is_snn:
                log_str += f"Sp: {train_sp:.3f} | "
            log_str += f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%{' [*]' if is_best else ''}"
            print(log_str)
            
            # 保存到历史记录
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss, 'train_acc': train_acc, 'train_sparsity': train_sp,
                'val_loss': val_loss, 'val_acc': val_acc, 'val_sparsity': val_sp,
                'lr': lr
            })
            
        # 测试集最终评估
        test_loss, test_acc, test_sp = self._evaluate(self.test_loader)
        print(f"[{self.model_type.upper()}] 训练完成! 最佳验证准确率: {self.best_acc:.2f}% | 最终测试准确率: {test_acc:.2f}%")
        
        # 保存日志
        self._save_logs(test_acc)
        return self.best_acc, test_acc
        
    def _save_checkpoint(self, is_best=False):
        ckpt = {
            'model_state': self.model.state_dict(),
            'model_type': self.model_type,
            'best_acc': self.best_acc,
            'T': self.T
        }
        path = os.path.join(CHECKPOINT_DIR, f"{self.model_type}_T{self.T}.pth")
        torch.save(ckpt, path)
        
    def _save_logs(self, test_acc):
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(RESULTS_DIR, f"{self.model_type}_T{self.T}_history.csv"), index=False)
        
        # 记录汇总结果
        summary = {
            'Model': self.model_type.upper(),
            'T': self.T,
            'Params(M)': sum(p.numel() for p in self.model.parameters()) / 1e6,
            'Best_Val_Acc': self.best_acc,
            'Test_Acc': test_acc,
            'Train_Time(s)': time.time() - self.start_time
        }
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(RESULTS_DIR, 'training_summary.csv')
        
        if os.path.exists(summary_path):
            summary_df.to_csv(summary_path, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)

def run_all_models():
    """自动化跑完三组对照训练"""
    print("=" * 60)
    print("开始 HemoSparse 三组对照模型自动化训练")
    print("=" * 60)
    
    # 确保保存路径存在
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 清空旧的汇总文件
    summary_path = os.path.join(RESULTS_DIR, 'training_summary.csv')
    if os.path.exists(summary_path):
        os.remove(summary_path)

    epochs_original = NUM_EPOCHS 
    
    # 1. 对照组C：ANN
    trainer_ann = Trainer(model_type='ann')
    trainer_ann.train()
    
    # 2. 实验组A：标准SNN
    trainer_snn = Trainer(model_type='snn')
    trainer_snn.train()
    
    # 3. 对照组B：稠密SNN
    trainer_dense = Trainer(model_type='densesnn')
    trainer_dense.train()
    
    print("\n所有模型训练完成！结果见 outputs/results/training_summary.csv")

if __name__ == '__main__':
    run_all_models()
