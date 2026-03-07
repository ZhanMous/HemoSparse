# -*- coding: utf-8 -*-
"""
标准SNN（实验组A）
- SpikingJelly 原生 API (step_mode='m')
- 保留 SNN 的天然时间/空间稀疏性
- 结构: 2层卷积 + 1层全连接
"""

import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional, surrogate

class SNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, T=20, tau=2.0, v_threshold=1.0):
        super().__init__()
        self.T = T
        
        # 替代梯度函数，使用 ATan 保证训练稳定
        surrogate_function = surrogate.ATan()
        
        # 特征提取网络 (Conv -> BatchNorm -> LIF -> MaxPool)
        self.feature_extractor = nn.Sequential(
            # Conv1: 3x28x28 -> 16x28x28
            layer.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(16, step_mode='m'),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m'),
            layer.MaxPool2d(2, 2, step_mode='m'), # -> 16x14x14
            
            # Conv2: 16x14x14 -> 32x14x14
            layer.Conv2d(16, 32, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(32, step_mode='m'),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m'),
            layer.MaxPool2d(2, 2, step_mode='m'), # -> 32x7x7
            
            layer.Flatten(step_mode='m')
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            layer.Linear(32 * 7 * 7, num_classes, bias=False, step_mode='m'),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m')
        )

        # 初始化权重 (Kaiming Normal)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        x: [N, T, C, H, W] (来自 DataLoader)
        """
        # 如果输入是 [N, C, H, W]，则扩展时间维度为 [T, N, C, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            # 数据加载器返回 [N, T, C, H, W]，转置为 [T, N, C, H, W] 以适配 SpikingJelly layer.step_mode='m'
            x = x.transpose(0, 1)
            
        x = self.feature_extractor(x)
        x = self.classifier(x)
        
        # 返回时间维度的平均值进行分类 (x shape after classifier is [T, N, classes])
        return x.mean(dim=0)
