# -*- coding: utf-8 -*-
"""
稠密SNN（对照组B）
- 结构与标准 SNN 100% 相同
- 将 LIFNode 的阈值设为极小值（如 0.001），强制神经元全时间步发放脉冲
- 完全破坏 SNN 的天然稀疏性，作为基线对比
"""

import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional, surrogate

class DenseSNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, T=20, tau=2.0):
        super().__init__()
        self.T = T
        
        # 极小的阈值，几乎任意输入都会触发脉冲
        v_threshold = 0.001 
        surrogate_function = surrogate.ATan()
        
        # 特征提取网络 (完全相同的拓扑)
        self.feature_extractor = nn.Sequential(
            layer.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(16, step_mode='m'),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m'),
            layer.MaxPool2d(2, 2, step_mode='m'),
            
            layer.Conv2d(16, 32, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(32, step_mode='m'),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m'),
            layer.MaxPool2d(2, 2, step_mode='m'),
            
            layer.Flatten(step_mode='m')
        )
        
        self.classifier = nn.Sequential(
            layer.Linear(32 * 7 * 7, num_classes, bias=False, step_mode='m'),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m')
        )

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
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            x = x.transpose(0, 1)
            
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x.mean(dim=0)
