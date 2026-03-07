# -*- coding: utf-8 -*-
"""
改进版SNN (实验组A-Plus)
- 引入 MS-ResNet 结构
- 替换 LIFNode 为 ParametricLIFNode (PLIF)
- 适配 Direct Coding (第一层直接接收模拟值)
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional, surrogate

class SpikingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, tau=2.0):
        super().__init__()
        self.conv_bn_lif = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_channels, step_mode='m'),
            neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan(), step_mode='m'),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_channels, step_mode='m'),
        )
        self.lif = neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, step_mode='m'),
                layer.BatchNorm2d(out_channels, step_mode='m')
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # x shape: [T, N, C, H, W]
        out = self.conv_bn_lif(x)
        out += self.shortcut(x)
        out = self.lif(out)
        return out

class ImprovedSNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, T=6, tau=2.0):
        super().__init__()
        self.T = T
        
        # 第一层作为 Direct Encoder: Conv -> BN -> PLIF
        # 它可以接收图像副本 (Direct Coding)，PLIF 会将其转换为脉冲传递给深层
        self.stem = nn.Sequential(
            layer.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(32, step_mode='m'),
            neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan(), step_mode='m'),
            layer.MaxPool2d(2, 2, step_mode='m') # -> 32x14x14
        )
        
        # 残差阶段
        self.layer1 = SpikingResBlock(32, 64, stride=2, tau=tau)  # -> 64x7x7
        self.layer2 = SpikingResBlock(64, 128, stride=2, tau=tau) # -> 128x3x3
        
        # 全局平均池化
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.flatten = layer.Flatten(step_mode='m')
        
        # 分类器
        self.classifier = nn.Sequential(
            layer.Linear(128, num_classes, bias=False, step_mode='m'),
            neuron.ParametricLIFNode(init_tau=tau, surrogate_function=surrogate.ATan(), step_mode='m')
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        # x: [N, T, C, H, W] from DataLoader
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            x = x.transpose(0, 1) # -> [T, N, C, H, W]
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x.mean(dim=0)
