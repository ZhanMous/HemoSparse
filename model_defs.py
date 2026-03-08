# -*- coding: utf-8 -*-
"""
HemoSparse 模型定义
包含三个对等结构的模型：SNN、DenseSNN、ANN
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate

class PLIFNode(neuron.LIFNode):
    """PLIF 神经元，α 为可学习参数"""
    def __init__(self, tau=2.0, v_threshold=1.0, surrogate_function=None):
        super().__init__(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function)
        # α 是可学习参数，对应膜时间常数 τ
        self.alpha = nn.Parameter(torch.tensor(1.0 / tau))
    
    def forward(self, x):
        self.tau = 1.0 / self.alpha
        return super().forward(x)

class SpikingResBlock(nn.Module):
    """脉冲残差块"""
    def __init__(self, in_channels, out_channels, stride=1, v_threshold=1.0):
        super().__init__()
        surrogate_function = surrogate.ATan()
        
        self.conv_bn_lif = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_channels, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(out_channels, step_mode='m')
        )
        self.lif = PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, step_mode='m'),
                layer.BatchNorm2d(out_channels, step_mode='m')
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv_bn_lif(x)
        out += self.shortcut(x)
        out = self.lif(out)
        return out

class SNN(nn.Module):
    """SNN 模型
    - MS-ResNet 结构
    - PLIF 神经元
    - 时间步 T=6
    - 保持稀疏计算优化
    """
    def __init__(self, in_channels=3, num_classes=8, T=6, v_threshold=1.0):
        super().__init__()
        self.T = T
        
        # 替代梯度函数
        surrogate_function = surrogate.ATan()
        
        # 第一层
        self.stem = nn.Sequential(
            layer.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(20, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function),
            layer.MaxPool2d(2, 2, step_mode='m') # -> 20x14x14
        )
        
        # 残差阶段
        self.layer1 = SpikingResBlock(20, 41, stride=2, v_threshold=v_threshold)  # -> 41x7x7
        self.layer2 = SpikingResBlock(41, 82, stride=2, v_threshold=v_threshold) # -> 82x3x3
        
        # 全局平均池化
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.flatten = layer.Flatten(step_mode='m')
        
        # 分类器
        self.classifier = nn.Sequential(
            layer.Linear(82, num_classes, bias=False, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """前向传播
        x: [N, T, C, H, W] 或 [N, C, H, W]
        """
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            x = x.transpose(0, 1)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x.mean(dim=0)

class DenseSNN(nn.Module):
    """DenseSNN 模型
    - 与 SNN 同结构、同阈值、同训练
    - 关闭稀疏计算，强制全张量稠密计算
    """
    def __init__(self, in_channels=3, num_classes=8, T=6, v_threshold=1.0):
        super().__init__()
        self.T = T
        
        # 替代梯度函数
        surrogate_function = surrogate.ATan()
        
        # 第一层
        self.stem = nn.Sequential(
            layer.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(20, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function),
            layer.MaxPool2d(2, 2, step_mode='m')
        )
        
        # 残差阶段
        self.layer1 = SpikingResBlock(20, 41, stride=2, v_threshold=v_threshold)
        self.layer2 = SpikingResBlock(41, 82, stride=2, v_threshold=v_threshold)
        
        # 全局平均池化
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.flatten = layer.Flatten(step_mode='m')
        
        # 分类器
        self.classifier = nn.Sequential(
            layer.Linear(82, num_classes, bias=False, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        elif x.dim() == 5:
            x = x.transpose(0, 1)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x.mean(dim=0)

class ANN(nn.Module):
    """ANN 模型
    - 同拓扑结构
    - ReLU 激活，无时间维度
    """
    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()
        
        # 第一层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 残差阶段
        class ResBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv_bn_relu = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                else:
                    self.shortcut = nn.Identity()
            def forward(self, x):
                out = self.conv_bn_relu(x)
                out += self.shortcut(x)
                out = nn.ReLU(inplace=True)(out)
                return out
        
        self.layer1 = ResBlock(20, 41, stride=2)
        self.layer2 = ResBlock(41, 82, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(82, num_classes, bias=False)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """前向传播
        x: [N, C, H, W] 或 [T, N, C, H, W]
        """
        if x.dim() == 5:
            x = x.mean(dim=0)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
