# -*- coding: utf-8 -*-
"""
HemoSparse 模型定义
包含三个对等结构的模型：SNN、DenseSNN、ANN
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional

class PLIFNode(neuron.LIFNode):
    """PLIF 神经元，α 为可学习参数"""
    def __init__(self, tau=2.0, v_threshold=1.0, surrogate_function=None, step_mode='m'):
        super().__init__(tau=tau, v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
        self.alpha = nn.Parameter(torch.tensor(1.0 / tau))
    
    def forward(self, x):
        self.tau = 1.0 / self.alpha
        return super().forward(x)

class SpikingResBlock(nn.Module):
    """脉冲残差块"""
    def __init__(self, in_channels, out_channels, stride=1, v_threshold=1.0, step_mode='m'):
        super().__init__()
        self.step_mode = step_mode
        surrogate_function = surrogate.ATan()
        
        self.conv_bn_lif = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, step_mode=step_mode),
            layer.BatchNorm2d(out_channels, step_mode=step_mode),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, step_mode=step_mode),
            layer.BatchNorm2d(out_channels, step_mode=step_mode)
        )
        self.lif = PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode=step_mode)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, step_mode=step_mode),
                layer.BatchNorm2d(out_channels, step_mode=step_mode)
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
        
        surrogate_function = surrogate.ATan()
        
        self.stem = nn.Sequential(
            layer.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False, step_mode='m'),
            layer.BatchNorm2d(20, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m'),
            layer.MaxPool2d(2, 2, step_mode='m')
        )
        
        self.layer1 = SpikingResBlock(20, 41, stride=2, v_threshold=v_threshold, step_mode='m')
        self.layer2 = SpikingResBlock(41, 82, stride=2, v_threshold=v_threshold, step_mode='m')
        
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode='m')
        self.flatten = layer.Flatten(step_mode='m')
        
        self.classifier = nn.Sequential(
            layer.Linear(82, num_classes, bias=False, step_mode='m'),
            PLIFNode(v_threshold=v_threshold, surrogate_function=surrogate_function, step_mode='m')
        )

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

class NonSparsePLIF(nn.Module):
    """无稀疏计算的PLIF：输出膜电位，而非脉冲，彻底关闭稀疏逻辑"""
    def __init__(self, tau=2.0, v_threshold=1.0, reset_mode='zero'):
        super().__init__()
        self.decay = nn.Parameter(torch.tensor(1.0 - 1.0 / tau))
        self.v_threshold = v_threshold
        self.reset_mode = reset_mode
        self.alpha = nn.Parameter(torch.tensor(1.0 / tau))
        self.v = None
    
    def forward(self, x_seq: torch.Tensor):
        T, N, C, H, W = x_seq.shape
        
        self.decay.data = 1.0 - self.alpha.data
        
        if self.v is None or self.v.shape != (N, C, H, W):
            self.v = torch.zeros((N, C, H, W), device=x_seq.device)
        
        v_seq = []
        for t in range(T):
            x = x_seq[t]
            
            self.v = self.v * self.decay + x
            
            if self.reset_mode == 'zero':
                reset_mask = (self.v >= self.v_threshold).float()
                self.v = self.v * (1 - reset_mask)
            
            v_seq.append(self.v.clone())
        
        return torch.stack(v_seq, dim=0)
    
    def reset(self):
        self.v = None

class NonSparseSpikingResBlock(nn.Module):
    """无稀疏脉冲残差块"""
    def __init__(self, in_channels, out_channels, stride=1, v_threshold=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = NonSparsePLIF(v_threshold=v_threshold)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = NonSparsePLIF(v_threshold=v_threshold)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x_seq: torch.Tensor):
        T, N, C, H, W = x_seq.shape
        
        out_seq = []
        for t in range(T):
            x = x_seq[t]
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.lif1(out.unsqueeze(0))[0]
            
            out = self.conv2(out)
            out = self.bn2(out)
            
            shortcut = self.shortcut(x)
            out += shortcut
            
            out = self.lif2(out.unsqueeze(0))[0]
            out_seq.append(out)
        
        return torch.stack(out_seq, dim=0)

class DenseSNN(nn.Module):
    """DenseSNN 模型
    - 与 SNN 完全相同的拓扑结构和参数量
    - 唯一区别：完全关闭稀疏计算，强制所有神经元参与计算
    - 使用普通 PyTorch 层（无 SpikingJelly 稀疏优化）+ NonSparsePLIF
    - 用于与 SNN 对照，评估稀疏计算的效果
    """
    def __init__(self, in_channels=3, num_classes=8, T=6, v_threshold=1.0):
        super().__init__()
        self.T = T
        self.v_threshold = v_threshold
        
        self.conv1 = nn.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(20)
        self.lif1 = NonSparsePLIF(v_threshold=v_threshold)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.layer1 = NonSparseSpikingResBlock(20, 41, stride=2, v_threshold=v_threshold)
        self.layer2 = NonSparseSpikingResBlock(41, 82, stride=2, v_threshold=v_threshold)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(82, num_classes, bias=False)
        
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
        
        T, N, C, H, W = x.shape
        
        out_seq = []
        for t in range(T):
            xt = x[t]
            
            xt = self.conv1(xt)
            xt = self.bn1(xt)
            xt = self.lif1(xt.unsqueeze(0))[0]
            xt = self.pool1(xt)
            
            xt = xt.unsqueeze(0)
            xt = self.layer1(xt)[0]
            
            xt = xt.unsqueeze(0)
            xt = self.layer2(xt)[0]
            
            xt = self.avgpool(xt)
            xt = self.flatten(xt)
            out_seq.append(xt)
        
        out = torch.stack(out_seq, dim=0).mean(dim=0)
        out = self.classifier(out)
        return out
    
    def reset(self):
        for m in self.modules():
            if isinstance(m, NonSparsePLIF):
                m.reset()

class ANN(nn.Module):
    """ANN 模型
    - 同拓扑结构
    - ReLU 激活，无时间维度
    """
    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
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
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(82, num_classes, bias=False)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if x.dim() == 5:
            x = x.mean(dim=0)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x