# -*- coding: utf-8 -*-
"""
同拓扑ANN（对照组C）
- 结构与标准 SNN 100% 相同，保持完全一致的参数量
- LIFNode替换为nn.ReLU
- 移除时间维度的循环处理
- 纯粹的前馈神经网络，作为最终的性能与功耗基线
"""

import math
import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), # 替换 LIFNode
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), # 替换 LIFNode
            nn.MaxPool2d(2, 2),
            
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, num_classes, bias=False),
            # 最后一层一般不加激活函数，因为交叉熵损失中包含 Softmax
            # 但为了严格对应 SNN 最后一层的 LIF，这里也可以加一个 ReLU 
            # (不过通常分类器最后一层直接输出 logits 即可)
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
        """
        前向传播
        x: [N, C, H, W] 只有空间维度 (或者如果是从 SNN 数据加载器来的是 [T, N, C, H, W]，则取 mean)
        """
        if x.dim() == 5:
            # 如果加载的是经过泊松编码的数据 [T, N, C, H, W]，取平均转化为像素强度 [N, C, H, W]
            x = x.mean(dim=0)
            
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
