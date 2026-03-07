# -*- coding: utf-8 -*-
"""
稀疏性统计钩子 (SpikingJelly 风格)
用于在 SNN 模型前向传播时，非侵入式地采集模型各层的脉冲发放率
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import monitor, neuron

class SparsityMonitor:
    def __init__(self, net: nn.Module):
        """
        初始化稀疏性统计监视器
        使用 SpikingJelly 原生 OutputMonitor，只监控 LIFNode
        """
        self.net = net
        self.monitor = monitor.OutputMonitor(net, neuron.LIFNode)
        self.monitor.enable()
        
    def get_sparsity_stats(self):
        """
        获取单次前向传播后的稀疏性统计信息
        返回字典: 包含各层稀疏度，以及整体的 时间/空间 稀疏度估计
        """
        stats = {}
        layer_rates = []
        
        # SpikingJelly monitor.records 是一个列表，列表的每个元素对应一个被监控的神经元层出的记录列表
        monitored_names = [name for name, _ in self.net.named_modules() if isinstance(_, neuron.LIFNode)]
        
        for i, module_records in enumerate(self.monitor.records):
            if len(module_records) > 0:
                # module_records 也是一个 list，存储了该层历次前向传播的输出
                out_spikes = module_records[-1] # 取最后一次前向传播的脉冲数据
                
                # 发放率 = 脉冲均值 (脉冲为 0 或 1)
                firing_rate = float(out_spikes.mean().item())
                name = monitored_names[i] if i < len(monitored_names) else f"layer_{i}"
                stats[name] = firing_rate
                layer_rates.append(firing_rate)
                
        if layer_rates:
            avg_rate = sum(layer_rates) / len(layer_rates)
            stats['global_avg_rate'] = avg_rate
            stats['global_sparsity'] = 1.0 - avg_rate
            
        # 清除记录，准备下一次前向传播
        self.monitor.clear_recorded_data()
        return stats
        
    def disable(self):
        """禁用钩子，避免不测试稀疏性时消耗资源"""
        self.monitor.disable()
        
    def enable(self):
        """启用钩子"""
        self.monitor.enable()
        self.monitor.clear_recorded_data()
