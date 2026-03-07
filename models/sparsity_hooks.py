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
        
        # 检查 monitor.records 的类型并相应处理
        records = self.monitor.records
        try:
            # 尝试作为字典处理
            for mod, module_records in records.items():
                if len(module_records) > 0:
                    out_spikes = module_records[-1] # 取最后一次前向传播的脉冲数据
                    firing_rate = float(out_spikes.mean().item())
                    
                    # 查找模块名
                    found_name = f"layer_{len(layer_rates)}"
                    for name, m in self.net.named_modules():
                        if m is mod:
                            found_name = name
                            break
                    
                    stats[found_name] = firing_rate
                    layer_rates.append(firing_rate)
        except AttributeError:
            # 如果不是字典，假设是列表
            for i, module_records in enumerate(records):
                if len(module_records) > 0:
                    out_spikes = module_records[-1] # 取最后一次前向传播的脉冲数据
                    firing_rate = float(out_spikes.mean().item())
                    
                    # 为列表元素生成名称
                    found_name = f"layer_{i}"
                    stats[found_name] = firing_rate
                    layer_rates.append(firing_rate)
                
        if layer_rates:
            avg_rate = sum(layer_rates) / len(layer_rates)
            stats['global_avg_rate'] = avg_rate
            stats['global_sparsity'] = 1.0 - avg_rate
        else:
            # 如果没有记录任何层，默认完全稀疏
            stats['global_avg_rate'] = 0.0
            stats['global_sparsity'] = 1.0
            
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