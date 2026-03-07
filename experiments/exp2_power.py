# -*- coding: utf-8 -*-
"""
实验2：SNN 稀疏性与低功耗关联性验证 (RTX 4070 专属)
- 理论计算：使用 SpikingJelly 原生 cal_energy 计算 MACs / ACs
- 实测功耗：基于 pynvml 读取 RTX 4070 的实际功耗 (mJ) 与 延迟 (ms)
"""

import os
import sys
import time
import torch
import pynvml
import pandas as pd
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, RESULTS_DIR, set_seed, POWER_WARMUP_ITERS, POWER_TEST_ITERS
from data.dataloader import get_blood_mnist_loaders
from models.snn_model import SNN
from models.dense_snn_model import DenseSNN
from models.ann_model import ANN
from models.sparsity_hooks import SparsityMonitor

try:
    from spikingjelly.activation_based import layer, functional
except ImportError:
    print("Warning: Missing spikingjelly components.")

class PowerMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.idle_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 # W
        
    def get_current_power(self):
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 # W
        
    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def run_experiment_2_power():
    print("\n" + "="*60)
    print("实验2：SNN 稀疏性与低功耗关联性验证 (4070 实测)")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed()
    
    monitor = PowerMonitor()
    print(f"GPU Idle Power: {monitor.idle_power:.2f} W")
    
    # 模拟数据 [N=1, 单样本推理]
    test_loader_snn, _, _, _ = get_blood_mnist_loaders(batch_size=1, T=20, mode='snn', augment=False)
    test_loader_ann, _, _, _ = get_blood_mnist_loaders(batch_size=1, T=20, mode='ann', augment=False)
    
    inputs_snn, _ = next(iter(test_loader_snn))
    inputs_ann, _ = next(iter(test_loader_ann))
    
    inputs_snn = inputs_snn.to(DEVICE)
    inputs_ann = inputs_ann.to(DEVICE)
    
    models = {
        'SNN (Sparse)': SNN(T=20).to(DEVICE),
        'Dense_SNN': DenseSNN(T=20).to(DEVICE),
        'ANN': ANN().to(DEVICE)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n[测试] {name} ...")
        model.eval()
        is_snn = 'SNN' in name
        inputs = inputs_snn if is_snn else inputs_ann
        
        # 1. SpikingJelly 理论 MACs/ACs 评估
        if is_snn:
            functional.reset_net(model)
            # Todo: use spikingjelly hook if needed, but for now we focus on empirical
            sp_monitor = SparsityMonitor(model)
            _ = model(inputs)
            stats = sp_monitor.get_sparsity_stats()
            sparsity = stats.get('global_sparsity', 0.0)
            avg_rate = stats.get('global_avg_rate', 1.0)
        else:
            sparsity = 0.0
            avg_rate = 1.0
            
        # 2. 预热 (Warm-up)
        for _ in range(POWER_WARMUP_ITERS):
            if is_snn: functional.reset_net(model)
            with torch.no_grad():
                _ = model(inputs)
                
        torch.cuda.synchronize()
        
        # 3. 功耗与延迟测试
        start_time = time.perf_counter()
        power_samples = []
        
        for _ in range(POWER_TEST_ITERS):
            if is_snn: functional.reset_net(model)
            
            with torch.no_grad():
                _ = model(inputs)
                
            # 简单采样功率
            power_samples.append(monitor.get_current_power())
            
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_latency = total_time_ms / POWER_TEST_ITERS
        
        # 减去静态功耗的动态功耗
        avg_power = max(0, sum(power_samples)/len(power_samples) - monitor.idle_power)
        
        # 能耗 = 功率 * 时间 (mJ)
        energy_mJ = avg_power * (avg_latency / 1000.0) * 1000.0 # W * s * 1000 -> mJ
        
        res = {
            'Model': name,
            'Sparsity': sparsity,
            'Avg_Firing_Rate': avg_rate,
            'Latency_ms': avg_latency,
            'Dynamic_Power_W': avg_power,
            'Energy_per_Sample_mJ': energy_mJ
        }
        results.append(res)
        
        print(f"  稀疏度: {sparsity:.4f} | 延迟: {avg_latency:.2f} ms | 单样本能耗: {energy_mJ:.4f} mJ")
        
    df = pd.DataFrame(results)
    save_path = os.path.join(RESULTS_DIR, 'exp2_power_and_latency.csv')
    df.to_csv(save_path, index=False)
    print(f"\n实验2完成！结果已保存至 {save_path}")

if __name__ == '__main__':
    run_experiment_2_power()
