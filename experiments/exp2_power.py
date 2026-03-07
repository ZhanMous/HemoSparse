# -*- coding: utf-8 -*-
"""
实验2：SNN 稀疏性与低功耗关联性验证
- 实测功耗：基于 pynvml 读取现代GPU的实际功耗 (mJ) 与 延迟 (ms)
- 关联分析：验证稀疏性与功耗的负相关性
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    print("  [警告] pynvml 未安装，无法测量功耗")
    NVML_AVAILABLE = False
except pynvml.NVMLError:
    print("  [警告] NVML 初始化失败，无法测量功耗")
    NVML_AVAILABLE = False

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

def measure_gpu_power(device_id=0):
    """测量GPU功耗 (mW)"""
    if not NVML_AVAILABLE:
        return None
        
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        return power
    except Exception as e:
        print(f"  [警告] 读取功耗失败: {e}")
        return None


def run_experiment_2_power():
    """运行功耗实验：SNN 稀疏性与低功耗关联性验证"""
    print("\n实验2：SNN 稀疏性与低功耗关联性验证")
    print("-" * 50)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed()
    
    # 加载模型
    print("  加载模型...")
    snn_model = SNN(T=20).to(DEVICE)
    dense_snn_model = DenseSNN(T=20).to(DEVICE)
    ann_model = ANN().to(DEVICE)
    
    # 设置为评估模式
    snn_model.eval()
    dense_snn_model.eval()
    ann_model.eval()
    
    # 加载测试数据 (小批量用于测试)
    print("  加载测试数据...")
    test_loader_snn, _, _, _ = get_blood_mnist_loaders(batch_size=16, T=20, mode='snn', augment=False)
    test_loader_ann, _, _, _ = get_blood_mnist_loaders(batch_size=16, T=20, mode='ann', augment=False)
    
    # 测试少量样本
    max_samples = 100  # 限制样本数量以加快实验
    sample_count = 0
    
    device_id = 0  # 默认GPU设备ID
    
    print(f"  开始测量功耗与延迟 (最大样本数: {max_samples})")
    
    results = {
        'model_type': [],
        'sample_id': [],
        'energy_mj': [],
        'latency_ms': [],
        'sparsity': [],
        'avg_firing_rate': []
    }
    
    # 根据是否可用决定是否测量功耗
    if not NVML_AVAILABLE:
        print("  [警告] NVML 不可用，仅测量延迟")
    
    for batch_idx, ((inputs_snn, _), (inputs_ann, _)) in enumerate(zip(test_loader_snn, test_loader_ann)):
        if sample_count >= max_samples:
            break
            
        inputs_snn = inputs_snn.to(DEVICE)
        inputs_ann = inputs_ann.to(DEVICE)
        
        # 对每种模型进行测试
        models_config = [
            ('SNN (Sparse)', snn_model, inputs_snn, True),
            ('Dense_SNN', dense_snn_model, inputs_snn, True),
            ('ANN', ann_model, inputs_ann, False)
        ]
        
        for model_name, model, inputs, is_snn in models_config:
            # 获取稀疏性信息
            if is_snn:
                functional.reset_net(model)
                sp_monitor = SparsityMonitor(model)
                _ = model(inputs)
                stats = sp_monitor.get_sparsity_stats()
                sparsity = stats.get('global_sparsity', 0.0)
                avg_rate = stats.get('global_avg_rate', 1.0)
            else:
                sparsity = 0.0
                avg_rate = 1.0
                
            # 预热
            for _ in range(POWER_WARMUP_ITERS):
                if is_snn: 
                    functional.reset_net(model)
                with torch.no_grad():
                    _ = model(inputs)
            
            torch.cuda.synchronize()
            
            # 测量延迟和功耗
            start_time = time.time()
            if NVML_AVAILABLE:
                start_power = measure_gpu_power(device_id)
            else:
                start_power = None
                
            with torch.no_grad():
                outputs = model(inputs)
                
            torch.cuda.synchronize()
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            if NVML_AVAILABLE:
                end_power = measure_gpu_power(device_id)
                avg_power = (start_power + end_power) / 2 if start_power and end_power else None
                energy_mj = avg_power * (end_time - start_time) / 1000 if avg_power else None
            else:
                energy_mj = None
                
            # 记录结果（按样本记录）
            batch_size = inputs.size(0)
            for i in range(min(batch_size, max_samples - sample_count)):
                results['model_type'].append(model_name)
                results['sample_id'].append(sample_count)
                results['energy_mj'].append(energy_mj)
                results['latency_ms'].append(latency_ms)
                results['sparsity'].append(sparsity)
                results['avg_firing_rate'].append(avg_rate)
                sample_count += 1
                
            if sample_count >= max_samples:
                break
    
    # 保存结果
    df = pd.DataFrame(results)
    save_path = os.path.join(RESULTS_DIR, 'exp2_power_and_latency.csv')
    df.to_csv(save_path, index=False)
    print(f"\n实验2完成！结果已保存至 {save_path}")
    
    # 输出摘要统计
    if not df.empty:
        print("\n实验结果摘要:")
        print(df.groupby('model_type')[['energy_mj', 'latency_ms', 'sparsity']].agg(['mean', 'std']))

if __name__ == '__main__':
    run_experiment_2_power()
