# -*- coding: utf-8 -*-
"""
实验1：SNN 天然稀疏性量化验证
- 基于 SpikingJelly 精准量化 标准SNN 的时间/空间稀疏性
- 分析稀疏度随训练轮次 T 的变化规律
- 分析神经元阈值与稀疏度的关系
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, DEFAULT_BATCH_SIZE, RESULTS_DIR, set_seed
from data.dataloader import get_blood_mnist_loaders
from models.snn_model import SNN
from models.sparsity_hooks import SparsityMonitor

def run_experiment_1_sparsity():
    print("\n" + "="*60)
    print("实验1：SNN 天然稀疏性量化验证")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed()
    
    results = []
    T_list = [10, 20, 30]
    threshold_list = [0.5, 1.0, 1.5, 2.0]
    
    # 模拟数据输入
    test_loader, _, _, _ = get_blood_mnist_loaders(batch_size=16, T=20, mode='snn', augment=False)
    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(DEVICE)
    
    print(f"开始测试不同 T 和 阈值 下的稀疏度...")
    
    for T in T_list:
        # 重塑输入时间步长
        inputs_t = inputs[:T, ...] if T <= inputs.shape[0] else inputs
        
        for v_th in threshold_list:
            model = SNN(T=T, v_threshold=v_th).to(DEVICE)
            model.eval()
            monitor = SparsityMonitor(model)
            
            with torch.no_grad():
                _ = model(inputs_t)
                
            stats = monitor.get_sparsity_stats()
            
            res = {
                'T': T,
                'v_threshold': v_th,
                'Global_Sparsity': stats.get('global_sparsity', 0.0),
                'Global_Avg_Rate': stats.get('global_avg_rate', 0.0)
            }
            
            # 添加各层信息
            for key, value in stats.items():
                if key not in ['global_sparsity', 'global_avg_rate']:
                    res[key] = value
                    
            results.append(res)
            print(f"  T={T:2d} | V_th={v_th:.1f} | 稀疏度={res['Global_Sparsity']:.4f}")
            
    df = pd.DataFrame(results)
    save_path = os.path.join(RESULTS_DIR, 'exp1_sparsity_quantification.csv')
    df.to_csv(save_path, index=False)
    print(f"\n实验1完成！结果已保存至 {save_path}")

if __name__ == '__main__':
    run_experiment_1_sparsity()
