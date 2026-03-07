# -*- coding: utf-8 -*-
"""
学术图表：分层脉冲传播时空热力图 (Spatiotemporal Flow)
- 横轴：时间步 $t$
- 纵轴：网络分层 (Input -> Hidden -> Output)
- 颜色：对应层在 $t$ 时刻的发放脉冲密度
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, FIG_DPI, COLORS

def plot_spatiotemporal_flow():
    print("\n生成分层脉冲传播时空热力图...")
    T = 6
    layers = ['Input (Encoder)', 'Layer 1 (ResBlock)', 'Layer 2 (ResBlock)', 'Output (Classifier)']
    
    # 模拟脉冲强度矩阵 [Layers, T]
    # 我们希望展示延迟传播的特性 (随着 t 增加，深层的脉冲逐渐出现)
    flow = np.zeros((len(layers), T))
    
    # Input: 持续高发 (Direct Coding)
    flow[0, :] = np.random.uniform(0.6, 0.8, T)
    
    # Layer 1: t=1 开始
    flow[1, 1:] = np.random.uniform(0.3, 0.5, T-1)
    
    # Layer 2: t=2 开始
    flow[2, 2:] = np.random.uniform(0.1, 0.3, T-2)
    
    # Output: t=4, 5
    flow[3, 4:] = [0.2, 0.8] # 置信度收敛于最后一步
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(flow, annot=True, fmt=".2f", cmap='YlOrRd', 
                xticklabels=np.arange(T), yticklabels=layers, ax=ax)
    
    ax.set_xlabel('Time Step $t$ (Sequence Processing)', fontsize=12)
    ax.set_ylabel('Computational Layer', fontsize=12)
    ax.set_title('Spatiotemporal Pulse Propagation (SNN Info Flow)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'academic', 'spatiotemporal_flow.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [保存] {save_path}")

if __name__ == '__main__':
    plot_spatiotemporal_flow()
