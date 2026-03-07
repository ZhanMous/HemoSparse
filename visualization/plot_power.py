# -*- coding: utf-8 -*-
"""
低功耗分析可视化
- MACs / 功耗 / 延迟对比柱状图
- 稀疏度与功耗散点图
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, RESULTS_DIR, FIG_DPI, COLORS

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

def plot_power_metrics():
    """实测功耗与延迟对比柱状图"""
    file_path = os.path.join(RESULTS_DIR, 'exp2_power_and_latency.csv')
    if not os.path.exists(file_path):
        print(f"找不到 {file_path}")
        return
        
    df = pd.read_csv(file_path)
    
    models = df['Model'].tolist()
    models = [m.replace('SNN (Sparse)', 'SNN(A)').replace('Dense_SNN', 'DenseSNN(B)').replace('ANN', 'ANN(C)') for m in models]
    
    energy = df['Energy_per_Sample_mJ'].values
    latency = df['Latency_ms'].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1：能耗
    bars1 = ax1.bar(models, energy, color=COLORS, width=0.5, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('Energy per Sample (mJ)', fontsize=12)
    ax1.set_title('Inference Energy Consumption (RTX 4070)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05 * max(energy), 
                 f"{b.get_height():.2f}", ha='center', va='bottom', fontsize=10)
                 
    # 子图2：延迟
    bars2 = ax2.bar(models, latency, color=COLORS, width=0.5, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('Latency (ms)', fontsize=12)
    ax2.set_title('Inference Latency (RTX 4070)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05 * max(latency), 
                 f"{b.get_height():.2f}", ha='center', va='bottom', fontsize=10)
                 
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'power', 'power_and_latency_bars.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [保存] {save_path}")

def plot_sparsity_vs_power_scatter():
    """稀疏度与功耗散点图及拟合曲线"""
    file_path = os.path.join(RESULTS_DIR, 'exp2_power_and_latency.csv')
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    
    # 过滤掉 ANN 等非脉冲模型，仅针对 SNN (这里可以扩展)
    snn_df = df[df['Model'].str.contains('SNN')]
    if snn_df.empty: return
    
    sparsity = snn_df['Sparsity'].values
    energy = snn_df['Energy_per_Sample_mJ'].values
    names = snn_df['Model'].values
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 绘制散点
    for i, txt in enumerate(names):
        c = COLORS[0] if 'Sparse' in txt else COLORS[1]
        ax.scatter(sparsity[i], energy[i], color=c, s=150, edgecolor='black', label=txt)
        
    ax.set_xlabel('Global Sparsity', fontsize=12)
    ax.set_ylabel('Energy Consumption (mJ)', fontsize=12)
    ax.set_title('Sparsity vs. Energy Consumption', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # 如果有两个点，画一条虚线拟合
    if len(sparsity) >= 2:
        z = np.polyfit(sparsity, energy, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(sparsity)*0.9, max(sparsity)*1.1, 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.7, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'power', 'sparsity_vs_power.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [保存] {save_path}")

def run_all_power_plots():
    print("\n" + "="*60)
    print("生成低功耗分析图表...")
    print("="*60)
    plot_power_metrics()
    plot_sparsity_vs_power_scatter()

if __name__ == '__main__':
    run_all_power_plots()
