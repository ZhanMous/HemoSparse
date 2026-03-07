# -*- coding: utf-8 -*-
"""
隐私保护分析可视化
- MIA 攻击准确率对比柱状图
- 稀疏度与 MIA 攻击准确率的相关性曲线
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

def plot_mia_accuracy_bars():
    """三组对照模型 MIA 攻击准确率对比柱状图"""
    file_path = os.path.join(RESULTS_DIR, 'exp3_privacy_mia.csv')
    if not os.path.exists(file_path): return
    
    df = pd.read_csv(file_path)
    models = df['Model'].tolist()
    models = [m.replace('SNN (Sparse)', 'SNN(A)').replace('Dense_SNN', 'DenseSNN(B)').replace('ANN', 'ANN(C)') for m in models]
    
    mia_acc = df['MIA_Accuracy'].values * 100 # 转为百分比
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, mia_acc, color=COLORS, width=0.5, edgecolor='black', alpha=0.85)
    
    # 50% 是随机抽取的基线（最安全水平）
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random Guess Baseline (50%)')
    
    ax.set_ylim(0, max(100, max(mia_acc) + 10))
    ax.set_ylabel('MIA Attack Accuracy (%)', fontsize=12)
    ax.set_title('Membership Inference Attack Vulnerability', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, 
                 f"{b.get_height():.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'privacy', 'mia_accuracy_bars.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [保存] {save_path}")

def plot_sparsity_vs_mia():
    """稀疏度与 MIA 攻击准确率的关系图"""
    power_path = os.path.join(RESULTS_DIR, 'exp2_power_and_latency.csv')
    mia_path = os.path.join(RESULTS_DIR, 'exp3_privacy_mia.csv')
    
    if not (os.path.exists(power_path) and os.path.exists(mia_path)): return
    
    df_power = pd.read_csv(power_path)
    df_mia = pd.read_csv(mia_path)
    
    # 根据模型名合并数据
    df = pd.merge(df_power, df_mia, on='Model')
    
    sparsity = df['Sparsity'].values
    mia_acc = df['MIA_Accuracy'].values * 100
    names = df['Model'].values
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for i, txt in enumerate(names):
        c = COLORS[0] if 'Sparse' in txt else (COLORS[1] if 'Dense' in txt else COLORS[2])
        model_name = txt.replace('SNN (Sparse)', 'SNN(A)').replace('Dense_SNN', 'DenseSNN(B)').replace('ANN', 'ANN(C)')
        ax.scatter(sparsity[i], mia_acc[i], color=c, s=150, edgecolor='black', label=model_name)
        
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Global Sparsity', fontsize=12)
    ax.set_ylabel('MIA Attack Accuracy (%)', fontsize=12)
    ax.set_title('Impact of Sparsity on Privacy (MIA)', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    if len(sparsity) >= 2:
        z = np.polyfit(sparsity, mia_acc, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(sparsity)-0.1, max(sparsity)+0.1, 100)
        ax.plot(x_trend, p(x_trend), "k--", alpha=0.5, label="Trend Line")
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'privacy', 'sparsity_vs_mia.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [保存] {save_path}")

def run_all_privacy_plots():
    print("\n" + "="*60)
    print("生成隐私保护分析图表...")
    print("="*60)
    plot_mia_accuracy_bars()
    plot_sparsity_vs_mia()

if __name__ == '__main__':
    run_all_privacy_plots()
