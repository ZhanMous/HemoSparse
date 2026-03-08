# -*- coding: utf-8 -*-
"""
隐私性能分析图
- MIA攻击成功率分析
- 置信度分布对比
- 隐私-准确率权衡
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FIGURES_DIR, RESULTS_DIR, FIG_DPI, COLORS


def plot_sparsity_vs_mia():
    """绘制稀疏性与MIA准确率关系图"""
    power_path = os.path.join(RESULTS_DIR, 'exp2_power_and_latency.csv')
    mia_path = os.path.join(RESULTS_DIR, 'exp3_privacy_mia.csv')
    
    if not (os.path.exists(power_path) and os.path.exists(mia_path)): 
        print("  [警告] 功耗或隐私实验结果文件不存在")
        return
    
    # 读取数据
    df_power = pd.read_csv(power_path)
    df_mia = pd.read_csv(mia_path)
    
    # 从功耗数据中提取每个模型的平均稀疏性
    avg_sparsity = df_power.groupby('model_type')['sparsity'].mean().reset_index()
    avg_sparsity.rename(columns={'model_type': 'Model'}, inplace=True)
    
    # 合并数据
    df = pd.merge(avg_sparsity, df_mia, on='Model')
    
    sparsity = df['sparsity'].values
    mia_acc = df['MIA_Accuracy'].values * 100
    names = df['Model'].values
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for i, txt in enumerate(names):
        c = COLORS[0] if 'Sparse' in txt else (COLORS[1] if 'Dense' in txt else COLORS[2])
        model_name = txt.replace('SNN (Sparse)', 'SNN(A)').replace('Dense_SNN', 'DenseSNN(B)').replace('ANN', 'ANN(C)')
        ax.scatter(sparsity[i], mia_acc[i], color=c, s=150, edgecolor='black', label=model_name)
        
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sparsity', fontsize=12)
    ax.set_ylabel('MIA Accuracy (%)', fontsize=12)
    ax.set_title('Privacy Protection vs Sparsity', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # 添加参考线
    ax.text(0.02, 52, 'Random Guess (50%)', color='red', fontsize=10, rotation=0)
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'academic', 'privacy_sparsity_correlation.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {path}")


def plot_mia_accuracy_comparison():
    """绘制MIA准确率对比柱状图"""
    mia_path = os.path.join(RESULTS_DIR, 'exp3_privacy_mia.csv')
    
    if not os.path.exists(mia_path):
        print("  [警告] 隐私实验结果文件不存在")
        return
    
    df = pd.read_csv(mia_path)
    
    models = df['Model'].values
    mia_accs = df['MIA_Accuracy'].values * 100
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = []
    for i, (model, acc) in enumerate(zip(models, mia_accs)):
        c = COLORS[0] if 'Sparse' in model else (COLORS[1] if 'Dense' in model else COLORS[2])
        bar = ax.bar(i, acc, color=c, edgecolor='black', linewidth=1.2)
        bars.append(bar)
        
        # 在柱子上方添加数值
        ax.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([
        name.replace('SNN (Sparse)', 'SNN(A)').replace('Dense_SNN', 'DenseSNN(B)').replace('ANN', 'ANN(C)')
        for name in models
    ])
    ax.set_ylabel('MIA Accuracy (%)', fontsize=12)
    ax.set_title('MIA Attack Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.set_ylim(0, 100)
    
    # 添加随机猜测参考线
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random Guess (50%)')
    ax.legend()
    
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'academic', 'mia_accuracy_comparison.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {path}")


def run_all_privacy_plots():
    """运行所有隐私分析图"""
    print("\n隐私性能分析图...")
    plot_sparsity_vs_mia()
    plot_mia_accuracy_comparison()


if __name__ == '__main__':
    run_all_privacy_plots()
