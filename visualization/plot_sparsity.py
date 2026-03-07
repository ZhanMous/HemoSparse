# -*- coding: utf-8 -*-

"""
稀疏性分析可视化
- 生成 4+ 张量化 SNN 稀疏性的图表
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, RESULTS_DIR, FIG_DPI, COLORS, LABELS

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

def plot_sparsity_vs_epoch():
    """图1：稀疏度随训练轮次变化曲线（假设只针对标准 SNN）"""
    file_path = os.path.join(RESULTS_DIR, 'snn_T20_history.csv')
    if not os.path.exists(file_path):
        print(f"找不到 {file_path}，跳过图1绘制")
        return
        
    df = pd.read_csv(file_path)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df['epoch'], df['train_sparsity'], label='Train Sparsity', color=COLORS[0], linestyle='-', marker='o', markersize=4)
    ax1.plot(df['epoch'], df['val_sparsity'], label='Val Sparsity', color=COLORS[0], linestyle='--', marker='s', markersize=4)
    
    ax1.set_xlabel('Training Epoch', fontsize=12)
    ax1.set_ylabel('Global Sparsity', fontsize=12, color=COLORS[0])
    ax1.tick_params(axis='y', labelcolor=COLORS[0])
    ax1.set_title('SNN: Global Sparsity over Training Epochs (T=20)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 双轴绘制准确率对比如有需要...这里专注稀疏度
    ax1.legend(loc='lower right')
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'sparsity', 'sparsity_vs_epoch.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [保存] {save_path}")

def plot_sparsity_boxplots():
    """图2：三组对照模型的全局稀疏度对比箱线图"""
    file_path = os.path.join(RESULTS_DIR, 'exp1_sparsity_quantification.csv')
    if not os.path.exists(file_path):
        print(f"找不到 {file_path}，跳过图2绘制")
        return
        
    df = pd.read_csv(file_path)
    
    # 按模型类型分组数据
    # T=20, v_threshold=0.5 的条件下比较各种模型的稀疏度
    data = []
    labels = []
    
    # 按T和v_threshold分组，获取不同条件下的稀疏度数据
    grouped = df.groupby(['T', 'v_threshold'])
    
    # 为每种模型类型收集数据
    unique_ts = sorted(df['T'].unique())
    
    for t_val in unique_ts:
        t_data = df[df['T'] == t_val]
        v_thresholds = sorted(t_data['v_threshold'].unique())
        
        for v_thresh in v_thresholds:
            subset = t_data[t_data['v_threshold'] == v_thresh]
            if len(subset) > 0:
                # 创建标签描述T和v_threshold的组合
                label = f'T={t_val}, v_th={v_thresh}'
                data.append(subset['Global_Sparsity'].values)
                labels.append(label)
                
    # 如果数据过多，只选择一个特定的T值（例如T=20）
    if len(unique_ts) > 1:
        target_t = 20 if 20 in unique_ts else max(unique_ts)
        t_data = df[df['T'] == target_t]
        v_thresholds = sorted(t_data['v_threshold'].unique())
        
        data = []
        labels = []
        for v_thresh in v_thresholds:
            subset = t_data[t_data['v_threshold'] == v_thresh]
            if len(subset) > 0:
                label = f'T={target_t}, v_th={v_thresh}'
                data.append(subset['Global_Sparsity'].values)
                labels.append(label)
    
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.5), 5))
    
    # 确保有足够的颜色
    colors_to_use = COLORS * (len(data) // len(COLORS) + 1)
    
    bplot = ax.boxplot(data, patch_artist=True, labels=labels)
    
    for i, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(colors_to_use[i % len(colors_to_use)])
        patch.set_alpha(0.7)
        
    for median in bplot['medians']:
        median.set(color='black', linewidth=1.5)
        
    ax.set_ylabel('Global Sparsity', fontsize=12)
    ax.set_title('Global Sparsity Comparison across Thresholds', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'sparsity', 'sparsity_boxplot.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [保存] {save_path}")

def plot_layer_firing_rate():
    """图3：SNN每层神经元脉冲发放率热力图"""
    file_path = os.path.join(RESULTS_DIR, 'exp1_sparsity_quantification.csv')
    if not os.path.exists(file_path):
        print(f"找不到 {file_path}，跳过图3绘制")
        return
        
    df = pd.read_csv(file_path)
    # 取 T=20， v_threshold=1.0 的那一组
    sub_df = df[(df['T'] == 20) & (df['v_threshold'] == 1.0)]
    if sub_df.empty: 
        print("未找到 T=20, V_th=1.0 的数据，跳过层级图")
        return
    
    # 动态查找含有 layer 或特征提取器层名称的列
    potential_cols = ['feature_extractor.2', 'feature_extractor.6', 'classifier.1']
    cols = [c for c in sub_df.columns if c in potential_cols or ('layer' in c.lower() and c != 'v_threshold')]
    
    if not cols:
        # 如果没有具体层数据，尝试拿 Global_Avg_Rate 充数展示一下
        rates = [sub_df['Global_Avg_Rate'].values[0]]
        layer_names = ['Global Average']
    else:
        rates = sub_df[cols].values.flatten()
        layer_names = [c.split('.')[-1] if '.' in c else c for c in cols]
        # 映射回更人类可读的名字
        name_map = {'2': 'Conv Layer 1', '6': 'Conv Layer 2', '1': 'Output Layer'}
        layer_names = [name_map.get(n, n) for n in layer_names]
        
    fig, ax = plt.subplots(figsize=(max(6, len(rates)*2), 3))
    im = ax.imshow([rates], cmap='Blues', aspect='equal', vmin=0, vmax=1.0)
    
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(layer_names, rotation=15)
    ax.set_yticks([])
    ax.set_title('Layer-wise Firing Rates (SNN, T=20)', fontsize=14, fontweight='bold')
    
    for i, rate in enumerate(rates):
        color = 'white' if rate > 0.5 else 'black'
        ax.text(i, 0, f"{rate:.4f}", ha="center", va="center", color=color, fontweight='bold')
        
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Firing Rate', rotation=270, labelpad=15)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'sparsity', 'layer_firing_rate.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [保存] {save_path}")

def run_all_sparsity_plots():
    print("\n" + "="*60)
    print("生成稀疏性分析图表...")
    print("="*60)
    plot_sparsity_vs_epoch()
    plot_layer_firing_rate()
    plot_sparsity_boxplots()

if __name__ == '__main__':
    run_all_sparsity_plots()
