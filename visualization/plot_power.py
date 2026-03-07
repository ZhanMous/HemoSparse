"""
学术图表：能耗与延迟分析图
- 关联分析：验证稀疏性与功耗的负相关性
- 性能对比：SNN vs DenseSNN vs ANN 推理性能
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIG_SIZE, FIG_DPI, COLORS, MARKERS, RESULTS_DIR, FIGURES_DIR


def plot_power_and_latency_analysis():
    """绘制功耗与延迟分析图"""
    print("\n生成功耗与延迟分析图...")
    
    # 加载实验结果
    result_file = os.path.join(RESULTS_DIR, 'exp2_power_and_latency.csv')
    if not os.path.exists(result_file):
        print(f"  [警告] 结果文件不存在: {result_file}")
        return
    
    df = pd.read_csv(result_file)
    
    # 过滤有效数据
    df_valid = df[(df['energy_mj'].notna()) & (df['latency_ms'].notna())]
    
    if df_valid.empty:
        print("  [警告] 没有有效的功耗/延迟数据")
        return
    
    # 计算各模型的平均值
    grouped = df_valid.groupby('model_type').agg({
        'energy_mj': ['mean', 'std'],
        'latency_ms': ['mean', 'std']
    }).round(4)
    
    # 重塑索引以方便访问
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
    
    # 获取模型类型列表
    model_types = grouped.index.tolist()
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 能耗柱状图
    energy_means = [grouped.loc[m, 'energy_mj_mean'] for m in model_types]
    energy_stds = [grouped.loc[m, 'energy_mj_std'] for m in model_types]
    
    bars1 = ax1.bar(model_types, energy_means, yerr=energy_stds, capsize=5,
                    color=COLORS[:len(model_types)], alpha=0.7, edgecolor='black')
    ax1.set_title('Inference Energy Consumption', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Energy (mJ)', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱子顶部添加数值标签
    for bar, mean_val in zip(bars1, energy_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (max(energy_means)*0.01),
                 f'{mean_val:.2f}',
                 ha='center', va='bottom', fontweight='bold')
    
    # 延迟柱状图
    latency_means = [grouped.loc[m, 'latency_ms_mean'] for m in model_types]
    latency_stds = [grouped.loc[m, 'latency_ms_std'] for m in model_types]
    
    bars2 = ax2.bar(model_types, latency_means, yerr=latency_stds, capsize=5,
                    color=COLORS[:len(model_types)], alpha=0.7, edgecolor='black')
    ax2.set_title('Inference Latency', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Latency (ms)', fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱子顶部添加数值标签
    for bar, mean_val in zip(bars2, latency_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (max(latency_means)*0.01),
                 f'{mean_val:.2f}',
                 ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(FIGURES_DIR, 'academic', 'power_and_latency_analysis.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  [保存] {save_path}")
    
    # 打印摘要统计
    print("\n功耗与延迟摘要统计:")
    for model_type in model_types:
        energy_mean = grouped.loc[model_type, 'energy_mj_mean']
        latency_mean = grouped.loc[model_type, 'latency_ms_mean']
        print(f"  {model_type}: 能耗={energy_mean:.2f}mJ, 延迟={latency_mean:.2f}ms")


if __name__ == '__main__':
    plot_power_and_latency_analysis()