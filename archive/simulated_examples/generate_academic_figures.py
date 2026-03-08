# -*- coding: utf-8 -*-
"""
生成学术图表
- 模型性能柱状图
- 稀疏度 ↔ MIA 鲁棒性折线图
- 成员 / 非成员置信度分布直方图
- 功耗 - 延迟散点图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# 保存目录
OUTPUT_DIR = 'outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置 IEEE 风格
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Times New Roman',
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (6.5, 4.5),
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.5
})

# 读取训练结果
def read_training_results():
    csv_path = os.path.join('outputs', 'training_summary.csv')
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model']
            results[model] = {
                'test_acc': float(row['test_acc'].split(' ± ')[0]),
                'training_time': float(row['training_time'].split(' ± ')[0]),
                'power': float(row['power'].split(' ± ')[0]),
                'latency': float(row['latency'].split(' ± ')[0])
            }
    return results

# 读取 MIA 结果
def read_mia_results():
    csv_path = os.path.join('outputs', 'mia_results.csv')
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            model = row[0]
            results[model] = float(row[1])
    return results

# 读取消融实验结果
def read_ablation_results():
    # 尝试从若干可能的 ablation CSV 中读取真实数据
    candidates = [
        os.path.join('outputs', 'p1_plif_ablation.csv'),
        os.path.join('outputs', 'ablation_results.csv'),
        os.path.join('outputs', 'sparsity_results.csv')
    ]
    results = []
    for csv_path in candidates:
        if not os.path.exists(csv_path):
            continue
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entry = {}
                    if 'v_threshold' in row:
                        try:
                            entry['v_threshold'] = float(row['v_threshold'])
                        except Exception:
                            entry['v_threshold'] = None
                    if 'sparsity' in row:
                        try:
                            entry['sparsity'] = float(str(row['sparsity']).split()[0])
                        except Exception:
                            entry['sparsity'] = None
                    if 'mia_acc' in row:
                        try:
                            entry['mia_acc'] = float(str(row['mia_acc']).split()[0])
                        except Exception:
                            entry['mia_acc'] = None
                    if 'test_acc' in row:
                        try:
                            entry['test_acc'] = float(str(row['test_acc']).split()[0])
                        except Exception:
                            entry['test_acc'] = None
                    # only keep rows with at least one useful metric
                    if any(v is not None for v in entry.values()):
                        results.append(entry)
        except Exception:
            continue
    return results

# 生成模型性能柱状图
def plot_model_performance():
    results = read_training_results()
    models = list(results.keys())
    test_acc = [results[m]['test_acc'] for m in models]
    
    fig, ax = plt.subplots()
    bars = ax.bar(models, test_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_performance.png'))
    print('Model performance plot saved')

# 生成稀疏度与 MIA 鲁棒性关系图
def plot_sparsity_vs_mia():
    rows = read_ablation_results()
    if not rows:
        print('警告: 未找到消融实验数据，跳过 sparsity vs MIA 绘图')
        return

    sparsity = [r.get('sparsity') for r in rows if r.get('sparsity') is not None]
    mia_acc = [r.get('mia_acc') for r in rows if r.get('mia_acc') is not None]

    if len(sparsity) == 0 or len(mia_acc) == 0:
        print('警告: 消融数据中缺少 sparsity 或 mia_acc 字段，跳过绘图')
        return

    fig, ax = plt.subplots()
    ax.plot(sparsity, mia_acc, 'o-', color='#1f77b4', label='MIA Accuracy')

    ax.set_xlabel('Sparsity')
    ax.set_ylabel('MIA Attack Accuracy')
    ax.set_title('Sparsity vs MIA Robustness')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sparsity_vs_mia.png'))
    print('Sparsity vs MIA plot saved')

# 生成置信度分布直方图
def plot_confidence_distribution():
    # 尝试从 outputs 中读取置信度分布（如果存在）
    conf_csv = os.path.join('outputs', 'mia_confidence_distribution.csv')
    if not os.path.exists(conf_csv):
        print('警告: 未找到置信度分布文件 (outputs/mia_confidence_distribution.csv)，跳过绘图')
        return

    member_conf = []
    non_member_conf = []
    try:
        with open(conf_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('label') == 'member':
                    try:
                        member_conf.append(float(row['confidence']))
                    except Exception:
                        continue
                elif row.get('label') == 'non_member':
                    try:
                        non_member_conf.append(float(row['confidence']))
                    except Exception:
                        continue
    except Exception:
        print('警告: 读取置信度分布文件失败，跳过绘图')
        return

    if len(member_conf) == 0 or len(non_member_conf) == 0:
        print('警告: 置信度分布数据不完整，跳过绘图')
        return

    fig, ax = plt.subplots()
    sns.histplot(member_conf, bins=30, alpha=0.6, label='Members', color='#1f77b4', ax=ax)
    sns.histplot(non_member_conf, bins=30, alpha=0.6, label='Non-members', color='#ff7f0e', ax=ax)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution: Members vs Non-members')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_distribution.png'))
    print('Confidence distribution plot saved')

# 生成功耗-延迟散点图
def plot_power_latency():
    results = read_training_results()
    models = list(results.keys())
    power = [results[m]['power'] for m in models]
    latency = [results[m]['latency'] for m in models]
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(latency, power, s=100, c=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])
    
    # 添加标签
    for i, model in enumerate(models):
        ax.annotate(model, (latency[i], power[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power-Latency Trade-off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'power_latency.png'))
    print('Power-latency plot saved')

# 主函数
def main():
    print("生成学术图表...")
    plot_model_performance()
    plot_sparsity_vs_mia()
    plot_confidence_distribution()
    plot_power_latency()
    print("所有图表生成完成，保存在 outputs/figures 目录")

if __name__ == '__main__':
    main()
