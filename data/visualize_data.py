# -*- coding: utf-8 -*-
"""
BloodMNIST 数据集可视化
- 类别分布柱状图
- 8类血液细胞样本展示（带临床标注）
- 像素分布热力图
- 泊松编码前后对比图 + 脉冲光栅图
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import medmnist
from medmnist import INFO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_ROOT, DATA_FLAG, FIGURES_DIR, FIG_DPI,
    COLORS, set_seed, SEED, DEFAULT_T
)
from data.dataloader import PoissonEncoder

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11


def load_raw_dataset():
    """加载原始BloodMNIST数据集（不做变换）"""
    import torchvision.transforms as transforms
    info = INFO[DATA_FLAG]
    DataClass = getattr(medmnist, info['python_class'])
    transform = transforms.Compose([transforms.ToTensor()])
    os.makedirs(DATA_ROOT, exist_ok=True)
    train_ds = DataClass(split='train', transform=transform, download=True, root=DATA_ROOT, size=28)
    val_ds = DataClass(split='val', transform=transform, download=True, root=DATA_ROOT, size=28)
    test_ds = DataClass(split='test', transform=transform, download=True, root=DATA_ROOT, size=28)
    return train_ds, val_ds, test_ds, info


def plot_class_distribution(train_ds, info, save_dir):
    """绘制类别分布柱状图"""
    labels = np.array([int(train_ds[i][1].squeeze()) for i in range(len(train_ds))])
    class_names = list(info['label'].values())
    counts = [np.sum(labels == i) for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("husl", len(class_names))
    bars = ax.bar(range(len(class_names)), counts, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Blood Cell Type', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('BloodMNIST Class Distribution (Training Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha='center', va='bottom', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(save_dir, 'class_distribution.png')
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {path}")


def plot_sample_gallery(train_ds, info, save_dir):
    """展示8类血液细胞样本（每类2张）"""
    class_names = list(info['label'].values())
    n_classes = len(class_names)
    labels = np.array([int(train_ds[i][1].squeeze()) for i in range(len(train_ds))])

    fig, axes = plt.subplots(2, n_classes, figsize=(16, 5))
    fig.suptitle('BloodMNIST Sample Gallery (8 Blood Cell Types)', fontsize=14, fontweight='bold')

    for cls_idx in range(n_classes):
        indices = np.where(labels == cls_idx)[0]
        for row in range(2):
            idx = indices[row]
            img, _ = train_ds[idx]
            img_np = img.permute(1, 2, 0).numpy()
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(-1)
                axes[row, cls_idx].imshow(img_np, cmap='gray')
            else:
                axes[row, cls_idx].imshow(img_np)
            axes[row, cls_idx].axis('off')
            if row == 0:
                axes[row, cls_idx].set_title(class_names[cls_idx], fontsize=8, pad=3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'sample_gallery.png')
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {path}")


def plot_pixel_heatmap(train_ds, info, save_dir):
    """单样本像素分布热力图 + 各类别平均图"""
    class_names = list(info['label'].values())
    labels = np.array([int(train_ds[i][1].squeeze()) for i in range(len(train_ds))])
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, n_classes, figsize=(16, 2.5))
    fig.suptitle('Average Pixel Intensity per Class (BloodMNIST)', fontsize=13, fontweight='bold')

    for cls_idx in range(n_classes):
        indices = np.where(labels == cls_idx)[0][:100]
        imgs = []
        for idx in indices:
            img, _ = train_ds[idx]
            imgs.append(img.mean(dim=0).numpy())  # 通道平均
        avg_img = np.mean(imgs, axis=0)

        im = axes[cls_idx].imshow(avg_img, cmap='hot', vmin=0, vmax=1)
        axes[cls_idx].set_title(class_names[cls_idx], fontsize=8, pad=3)
        axes[cls_idx].axis('off')

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Pixel Intensity', fontsize=10)
    plt.tight_layout()

    path = os.path.join(save_dir, 'pixel_heatmap.png')
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {path}")


def plot_poisson_encoding(train_ds, save_dir, T=DEFAULT_T):
    """泊松编码前后对比图 + 脉冲光栅图"""
    set_seed(SEED)
    encoder = PoissonEncoder(T)

    img, label = train_ds[0]  # [C, H, W]
    spikes = encoder(img)     # [T, C, H, W]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.3)

    # 原始图像
    ax0 = fig.add_subplot(gs[0, 0])
    img_show = img.permute(1, 2, 0).numpy()
    if img_show.shape[2] == 1:
        ax0.imshow(img_show.squeeze(-1), cmap='gray')
    else:
        ax0.imshow(img_show)
    ax0.set_title('Original Image', fontsize=11, fontweight='bold')
    ax0.axis('off')

    # 选取4个时间步展示脉冲
    t_steps = np.linspace(0, T - 1, 4, dtype=int)
    for i, t in enumerate(t_steps):
        ax = fig.add_subplot(gs[0, i] if i > 0 else gs[0, 1])
        if i > 0:
            ax = fig.add_subplot(gs[0, i])
        spike_img = spikes[t].mean(dim=0).numpy()  # 通道平均
        ax.imshow(spike_img, cmap='binary', vmin=0, vmax=1)
        ax.set_title(f'Spike t={t}', fontsize=10)
        ax.axis('off')

    # 脉冲光栅图（Raster Plot）- 选取中心区域像素
    ax_raster = fig.add_subplot(gs[1, :])
    spike_flat = spikes[:, 0, 14, :].numpy()  # [T, W] 取通道0, 第14行
    n_neurons = spike_flat.shape[1]
    for neuron_idx in range(n_neurons):
        firing_times = np.where(spike_flat[:, neuron_idx] > 0)[0]
        ax_raster.scatter(firing_times, np.full_like(firing_times, neuron_idx),
                         s=2, c='black', marker='|')
    ax_raster.set_xlabel('Time Step', fontsize=11)
    ax_raster.set_ylabel('Neuron Index (Row 14 pixels)', fontsize=11)
    ax_raster.set_title(f'Spike Raster Plot (T={T})', fontsize=12, fontweight='bold')
    ax_raster.set_xlim(-0.5, T - 0.5)
    ax_raster.set_ylim(-0.5, n_neurons - 0.5)

    # 发放率统计
    ax_rate = fig.add_subplot(gs[2, :2])
    firing_rate_per_t = spikes.mean(dim=(1, 2, 3)).numpy()  # [T]
    ax_rate.bar(range(T), firing_rate_per_t, color='steelblue', alpha=0.8)
    ax_rate.set_xlabel('Time Step', fontsize=11)
    ax_rate.set_ylabel('Mean Firing Rate', fontsize=11)
    ax_rate.set_title('Firing Rate per Time Step', fontsize=12, fontweight='bold')
    ax_rate.axhline(y=firing_rate_per_t.mean(), color='red', linestyle='--',
                    label=f'Mean={firing_rate_per_t.mean():.3f}')
    ax_rate.legend()

    # 空间发放率热力图
    ax_spatial = fig.add_subplot(gs[2, 2:])
    spatial_rate = spikes.mean(dim=(0, 1)).numpy()  # [H, W]
    im = ax_spatial.imshow(spatial_rate, cmap='YlOrRd', vmin=0)
    ax_spatial.set_title('Spatial Firing Rate', fontsize=12, fontweight='bold')
    ax_spatial.axis('off')
    plt.colorbar(im, ax=ax_spatial, shrink=0.8)

    path = os.path.join(save_dir, 'poisson_encoding.png')
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {path}")


def run_all_data_visualization():
    """运行所有数据可视化"""
    print("\n" + "=" * 60)
    print("BloodMNIST 数据集可视化")
    print("=" * 60)

    save_dir = os.path.join(FIGURES_DIR, 'data')
    os.makedirs(save_dir, exist_ok=True)

    train_ds, val_ds, test_ds, info = load_raw_dataset()
    print(f"\n数据集信息:")
    print(f"  训练集: {len(train_ds)} 样本")
    print(f"  验证集: {len(val_ds)} 样本")
    print(f"  测试集: {len(test_ds)} 样本")
    print(f"  类别: {list(info['label'].values())}")
    print(f"  通道数: {info['n_channels']}")

    print("\n生成可视化图表...")
    plot_class_distribution(train_ds, info, save_dir)
    plot_sample_gallery(train_ds, info, save_dir)
    plot_pixel_heatmap(train_ds, info, save_dir)
    plot_poisson_encoding(train_ds, save_dir)

    print(f"\n所有图表保存至: {save_dir}")


if __name__ == '__main__':
    run_all_data_visualization()
