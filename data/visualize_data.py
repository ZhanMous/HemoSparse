# -*- coding: utf-8 -*-
"""
BloodMNIST 数据集可视化
- 类别分布柱状图
- 8类血液细胞样本展示（带临床标注）
- 像素分布热力图
- 脉冲编码对比图 + 脉冲光栅图
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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import PoissonEncoder  # 保留用于对比
from config import (
    DATA_ROOT, DATA_FLAG, FIGURES_DIR, FIG_DPI,
    COLORS, set_seed, SEED, DEFAULT_T
)

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


def plot_pixel_heatmap(train_ds, save_dir):
    """单样本像素分布热力图 + 各类别平均图"""
    info = INFO[DATA_FLAG]
    class_names = list(info['label'].values())
    labels = np.array([int(train_ds[i][1].squeeze()) for i in range(len(train_ds))])
    n_classes = len(class_names)

    # 简化过长的类别名称
    short_class_names = []
    for name in class_names:
        if 'granulocytes' in name and '(' in name:
            # 将过长的标签简化
            short_name = 'Granulocytes'
        elif 'immature granulocytes' in name:
            short_name = 'Imm. Gran.'
        elif 'promyelocytes' in name:
            short_name = 'Promyelo.'
        else:
            short_name = name
        short_class_names.append(short_name)

    # 确保有足够的类别数来创建子图
    if n_classes <= 1:
        print("  [跳过] 类别数不足，无法生成像素热力图")
        return

    # 如果类别过多，可以选择只显示前几个类别
    display_classes = min(n_classes, 8)  # 最多显示8个类别
    fig, axes = plt.subplots(1, display_classes, figsize=(2.2*display_classes, 3))
    fig.suptitle('Average Pixel Intensity per Class (BloodMNIST)', fontsize=13, fontweight='bold')

    for cls_idx in range(display_classes):
        indices = np.where(labels == cls_idx)[0][:100]  # 取前100个样本
        if len(indices) == 0:
            # 如果该类别没有样本，跳过
            continue
            
        imgs = []
        for idx in indices:
            img, _ = train_ds[idx]
            imgs.append(img.mean(dim=0).numpy())  # 通道平均
        avg_img = np.mean(imgs, axis=0)

        if display_classes == 1:
            # 如果只显示一个类别，axes不是一个数组
            ax = axes
        else:
            ax = axes[cls_idx]
            
        im = ax.imshow(avg_img, cmap='hot', vmin=0, vmax=1)
        ax.set_title(short_class_names[cls_idx], fontsize=8, pad=3)
        ax.axis('off')

    # 不使用子图共享的colorbar，而是使用单独的区域放置colorbar
    # 调整布局以留出空间给colorbar
    plt.subplots_adjust(top=0.8, bottom=0.15, left=0.05, right=0.85, wspace=0.3)
    
    # 创建一个单独的轴用于colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Pixel Intensity', fontsize=10)
    
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


def plot_spike_encoding_comparison(train_ds, save_dir, T=DEFAULT_T):
    """脉冲编码对比图（原始图像与直接编码后的时间序列）+ 脉冲光栅图"""
    set_seed(SEED)
    encoder = PoissonEncoder(T)  # 保留泊松编码器用于对比

    img, label = train_ds[0]  # [C, H, W]
    
    # 使用泊松编码
    poisson_spikes = encoder(img)     # [T, C, H, W]
    
    # 直接编码：复制图像T次
    direct_input = img.unsqueeze(0).repeat(T, 1, 1, 1)  # [T, C, H, W]

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 4, hspace=0.4, wspace=0.3)

    # 原始图像
    ax0 = fig.add_subplot(gs[0, 0])
    img_show = img.permute(1, 2, 0).numpy()
    if img_show.shape[2] == 1:
        ax0.imshow(img_show.squeeze(-1), cmap='gray')
    else:
        ax0.imshow(img_show)
    ax0.set_title('Original Image', fontsize=11, fontweight='bold')
    ax0.axis('off')

    # 选取4个时间步展示直接编码
    t_steps = np.linspace(0, T - 1, 4, dtype=int)
    for i, t in enumerate(t_steps):
        ax = fig.add_subplot(gs[0, i] if i > 0 else gs[0, 1])
        if i > 0:
            ax = fig.add_subplot(gs[0, i])
        direct_img = direct_input[t].mean(dim=0).numpy()  # 通道平均
        ax.imshow(direct_img, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Direct t={t}', fontsize=10)
        ax.axis('off')

    # 选取4个时间步展示泊松编码
    for i, t in enumerate(t_steps):
        ax = fig.add_subplot(gs[1, i])
        poisson_img = poisson_spikes[t].mean(dim=0).numpy()  # 通道平均
        ax.imshow(poisson_img, cmap='binary', vmin=0, vmax=1)
        ax.set_title(f'Poisson t={t}', fontsize=10)
        ax.axis('off')

    # 脉冲光栅图（Raster Plot）- 直接编码
    ax_raster_direct = fig.add_subplot(gs[2, :2])
    direct_flat = direct_input[:, 0, 14, :].numpy()  # [T, W] 取通道0, 第14行
    n_neurons = direct_flat.shape[1]
    for neuron_idx in range(n_neurons):
        firing_times = np.where(direct_flat[:, neuron_idx] > 0.5)[0]  # 直接编码值大于0.5认为是激活
        if len(firing_times) > 0:
            ax_raster_direct.scatter(firing_times, np.full_like(firing_times, neuron_idx),
                                 s=2, c='blue', marker='|', alpha=0.6)
    ax_raster_direct.set_xlabel('Time Step', fontsize=11)
    ax_raster_direct.set_ylabel('Neuron Index (Row 14 pixels)', fontsize=11)
    ax_raster_direct.set_title('Direct Encoding Raster Plot', fontsize=12, fontweight='bold')
    ax_raster_direct.set_xlim(-0.5, T - 0.5)
    ax_raster_direct.set_ylim(-0.5, n_neurons - 0.5)

    # 脉冲光栅图（Raster Plot）- 泊松编码
    ax_raster_poisson = fig.add_subplot(gs[2, 2:])
    poisson_flat = poisson_spikes[:, 0, 14, :].numpy()  # [T, W] 取通道0, 第14行
    n_neurons = poisson_flat.shape[1]
    for neuron_idx in range(n_neurons):
        firing_times = np.where(poisson_flat[:, neuron_idx] > 0)[0]  # 泊松编码有脉冲
        if len(firing_times) > 0:
            ax_raster_poisson.scatter(firing_times, np.full_like(firing_times, neuron_idx),
                                  s=2, c='red', marker='|', alpha=0.6)
    ax_raster_poisson.set_xlabel('Time Step', fontsize=11)
    ax_raster_poisson.set_ylabel('Neuron Index (Row 14 pixels)', fontsize=11)
    ax_raster_poisson.set_title('Poisson Encoding Raster Plot', fontsize=12, fontweight='bold')
    ax_raster_poisson.set_xlim(-0.5, T - 0.5)
    ax_raster_poisson.set_ylim(-0.5, n_neurons - 0.5)

    # 发放率统计对比
    ax_rate = fig.add_subplot(gs[3, :])
    direct_firing_rate = direct_input.mean(dim=(1, 2, 3)).numpy()  # [T]
    poisson_firing_rate = poisson_spikes.mean(dim=(1, 2, 3)).numpy()  # [T]
    
    ax_rate.plot(range(T), direct_firing_rate, 'b-o', label=f'Direct Encoding (Mean={direct_firing_rate.mean():.3f})', linewidth=2)
    ax_rate.plot(range(T), poisson_firing_rate, 'r-s', label=f'Poisson Encoding (Mean={poisson_firing_rate.mean():.3f})', linewidth=2)
    ax_rate.set_xlabel('Time Step', fontsize=11)
    ax_rate.set_ylabel('Mean Firing Rate', fontsize=11)
    ax_rate.set_title('Firing Rate per Time Step - Encoding Comparison', fontsize=12, fontweight='bold')
    ax_rate.legend()
    ax_rate.grid(True, linestyle=':', alpha=0.6)

    path = os.path.join(save_dir, 'spike_encoding_comparison.png')
    # 不使用tight_layout，而是直接保存
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {path}")


def plot_spike_dynamics(train_ds, save_dir, T=DEFAULT_T):
    """脉冲时空动态图 - 展示直接编码的稀疏性"""
    set_seed(SEED)

    img, label = train_ds[0]  # [C, H, W]
    
    # 直接编码：复制图像T次
    direct_input = img.unsqueeze(0).repeat(T, 1, 1, 1)  # [T, C, H, W]
    
    # 注意：实际的稀疏性是由SNN模型内部的LIF神经元产生的
    # 这里我们模拟经过SNN层后的脉冲活动
    # 在实际SNN中，神经元需要累积膜电位达到阈值才会发放脉冲
    
    # 模拟SNN处理后的脉冲活动（更稀疏）
    simulated_spikes = torch.zeros_like(direct_input)
    for t in range(T):
        # 模拟LIF神经元的脉冲发放，只有部分神经元会发放脉冲
        membrane_potential = direct_input[t] if t == 0 else membrane_potential + direct_input[t]*0.3  # 简化模拟
        # 当膜电位超过阈值时发放脉冲
        spikes = (membrane_potential > 0.8).float()
        # 重置膜电位
        membrane_potential = membrane_potential * (1 - spikes) + spikes * 0.2  # 模拟不应期
        simulated_spikes[t] = spikes

    # 选择一个更合适的区域，展示稀疏性
    h_slice = slice(0, 28)  # 使用整个图像（28x28）
    w_slice = slice(0, 28)

    fig, axes = plt.subplots(4, T, figsize=(16, 8))
    fig.suptitle('Spatio-Temporal Spike Dynamics - Visualizing Sparsity in SNNs', fontsize=14, fontweight='bold')

    # 第一行：原始图像
    for t in range(T):
        ax = axes[0, t]
        orig_img = img[0].cpu().numpy()  # 取第一个通道
        ax.imshow(orig_img[h_slice, w_slice], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Original Image t={t}' if t == T//2 else '', fontsize=10)
        ax.axis('off')

    # 第二行：直接编码（复制T次）
    for t in range(T):
        ax = axes[1, t]
        input_slice = direct_input[t, 0, h_slice, w_slice].cpu().numpy()  # 取第一个通道
        im = ax.imshow(input_slice, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Direct Encoding t={t}' if t == T//2 else '', fontsize=10)
        ax.axis('off')

    # 第三行：模拟SNN处理后的脉冲（稀疏）
    for t in range(T):
        ax = axes[2, t]
        spike_slice = simulated_spikes[t, 0, h_slice, w_slice].cpu().numpy()  # 取第一个通道
        im = ax.imshow(spike_slice, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f'Simulated SNN Spikes t={t}' if t == T//2 else '', fontsize=10)
        ax.axis('off')

    # 第四行：脉冲活动的稀疏性统计
    for t in range(T):
        ax = axes[3, t]
        spike_slice = simulated_spikes[t, 0, h_slice, w_slice].cpu().numpy()
        
        # 计算并显示稀疏性
        total_neurons = spike_slice.size
        active_neurons = (spike_slice > 0.5).sum()  # 计算激活的神经元
        sparsity = 1 - (active_neurons / total_neurons) if total_neurons > 0 else 0
        
        # 显示激活神经元数量和稀疏性
        ax.text(0.5, 0.6, f'Active:\n{int(active_neurons)}/{total_neurons}', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=10, fontweight='bold')
        ax.text(0.5, 0.3, f'Sparsity:\n{sparsity:.2f}', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=10, fontweight='bold')
        
        ax.set_title(f'Stats t={t}' if t == T//2 else '', fontsize=10)
        ax.axis('off')

    # 添加颜色条
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.015, 0.25])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap='Blues'), cbar_ax1, orientation='vertical')
    cbar1.set_label('Direct Encoding Intensity', rotation=270, labelpad=15)

    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.25])  # [left, bottom, width, height]
    cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap='Reds'), cbar_ax2, orientation='vertical')
    cbar2.set_label('Spike Activity', rotation=270, labelpad=15)

    # 添加说明文本
    fig.text(0.5, 0.02, 'SNN Sparsity: Neurons only fire when membrane potential reaches threshold,\nresulting in sparse temporal activation patterns.', 
             ha='center', fontsize=10, style='italic')

    # 不使用tight_layout，而是手动调整间距
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.9, hspace=0.5, wspace=0.3)
    path = os.path.join(save_dir, 'spike_dynamics.png')
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {path}")


def create_dynamic_spike_gif(train_ds, save_dir, T=DEFAULT_T):
    """创建动态脉冲gif图，展示时间维度上的稀疏性"""
    import matplotlib.animation as animation
    
    set_seed(SEED)
    
    img, label = train_ds[0]  # [C, H, W]
    
    # 直接编码：复制图像T次
    direct_input = img.unsqueeze(0).repeat(T, 1, 1, 1)  # [T, C, H, W]
    
    # 模拟SNN处理后的脉冲活动（更稀疏）
    simulated_spikes = torch.zeros_like(direct_input)
    for t in range(T):
        # 模拟LIF神经元的脉冲发放
        membrane_potential = direct_input[t] if t == 0 else membrane_potential + direct_input[t]*0.3
        # 当膜电位超过阈值时发放脉冲
        spikes = (membrane_potential > 0.8).float()
        # 重置膜电位
        membrane_potential = membrane_potential * (1 - spikes) + spikes * 0.2
        simulated_spikes[t] = spikes

    # 选择一个较小的区域来可视化（避免图像太大）
    h_slice = slice(8, 20)  # 选择一个12x12的小块
    w_slice = slice(8, 20)
    
    # 提取感兴趣区域
    roi_spikes = simulated_spikes[:, 0, h_slice, w_slice].cpu().numpy()  # [T, H, W]

    # 创建动态图
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def animate(frame):
        ax.clear()
        im = ax.imshow(roi_spikes[frame], cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f'Simulated SNN Spikes - Time Step {frame+1}/{T}', fontsize=12)
        ax.axis('off')
        return [im]
    
    # 创建动画
    ani = animation.FuncAnimation(fig, animate, frames=T, interval=500, blit=True, repeat=True)
    
    # 保存为gif
    gif_path = os.path.join(save_dir, 'dynamic_spikes.gif')
    ani.save(gif_path, writer='pillow', fps=2)
    plt.close(fig)
    print(f"  [保存] {gif_path}")
    
    # 同时保存为HTML，可以在浏览器中播放
    html_path = os.path.join(save_dir, 'dynamic_spikes.html')
    ani.save(html_path, writer='html', fps=2)
    print(f"  [保存] {html_path}")


def compare_ann_snn_dynamics(train_ds, save_dir, T=DEFAULT_T):
    """对比原始图像、ANN和SNN的动态特性"""
    set_seed(SEED)

    img, label = train_ds[0]  # [C, H, W]
    
    # ANN模式：直接使用原始图像，每个时间步都完全激活
    ann_activation = img.unsqueeze(0).repeat(T, 1, 1, 1)  # [T, C, H, W]，每次完全激活
    
    # SNN模式：模拟LIF神经元的脉冲发放（稀疏激活）
    snn_spikes = torch.zeros_like(ann_activation)
    membrane_potential = torch.zeros_like(img)  # 初始化膜电位为0
    
    for t in range(T):
        # 累积输入到膜电位
        membrane_potential = membrane_potential + ann_activation[t] * 0.3
        # 当膜电位超过阈值时发放脉冲
        spikes = (membrane_potential > 0.8).float()
        # 重置超过阈值的部分
        membrane_potential = membrane_potential * (1 - spikes) + spikes * 0.2  # 模拟不应期
        snn_spikes[t] = spikes

    # 选择一个较小的区域来可视化（避免图像太大）
    h_slice = slice(8, 20)  # 选择一个12x12的小块
    w_slice = slice(8, 20)
    
    # 提取感兴趣区域
    orig_roi = img[0, h_slice, w_slice].cpu().numpy()  # [H, W] 原始图像
    ann_roi = ann_activation[:, 0, h_slice, w_slice].cpu().numpy()  # [T, H, W]
    snn_roi = snn_spikes[:, 0, h_slice, w_slice].cpu().numpy()      # [T, H, W]

    # 创建对比图 - 3行T列
    fig, axes = plt.subplots(3, T, figsize=(16, 8))
    fig.suptitle('Input vs ANN vs SNN Activation Patterns - Demonstrating Temporal Sparsity', fontsize=14, fontweight='bold')

    # 原始图像行：显示原始输入
    for t in range(T):
        ax = axes[0, t]
        im = ax.imshow(orig_roi, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Input Image t={t}', fontsize=10)
        ax.axis('off')

    # ANN行：每个时间步都有激活（密集）
    for t in range(T):
        ax = axes[1, t]
        im = ax.imshow(ann_roi[t], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'ANN Activations t={t}', fontsize=10)
        ax.axis('off')

    # SNN行：仅部分时间步有激活（稀疏）
    for t in range(T):
        ax = axes[2, t]
        im = ax.imshow(snn_roi[t], cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f'SNN Spikes t={t}', fontsize=10)
        ax.axis('off')

    # 添加颜色条
    cbar_ax1 = fig.add_axes([0.92, 0.65, 0.015, 0.2])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap='gray'), cbar_ax1, orientation='vertical')
    cbar1.set_label('Input Intensity', rotation=270, labelpad=15)

    cbar_ax2 = fig.add_axes([0.92, 0.35, 0.015, 0.2])  # [left, bottom, width, height]
    cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap='Blues'), cbar_ax2, orientation='vertical')
    cbar2.set_label('ANN Activation Level', rotation=270, labelpad=15)

    cbar_ax3 = fig.add_axes([0.92, 0.05, 0.015, 0.2])  # [left, bottom, width, height]
    cbar3 = fig.colorbar(plt.cm.ScalarMappable(cmap='Reds'), cbar_ax3, orientation='vertical')
    cbar3.set_label('SNN Spike Activity', rotation=270, labelpad=15)

    # 添加说明文本
    fig.text(0.5, 0.02, 
             'Input: Original grayscale image | ANN: Dense activation at every time step | SNN: Sparse spikes when threshold reached', 
             ha='center', fontsize=11, style='italic')

    # 不使用tight_layout，而是手动调整
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.9, hspace=0.5, wspace=0.3)
    path = os.path.join(save_dir, 'input_ann_snn_comparison.png')
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存] {path}")
    
    # 统计稀疏性
    ann_total_activations = ann_roi.size
    ann_active_count = (ann_roi > 0.01).sum()  # 对于ANN，几乎所有值都是>0.01的
    ann_sparsity = 1 - (ann_active_count / ann_total_activations)
    
    snn_total_activations = snn_roi.size
    snn_active_count = (snn_roi > 0.5).sum()  # 对于SNN，只有>0.5的是脉冲
    snn_sparsity = 1 - (snn_active_count / snn_total_activations)
    
    print(f"  ANN Sparsity: {ann_sparsity:.4f} (Activations: {ann_active_count}/{ann_total_activations})")
    print(f"  SNN Sparsity: {snn_sparsity:.4f} (Spikes: {snn_active_count}/{snn_total_activations})")
    print(f"  Note: Input images are grayscale (single channel) as in BloodMNIST dataset")


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
    plot_pixel_heatmap(train_ds, save_dir)
    plot_spike_encoding_comparison(train_ds, save_dir)
    plot_spike_dynamics(train_ds, save_dir)
    create_dynamic_spike_gif(train_ds, save_dir)
    compare_ann_snn_dynamics(train_ds, save_dir)

    print(f"\n所有图表保存至: {save_dir}")


if __name__ == '__main__':
    run_all_data_visualization()
