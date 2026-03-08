# -*- coding: utf-8 -*-
"""
训练脚本
- 5次独立重复实验
- 支持 SNN / DenseSNN / ANN
- 记录 test_acc / MIA_acc / 训练时间 / 功耗 / 延迟
- 性能优化：混合精度训练、梯度累积、多进程数据加载
"""

import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from spikingjelly.activation_based import functional
from data.dataloader import get_blood_mnist_loaders
from models import SNN, DenseSNN, ANN
import numpy as np
import csv
import warnings
warnings.filterwarnings('ignore')

try:
    import pynvml
except Exception:
    pynvml = None

# 超参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
T = 6
V_THRESHOLD = 1.0
ACCUMULATION_STEPS = 1  # 梯度累积步数
GRADIENT_CLIP = 1.0     # 梯度裁剪

# 保存目录
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed, deterministic=False):
    """设置随机种子；若 deterministic=True 则启用严格可复现设置"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 默认：允许小幅波动以换取性能
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reset_model_state(model_name, model):
    if model_name == 'SNN':
        functional.reset_net(model)
    elif model_name == 'DenseSNN' and hasattr(model, 'reset'):
        model.reset()


def measure_efficiency(model, model_name, data_loader, device, max_batches=10):
    """测量真实推理延迟，并在可用时采样 GPU 功耗。"""
    latencies_ms = []
    power_samples_w = []
    model.eval()

    nvml_handle = None
    if device.type == 'cuda' and pynvml is not None:
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
        except Exception:
            nvml_handle = None

    try:
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break

                data = data.to(device, non_blocking=True)
                reset_model_state(model_name, model)

                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start_time = time.perf_counter()
                _ = model(data)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                latencies_ms.append(elapsed_ms / data.size(0))

                if nvml_handle is not None:
                    try:
                        power_samples_w.append(pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0)
                    except Exception:
                        pass
    finally:
        if nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else None
    avg_power_w = float(np.mean(power_samples_w)) if power_samples_w else None
    return avg_power_w, avg_latency_ms


def format_summary_metric(mean_value, std_value, unit=''):
    if mean_value is None or std_value is None:
        return 'N/A'
    suffix = unit if unit else ''
    return f"{mean_value:.2f} ± {std_value:.2f}{suffix}"

# 训练一个模型
def train_model(model_name, seed, deterministic=False):
    # 设置随机种子
    set_seed(seed, deterministic=deterministic)
    
    # 加载数据
    train_loader, _, test_loader, _ = get_blood_mnist_loaders(
        batch_size=BATCH_SIZE, 
        mode='snn' if model_name in ['SNN', 'DenseSNN'] else 'ann',
        seed=seed,
    )
    
    # 初始化模型
    if model_name == 'SNN':
        model = SNN(T=T, v_threshold=V_THRESHOLD)
    elif model_name == 'DenseSNN':
        model = DenseSNN(T=T, v_threshold=V_THRESHOLD)
    elif model_name == 'ANN':
        model = ANN()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 计算参数量
    params = count_parameters(model)
    print(f"{model_name} 参数数量: {params} = {params/1e6:.3f}M")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[HemoSparse] 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 训练记录
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # 混合精度前向传播
            if torch.cuda.is_available() and scaler is not None:
                with autocast():
                    reset_model_state(model_name, model)
                    outputs = model(data)
                    
                    loss = criterion(outputs, targets) / ACCUMULATION_STEPS
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    # 梯度裁剪
                    if GRADIENT_CLIP > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # CPU 模式
                reset_model_state(model_name, model)
                outputs = model(data)
                
                loss = criterion(outputs, targets) / ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    if GRADIENT_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # 更新学习率
        scheduler.step()
        
        # 测试
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                reset_model_state(model_name, model)
                outputs = model(data)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total
        
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存模型
            model_path = os.path.join(OUTPUT_DIR, f'{model_name}_T{T}_seed{seed}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, model_path)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"{model_name} Seed {seed} Epoch {epoch+1}/{EPOCHS}: "
              f"Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%, "
              f"Best Acc {best_acc:.2f}%, LR {lr:.6f}")
    
    training_time = time.time() - start_time

    power, latency = measure_efficiency(model, model_name, test_loader, device)
    
    return best_acc, training_time, power, latency, params

# 主函数
def main(start_from=None, deterministic=False):
    all_models = ['SNN', 'DenseSNN', 'ANN']
    
    # 如果指定了 start_from，从指定模型开始
    if start_from and start_from in all_models:
        start_idx = all_models.index(start_from)
        models = all_models[start_idx:]
        print(f"\n=== 从 {start_from} 开始训练 ===")
    else:
        models = all_models
    
    repeats = 5
    # 5轮实验用不同种子，保证可复现但有小幅波动
    seeds = [42, 43, 44, 45, 46]
    
    # 结果存储
    results = {}
    for model in models:
        results[model] = {
            'test_acc': [],
            'training_time': [],
            'power': [],
            'latency': [],
            'params': None
        }
    
    # 运行实验
    for model in models:
        print(f"\n{'='*60}")
        print(f"  训练 {model}")
        print(f"{'='*60}")
        for i, seed in enumerate(seeds):
            print(f"\n--- 第 {i+1} 次重复实验 (seed={seed}) ---")
            # 每次实验前重新设置种子
            set_seed(seed, deterministic=deterministic)
            acc, train_time, power, latency, params = train_model(model, seed, deterministic=deterministic)
            results[model]['test_acc'].append(acc)
            results[model]['training_time'].append(train_time)
            results[model]['power'].append(power)
            results[model]['latency'].append(latency)
            if results[model]['params'] is None:
                results[model]['params'] = params
    
    # 计算均值和标准差
    summary = []
    for model in models:
        acc_mean = np.mean(results[model]['test_acc'])
        acc_std = np.std(results[model]['test_acc'])
        time_mean = np.mean(results[model]['training_time'])
        time_std = np.std(results[model]['training_time'])
        valid_power = [value for value in results[model]['power'] if value is not None]
        valid_latency = [value for value in results[model]['latency'] if value is not None]
        power_mean = float(np.mean(valid_power)) if valid_power else None
        power_std = float(np.std(valid_power)) if valid_power else None
        latency_mean = float(np.mean(valid_latency)) if valid_latency else None
        latency_std = float(np.std(valid_latency)) if valid_latency else None
        
        summary.append({
            'model': model,
            'test_acc': format_summary_metric(acc_mean, acc_std),
            'training_time': format_summary_metric(time_mean, time_std, 's'),
            'power': format_summary_metric(power_mean, power_std, 'W'),
            'latency': format_summary_metric(latency_mean, latency_std, 'ms/sample'),
            'params': f"{results[model]['params']} ({results[model]['params']/1e6:.3f}M)"
        })
    
    # 保存结果
    csv_path = os.path.join(OUTPUT_DIR, 'training_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'test_acc', 'training_time', 'power', 'latency', 'params'])
        writer.writeheader()
        writer.writerows(summary)
    
    print(f"\n{'='*60}")
    print("  训练完成")
    print(f"{'='*60}")
    print(f"结果已保存到 {csv_path}")
    
    # 打印结果
    for item in summary:
        print(f"\n{item['model']}:")
        print(f"  测试准确率: {item['test_acc']}%")
        print(f"  训练时间: {item['training_time']}s")
        print(f"  功耗: {item['power']}W")
        print(f"  延迟: {item['latency']}ms")
        print(f"  参数量: {item['params']}")

if __name__ == '__main__':
    import sys
    # 支持命令行参数指定从哪个模型开始训练
    # 用法: python train.py [SNN|DenseSNN|ANN]
    parser = argparse.ArgumentParser()
    parser.add_argument('start_from', nargs='?', default=None, choices=['SNN', 'DenseSNN', 'ANN'], help='从哪个模型开始训练')
    parser.add_argument('--deterministic', action='store_true', help='启用严格可复现的 cudnn 设置')
    args = parser.parse_args()
    main(args.start_from, deterministic=args.deterministic)
