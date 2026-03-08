# -*- coding: utf-8 -*-
"""
稀疏度消融实验
- 遍历 v_threshold = [0.5, 0.75, 1.0, 1.5]
- 记录：sparsity / test_acc / MIA_acc
- 5 次独立重复实验
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import functional
from data.dataloader import get_blood_mnist_loaders
from models import SNN
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv

# 超参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
T = 6
V_THRESHOLDS = [0.5, 0.75, 1.0, 1.5]
NUM_REPEATS = 5

# 保存目录
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 计算熵
def compute_entropy(p):
    return -torch.sum(p * torch.log(p + 1e-10), dim=1)

# 计算稀疏度
def compute_sparsity(model, data_loader):
    device = next(model.parameters()).device
    total_spikes = 0
    total_neurons = 0
    
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            
            # 重置网络
            functional.reset_net(model)
            
            # 前向传播并收集脉冲
            spikes = []
            
            def hook_fn(module, input, output):
                if hasattr(module, 'v'):
                    spikes.append(output)
            
            # 注册钩子
            handles = []
            for module in model.modules():
                if isinstance(module, nn.Module) and hasattr(module, 'v'):
                    handles.append(module.register_forward_hook(hook_fn))
            
            # 前向传播
            model(data)
            
            # 移除钩子
            for handle in handles:
                handle.remove()
            
            # 计算脉冲数
            for spike in spikes:
                total_spikes += spike.sum().item()
                total_neurons += spike.numel()
    
    sparsity = 1 - (total_spikes / total_neurons)
    return sparsity

# 训练并评估模型（包括 MIA 攻击）
def train_and_evaluate(v_threshold, seed):
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 加载数据
    train_loader, _, test_loader, _ = get_blood_mnist_loaders(batch_size=BATCH_SIZE, mode='snn')
    
    # 初始化模型
    model = SNN(T=T, v_threshold=v_threshold)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    for epoch in range(EPOCHS):
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            functional.reset_net(model)
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # 测试准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            functional.reset_net(model)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_acc = 100. * correct / total
    
    # 计算稀疏度
    sparsity = compute_sparsity(model, test_loader)
    
    # 执行 MIA 攻击
    mia_acc = compute_mia_accuracy(model, train_loader, test_loader, device)
    
    return sparsity, test_acc, mia_acc

# 计算 MIA 准确率
def compute_mia_accuracy(model, train_loader, test_loader, device):
    # 分割训练数据为成员和非成员
    train_dataset = train_loader.dataset
    n_train = len(train_dataset)
    n_member = n_train // 2
    member_indices = np.random.choice(n_train, n_member, replace=False)
    non_member_indices = np.setdiff1d(np.arange(n_train), member_indices)
    
    member_loader = DataLoader(Subset(train_dataset, member_indices), batch_size=BATCH_SIZE, shuffle=False)
    non_member_loader = DataLoader(Subset(test_loader.dataset, np.arange(len(test_loader.dataset))), batch_size=BATCH_SIZE, shuffle=False)
    
    # 提取特征
    def extract_features(loader):
        features = []
        model.eval()
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(device)
                functional.reset_net(model)
                outputs = model(data)
                probs = nn.functional.softmax(outputs, dim=1)
                max_conf = probs.max(dim=1)[0].cpu().numpy()
                entropy = compute_entropy(probs).cpu().numpy()
                batch_features = np.column_stack((max_conf, entropy))
                features.append(batch_features)
        return np.vstack(features)
    
    member_features = extract_features(member_loader)
    non_member_features = extract_features(non_member_loader)
    
    # 合并特征
    X = np.vstack((member_features, non_member_features))
    y = np.concatenate((np.ones(len(member_features)), np.zeros(len(non_member_features))))
    
    # 训练攻击模型
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        attack_model = LogisticRegression(max_iter=1000)
        attack_model.fit(X_train, y_train)
        y_pred = attack_model.predict(X_test)
        mia_acc = accuracy_score(y_test, y_pred)
    else:
        mia_acc = 0.5
    
    return mia_acc

# 主函数
def main():
    results = {}
    
    for v_threshold in V_THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"  测试 v_threshold = {v_threshold}")
        print(f"{'='*60}")
        results[v_threshold] = {
            'sparsity': [],
            'test_acc': [],
            'mia_acc': []
        }
        
        for seed in range(NUM_REPEATS):
            print(f"\n--- 第 {seed+1} 次重复实验 ---")
            sparsity, test_acc, mia_acc = train_and_evaluate(v_threshold, seed)
            results[v_threshold]['sparsity'].append(sparsity)
            results[v_threshold]['test_acc'].append(test_acc)
            results[v_threshold]['mia_acc'].append(mia_acc)
            print(f"  稀疏度: {sparsity:.4f}, 测试准确率: {test_acc:.2f}%, MIA 准确率: {mia_acc:.4f}")
    
    # 计算均值和标准差
    summary = []
    for v_threshold in V_THRESHOLDS:
        sparsity_mean = np.mean(results[v_threshold]['sparsity'])
        sparsity_std = np.std(results[v_threshold]['sparsity'])
        test_acc_mean = np.mean(results[v_threshold]['test_acc'])
        test_acc_std = np.std(results[v_threshold]['test_acc'])
        mia_acc_mean = np.mean(results[v_threshold]['mia_acc'])
        mia_acc_std = np.std(results[v_threshold]['mia_acc'])
        
        summary.append({
            'v_threshold': v_threshold,
            'sparsity': f"{sparsity_mean:.4f} ± {sparsity_std:.4f}",
            'test_acc': f"{test_acc_mean:.2f} ± {test_acc_std:.2f}",
            'mia_acc': f"{mia_acc_mean:.4f} ± {mia_acc_std:.4f}"
        })
    
    # 保存结果
    csv_path = os.path.join(OUTPUT_DIR, 'ablation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['v_threshold', 'sparsity', 'test_acc', 'mia_acc'])
        writer.writeheader()
        writer.writerows(summary)
    
    print(f"\n{'='*60}")
    print("  消融实验完成")
    print(f"{'='*60}")
    print(f"结果已保存到 {csv_path}")
    
    # 打印结果
    print("\n稀疏度消融实验结果:")
    for item in summary:
        print(f"  v_threshold={item['v_threshold']}: 稀疏度={item['sparsity']}, 测试准确率={item['test_acc']}%, MIA 准确率={item['mia_acc']}")

if __name__ == '__main__':
    main()
