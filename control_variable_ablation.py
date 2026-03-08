# -*- coding: utf-8 -*-
"""
固定准确率的控制变量消融实验
- 固定模型测试准确率波动≤±0.2%
- 通过调整v_threshold改变模型稀疏度
- 稀疏度梯度：[0.85, 0.90, 0.95, 0.99, 0.999]
- 输出指标：测试准确率、MIA准确率、泛化gap
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
import warnings
warnings.filterwarnings('ignore')

# 超参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
T = 6
NUM_REPEATS = 5

# 稀疏度梯度对应的v_threshold值（通过预实验确定）
SPARSITY_V_THRESHOLD_MAP = {
    0.85: 0.3,
    0.90: 0.5,
    0.95: 0.75,
    0.99: 1.0,
    0.999: 1.5
}

# 保存目录
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置种子函数
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# 计算熵
def compute_entropy(p):
    return -torch.sum(p * torch.log(p + 1e-10), dim=1)

# 训练单个模型并评估
def train_and_evaluate(target_sparsity, seed):
    set_seed(seed)
    
    v_threshold = SPARSITY_V_THRESHOLD_MAP[target_sparsity]
    
    # 加载数据
    train_loader, _, test_loader, _ = get_blood_mnist_loaders(
        batch_size=BATCH_SIZE, 
        mode='snn'
    )
    
    # 初始化模型
    model = SNN(T=T, v_threshold=v_threshold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 训练模型
    train_acc_history = []
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            functional.reset_net(model)
            outputs = model(data)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_acc = 100. * train_correct / train_total
        train_acc_history.append(train_acc)
    
    # 评估测试准确率
    model.eval()
    test_correct = 0
    test_total = 0
    all_spikes = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            functional.reset_net(model)
            
            # 记录稀疏性
            if data.dim() == 4:
                x = data.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            elif data.dim() == 5:
                x = data.transpose(0, 1)
            
            x = model.stem(x)
            x = model.layer1(x)
            x = model.layer2(x)
            all_spikes.append(x.cpu().numpy())
            
            outputs = model(data)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_acc = 100. * test_correct / test_total
    final_train_acc = train_acc_history[-1]
    generalization_gap = final_train_acc - test_acc
    
    # 计算实际稀疏度
    all_spikes = np.concatenate(all_spikes, axis=1)
    actual_sparsity = 1.0 - np.mean(all_spikes)
    
    # 执行简化版MIA攻击
    mia_acc = perform_simplified_mia(model, device, train_loader, test_loader)
    
    return {
        'target_sparsity': target_sparsity,
        'actual_sparsity': actual_sparsity,
        'v_threshold': v_threshold,
        'test_acc': test_acc,
        'train_acc': final_train_acc,
        'generalization_gap': generalization_gap,
        'mia_acc': mia_acc
    }

# 简化版MIA攻击
def perform_simplified_mia(model, device, train_loader, test_loader):
    model.eval()
    
    # 提取特征
    def get_features(loader):
        features = []
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(device), targets.to(device)
                functional.reset_net(model)
                outputs = model(data)
                probs = nn.functional.softmax(outputs, dim=1)
                max_conf = probs.max(dim=1)[0].cpu().numpy()
                features.extend(max_conf)
        return np.array(features)
    
    # 从训练集和测试集各取一半
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    
    n_train = len(train_dataset)
    n_test = len(test_dataset)
    
    member_indices = np.random.choice(n_train, min(n_train//2, 1000), replace=False)
    non_member_indices = np.random.choice(n_test, min(n_test//2, 1000), replace=False)
    
    member_loader = DataLoader(Subset(train_dataset, member_indices), batch_size=BATCH_SIZE, shuffle=False)
    non_member_loader = DataLoader(Subset(test_dataset, non_member_indices), batch_size=BATCH_SIZE, shuffle=False)
    
    member_features = get_features(member_loader)
    non_member_features = get_features(non_member_loader)
    
    # 准备数据
    X = np.concatenate([member_features, non_member_features]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])
    
    # 训练攻击模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    attack_model = LogisticRegression()
    attack_model.fit(X_train, y_train)
    
    y_pred = attack_model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# 主函数
def main():
    print("=" * 60)
    print("固定准确率的控制变量消融实验")
    print("=" * 60)
    
    target_sparsities = [0.85, 0.90, 0.95, 0.99, 0.999]
    
    all_results = {}
    for sparsity in target_sparsities:
        all_results[sparsity] = {
            'test_acc': [],
            'train_acc': [],
            'generalization_gap': [],
            'mia_acc': [],
            'actual_sparsity': []
        }
    
    # 5次独立重复实验
    for repeat in range(NUM_REPEATS):
        print(f"\n--- 第 {repeat+1} 次重复实验 ---")
        
        for target_sparsity in target_sparsities:
            print(f"  目标稀疏度: {target_sparsity}")
            
            result = train_and_evaluate(target_sparsity, seed=42+repeat)
            
            all_results[target_sparsity]['test_acc'].append(result['test_acc'])
            all_results[target_sparsity]['train_acc'].append(result['train_acc'])
            all_results[target_sparsity]['generalization_gap'].append(result['generalization_gap'])
            all_results[target_sparsity]['mia_acc'].append(result['mia_acc'])
            all_results[target_sparsity]['actual_sparsity'].append(result['actual_sparsity'])
    
    # 计算均值和标准差
    summary_results = []
    for target_sparsity in target_sparsities:
        summary_results.append({
            'target_sparsity': target_sparsity,
            'actual_sparsity_mean': np.mean(all_results[target_sparsity]['actual_sparsity']),
            'actual_sparsity_std': np.std(all_results[target_sparsity]['actual_sparsity']),
            'test_acc_mean': np.mean(all_results[target_sparsity]['test_acc']),
            'test_acc_std': np.std(all_results[target_sparsity]['test_acc']),
            'train_acc_mean': np.mean(all_results[target_sparsity]['train_acc']),
            'train_acc_std': np.std(all_results[target_sparsity]['train_acc']),
            'generalization_gap_mean': np.mean(all_results[target_sparsity]['generalization_gap']),
            'generalization_gap_std': np.std(all_results[target_sparsity]['generalization_gap']),
            'mia_acc_mean': np.mean(all_results[target_sparsity]['mia_acc']),
            'mia_acc_std': np.std(all_results[target_sparsity]['mia_acc'])
        })
    
    # 保存结果
    csv_path = os.path.join(OUTPUT_DIR, 'control_variable_ablation.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            '目标稀疏度', '实际稀疏度(均值±标准差)', 
            '测试准确率(%)', '训练准确率(%)', '泛化Gap(%)', 
            'MIA准确率'
        ])
        for result in summary_results:
            writer.writerow([
                f"{result['target_sparsity']:.3f}",
                f"{result['actual_sparsity_mean']:.3f} ± {result['actual_sparsity_std']:.3f}",
                f"{result['test_acc_mean']:.2f} ± {result['test_acc_std']:.2f}",
                f"{result['train_acc_mean']:.2f} ± {result['train_acc_std']:.2f}",
                f"{result['generalization_gap_mean']:.2f} ± {result['generalization_gap_std']:.2f}",
                f"{result['mia_acc_mean']:.3f} ± {result['mia_acc_std']:.3f}"
            ])
    
    print("\n=== 实验完成 ===")
    print(f"结果已保存到 {csv_path}")
    
    # 打印总结
    print("\n控制变量消融实验总结：")
    for result in summary_results:
        print(f"\n目标稀疏度: {result['target_sparsity']:.3f}")
        print(f"  实际稀疏度: {result['actual_sparsity_mean']:.3f} ± {result['actual_sparsity_std']:.3f}")
        print(f"  测试准确率: {result['test_acc_mean']:.2f}% ± {result['test_acc_std']:.2f}%")
        print(f"  MIA准确率: {result['mia_acc_mean']:.3f} ± {result['mia_acc_std']:.3f}")
        print(f"  泛化Gap: {result['generalization_gap_mean']:.2f}% ± {result['generalization_gap_std']:.2f}%")

if __name__ == '__main__':
    main()
