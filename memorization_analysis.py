# -*- coding: utf-8 -*-
"""
训练数据记忆程度分析
- 样本级记忆分数计算
- 影响函数计算
- SNN/ANN/DenseSNN对比
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import functional
from data.dataloader import get_blood_mnist_loaders
from models import SNN, DenseSNN, ANN
import numpy as np
import csv
import warnings
warnings.filterwarnings('ignore')

# 超参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
T = 6
V_THRESHOLD = 1.0
NUM_SAMPLES = 500  # 用于分析的样本数

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

# 训练单个模型
def train_model(model_name, seed):
    set_seed(seed)
    
    # 加载数据
    train_loader, _, test_loader, _ = get_blood_mnist_loaders(
        batch_size=BATCH_SIZE, 
        mode='snn' if model_name in ['SNN', 'DenseSNN'] else 'ann'
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 训练模型
    for epoch in range(EPOCHS):
        model.train()
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if model_name in ['SNN', 'DenseSNN']:
                functional.reset_net(model)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    return model, train_loader, test_loader, device

# 计算记忆分数
def compute_memorization_scores(model, model_name, train_loader, test_loader, device):
    model.eval()
    
    # 从训练集和测试集各取NUM_SAMPLES个样本
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    
    train_indices = np.random.choice(len(train_dataset), min(NUM_SAMPLES, len(train_dataset)), replace=False)
    test_indices = np.random.choice(len(test_dataset), min(NUM_SAMPLES, len(test_dataset)), replace=False)
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_subloader = DataLoader(train_subset, batch_size=1, shuffle=False)
    test_subloader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
    # 获取训练样本置信度
    train_confidences = []
    with torch.no_grad():
        for data, targets in train_subloader:
            data, targets = data.to(device), targets.to(device)
            
            if model_name in ['SNN', 'DenseSNN']:
                functional.reset_net(model)
            
            outputs = model(data)
            probs = nn.functional.softmax(outputs, dim=1)
            max_conf = probs.max(dim=1)[0].item()
            train_confidences.append(max_conf)
    
    # 获取测试样本置信度
    test_confidences = []
    with torch.no_grad():
        for data, targets in test_subloader:
            data, targets = data.to(device), targets.to(device)
            
            if model_name in ['SNN', 'DenseSNN']:
                functional.reset_net(model)
            
            outputs = model(data)
            probs = nn.functional.softmax(outputs, dim=1)
            max_conf = probs.max(dim=1)[0].item()
            test_confidences.append(max_conf)
    
    # 计算平均测试置信度
    mean_test_confidence = np.mean(test_confidences)
    
    # 计算记忆分数
    memorization_scores = [conf - mean_test_confidence for conf in train_confidences]
    
    return {
        'train_confidences': np.array(train_confidences),
        'test_confidences': np.array(test_confidences),
        'memorization_scores': np.array(memorization_scores),
        'mean_test_confidence': mean_test_confidence
    }

# 简化版影响函数计算（使用一阶近似）
def compute_influence_functions(model, model_name, train_loader, device):
    model.eval()
    
    train_dataset = train_loader.dataset
    subset_indices = np.random.choice(len(train_dataset), min(100, len(train_dataset)), replace=False)
    subset = Subset(train_dataset, subset_indices)
    subloader = DataLoader(subset, batch_size=1, shuffle=False)
    
    influences = []
    
    for data, target in subloader:
        data, target = data.to(device), target.to(device)
        
        # 计算梯度
        model.zero_grad()
        
        if model_name in ['SNN', 'DenseSNN']:
            functional.reset_net(model)
        
        outputs = model(data)
        loss = nn.CrossEntropyLoss()(outputs, target)
        loss.backward()
        
        # 收集梯度范数作为影响函数的近似
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += torch.norm(param.grad).item() ** 2
        
        influences.append(np.sqrt(grad_norm))
    
    return np.array(influences)

# 主函数
def main():
    print("=" * 60)
    print("训练数据记忆程度分析")
    print("=" * 60)
    
    models = ['SNN', 'DenseSNN', 'ANN']
    
    all_results = {}
    
    for model_name in models:
        print(f"\n--- 分析 {model_name} ---")
        
        # 训练模型
        model, train_loader, test_loader, device = train_model(model_name, seed=42)
        
        # 计算记忆分数
        mem_results = compute_memorization_scores(
            model, model_name, train_loader, test_loader, device
        )
        
        # 计算影响函数
        influences = compute_influence_functions(
            model, model_name, train_loader, device
        )
        
        all_results[model_name] = {
            'memorization_scores': mem_results['memorization_scores'],
            'train_confidences': mem_results['train_confidences'],
            'test_confidences': mem_results['test_confidences'],
            'influences': influences
        }
    
    # 计算统计量
    summary_results = []
    for model_name in models:
        results = all_results[model_name]
        summary_results.append({
            'model': model_name,
            'mean_memorization_score': np.mean(results['memorization_scores']),
            'std_memorization_score': np.std(results['memorization_scores']),
            'mean_train_confidence': np.mean(results['train_confidences']),
            'std_train_confidence': np.std(results['train_confidences']),
            'mean_test_confidence': np.mean(results['test_confidences']),
            'std_test_confidence': np.std(results['test_confidences']),
            'mean_influence': np.mean(results['influences']),
            'std_influence': np.std(results['influences'])
        })
    
    # 保存结果
    csv_path = os.path.join(OUTPUT_DIR, 'memorization_analysis.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            '模型', '平均记忆分数', '记忆分数标准差',
            '平均训练置信度', '训练置信度标准差',
            '平均测试置信度', '测试置信度标准差',
            '平均影响函数', '影响函数标准差'
        ])
        for result in summary_results:
            writer.writerow([
                result['model'],
                f"{result['mean_memorization_score']:.4f} ± {result['std_memorization_score']:.4f}",
                f"{result['mean_train_confidence']:.4f} ± {result['std_train_confidence']:.4f}",
                f"{result['mean_test_confidence']:.4f} ± {result['std_test_confidence']:.4f}",
                f"{result['mean_influence']:.4f} ± {result['std_influence']:.4f}"
            ])
    
    print("\n=== 分析完成 ===")
    print(f"结果已保存到 {csv_path}")
    
    # 打印总结
    print("\n记忆程度分析总结：")
    for result in summary_results:
        print(f"\n{result['model']}:")
        print(f"  平均记忆分数: {result['mean_memorization_score']:.4f} ± {result['std_memorization_score']:.4f}")
        print(f"  平均影响函数: {result['mean_influence']:.4f} ± {result['std_influence']:.4f}")

if __name__ == '__main__':
    main()
