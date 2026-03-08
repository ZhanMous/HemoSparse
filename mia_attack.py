# -*- coding: utf-8 -*-
"""
MIA 攻击脚本
- 5个影子模型
- 攻击模型：Logistic Regression
- 输入特征：max confidence + entropy
- 统计检验：双侧 t 检验
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import csv


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reset_model_state(model_name, model):
    if model_name == 'SNN':
        functional.reset_net(model)
    elif model_name == 'DenseSNN' and hasattr(model, 'reset'):
        model.reset()


def get_loaders_for_model(model_name, seed=None):
    mode = 'snn' if model_name in ['SNN', 'DenseSNN'] else 'ann'
    return get_blood_mnist_loaders(batch_size=BATCH_SIZE, mode=mode, seed=seed)

# 统计检验函数
def t_test(data1, data2):
    import scipy.stats as stats
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    return t_stat, p_value

def get_significance_label(p_value):
    if p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

# 超参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
T = 6
V_THRESHOLD = 1.0
NUM_SHADOW_MODELS = 5
NUM_REPEATS = 5

# 保存目录
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 计算熵
def compute_entropy(p):
    return -torch.sum(p * torch.log(p + 1e-10), dim=1)

# 训练影子模型
def train_shadow_model(model_name, shadow_id, seed):
    # 设置随机种子
    set_seed(seed)
    
    # 加载数据
    train_loader, val_loader, test_loader, info = get_loaders_for_model(model_name, seed=seed)
    
    # 分割训练数据为成员和非成员
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    
    # 随机选择一半数据作为成员，一半作为非成员
    n_train = len(train_dataset)
    n_member = n_train // 2
    member_indices = np.random.choice(n_train, n_member, replace=False)
    non_member_indices = np.setdiff1d(np.arange(n_train), member_indices)
    
    member_loader = DataLoader(Subset(train_dataset, member_indices), batch_size=BATCH_SIZE, shuffle=True)
    non_member_loader = DataLoader(Subset(test_dataset, np.arange(len(test_dataset))), batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型
    if model_name == 'SNN':
        model = SNN(T=T, v_threshold=V_THRESHOLD)
    elif model_name == 'DenseSNN':
        model = DenseSNN(T=T, v_threshold=V_THRESHOLD)
    elif model_name == 'ANN':
        model = ANN()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # 训练影子模型
    for epoch in range(EPOCHS):
        model.train()
        for data, targets in member_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            reset_model_state(model_name, model)
            outputs = model(data)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return model, member_loader, non_member_loader

# 提取特征
def extract_features(model, model_name, data_loader):
    device = next(model.parameters()).device
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            reset_model_state(model_name, model)
            outputs = model(data)
            
            # 计算 softmax 概率
            probs = nn.functional.softmax(outputs, dim=1)
            # 提取特征：max confidence + entropy
            max_conf = probs.max(dim=1)[0].cpu().numpy()
            entropy = compute_entropy(probs).cpu().numpy()
            
            # 组合特征
            batch_features = np.column_stack((max_conf, entropy))
            features.append(batch_features)
            labels.append(np.ones(len(batch_features)))
    
    return np.vstack(features), np.concatenate(labels)

# 执行 MIA 攻击
def run_mia_attack(model_name):
    print(f"\n=== 执行 {model_name} 的 MIA 攻击 ===")
    
    # 收集所有影子模型的特征
    all_features = []
    all_labels = []
    
    for i in range(NUM_SHADOW_MODELS):
        print(f"\n--- 训练第 {i+1} 个影子模型 ---")
        shadow_model, member_loader, non_member_loader = train_shadow_model(model_name, i, i)
        
        # 提取成员特征
        member_features, member_labels = extract_features(shadow_model, model_name, member_loader)
        # 提取非成员特征
        non_member_features, non_member_labels = extract_features(shadow_model, model_name, non_member_loader)
        non_member_labels = np.zeros(len(non_member_labels))
        
        # 合并特征
        shadow_features = np.vstack((member_features, non_member_features))
        shadow_labels = np.concatenate((member_labels, non_member_labels))
        
        all_features.append(shadow_features)
        all_labels.append(shadow_labels)
    
    # 合并所有影子模型的特征
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    # 训练攻击模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    attack_model = LogisticRegression(max_iter=1000)
    attack_model.fit(X_train, y_train)
    
    # 测试攻击模型
    y_pred = attack_model.predict(X_test)
    y_pred_proba = attack_model.predict_proba(X_test)[:, 1]
    
    # 计算完整指标
    attack_acc = accuracy_score(y_test, y_pred)
    attack_auc = roc_auc_score(y_test, y_pred_proba)
    attack_f1 = f1_score(y_test, y_pred)
    attack_precision = precision_score(y_test, y_pred, zero_division=0)
    attack_recall = recall_score(y_test, y_pred)
    
    print(f"\n{model_name} MIA 攻击指标:")
    print(f"  Accuracy: {attack_acc:.4f}")
    print(f"  AUC: {attack_auc:.4f}")
    print(f"  F1: {attack_f1:.4f}")
    print(f"  Precision: {attack_precision:.4f}")
    print(f"  Recall: {attack_recall:.4f}")
    
    return {
        'accuracy': attack_acc,
        'auc': attack_auc,
        'f1': attack_f1,
        'precision': attack_precision,
        'recall': attack_recall
    }

# 训练过拟合 ANN（无 weight_decay，100 epoch）
def train_overfit_ann():
    print("\n=== 训练过拟合 ANN ===")
    
    # 加载数据
    train_loader, val_loader, test_loader, info = get_loaders_for_model('ANN')
    
    # 初始化模型
    model = ANN()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器（无 weight_decay）
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    
    # 训练 100 epoch
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%")
    
    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, 'overfit_ann.pth')
    torch.save(model.state_dict(), model_path)
    
    return model

# 主函数
def main():
    models = ['SNN', 'DenseSNN', 'ANN']
    
    # 运行 MIA 攻击（5 次重复实验）
    all_mia_results = {}
    for model in models:
        all_mia_results[model] = {
            'accuracy': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': []
        }
        print(f"\n=== 执行 {model} 的 MIA 攻击 ({NUM_REPEATS} 次重复) ===")
        for repeat in range(NUM_REPEATS):
            print(f"\n--- 第 {repeat+1} 次重复实验 ---")
            metrics = run_mia_attack(model)
            for metric in metrics:
                all_mia_results[model][metric].append(metrics[metric])
    
    # 训练过拟合 ANN 并测试其 MIA 攻击指标
    all_mia_results['Overfit ANN'] = {
        'accuracy': [],
        'auc': [],
        'f1': [],
        'precision': [],
        'recall': []
    }
    print(f"\n=== 训练过拟合 ANN 并执行 MIA 攻击 ({NUM_REPEATS} 次重复) ===")
    for repeat in range(NUM_REPEATS):
        print(f"\n--- 过拟合 ANN 第 {repeat+1} 次重复实验 ---")
        overfit_ann = train_overfit_ann()
        
        # 提取过拟合 ANN 的特征用于 MIA 攻击
        train_loader, _, test_loader, _ = get_loaders_for_model('ANN', seed=repeat)
        
        # 分割数据
        train_dataset = train_loader.dataset
        n_train = len(train_dataset)
        n_member = n_train // 2
        member_indices = np.random.choice(n_train, n_member, replace=False)
        non_member_indices = np.setdiff1d(np.arange(n_train), member_indices)
        
        member_loader = DataLoader(Subset(train_dataset, member_indices), batch_size=BATCH_SIZE, shuffle=True)
        non_member_loader = DataLoader(Subset(test_loader.dataset, np.arange(len(test_loader.dataset))), batch_size=BATCH_SIZE, shuffle=True)
        
        # 提取特征
        member_features, member_labels = extract_features(overfit_ann, 'ANN', member_loader)
        non_member_features, non_member_labels = extract_features(overfit_ann, 'ANN', non_member_loader)
        non_member_labels = np.zeros(len(non_member_labels))
        
        # 合并特征
        X = np.vstack((member_features, non_member_features))
        y = np.concatenate((member_labels, non_member_labels))
        
        # 训练攻击模型
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        attack_model = LogisticRegression(max_iter=1000)
        attack_model.fit(X_train, y_train)
        
        # 测试攻击模型
        y_pred = attack_model.predict(X_test)
        y_pred_proba = attack_model.predict_proba(X_test)[:, 1]
        
        # 计算完整指标
        overfit_ann_mia = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred)
        }
        
        for metric in overfit_ann_mia:
            all_mia_results['Overfit ANN'][metric].append(overfit_ann_mia[metric])
        
        print(f"过拟合 ANN 第 {repeat+1} 次 MIA 攻击指标:")
        print(f"  Accuracy: {overfit_ann_mia['accuracy']:.4f}")
        print(f"  AUC: {overfit_ann_mia['auc']:.4f}")
        print(f"  F1: {overfit_ann_mia['f1']:.4f}")
        print(f"  Precision: {overfit_ann_mia['precision']:.4f}")
        print(f"  Recall: {overfit_ann_mia['recall']:.4f}")
    
    # 计算均值和标准差
    summary_results = {}
    for model in all_mia_results:
        summary_results[model] = {}
        for metric in all_mia_results[model]:
            mean_val = np.mean(all_mia_results[model][metric])
            std_val = np.std(all_mia_results[model][metric])
            summary_results[model][metric] = {'mean': mean_val, 'std': std_val, 'values': all_mia_results[model][metric]}
    
    # 统计检验（与 ANN 对比）
    significance = {}
    if 'ANN' in all_mia_results:
        ann_accuracy_values = all_mia_results['ANN']['accuracy']
        for model in all_mia_results:
            if model != 'ANN' and model != 'Overfit ANN':
                t_stat, p_value = t_test(ann_accuracy_values, all_mia_results[model]['accuracy'])
                significance[model] = {'t_stat': t_stat, 'p_value': p_value, 'label': get_significance_label(p_value)}
    
    # 保存结果
    csv_path = os.path.join(OUTPUT_DIR, 'mia_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy (Mean ± Std)', 'AUC (Mean ± Std)', 'F1 (Mean ± Std)', 'Precision (Mean ± Std)', 'Recall (Mean ± Std)', 'Significance (vs ANN)'])
        for model in summary_results:
            sig_label = significance.get(model, {}).get('label', '')
            writer.writerow([
                model, 
                f"{summary_results[model]['accuracy']['mean']:.4f} ± {summary_results[model]['accuracy']['std']:.4f}",
                f"{summary_results[model]['auc']['mean']:.4f} ± {summary_results[model]['auc']['std']:.4f}",
                f"{summary_results[model]['f1']['mean']:.4f} ± {summary_results[model]['f1']['std']:.4f}",
                f"{summary_results[model]['precision']['mean']:.4f} ± {summary_results[model]['precision']['std']:.4f}",
                f"{summary_results[model]['recall']['mean']:.4f} ± {summary_results[model]['recall']['std']:.4f}",
                sig_label
            ])
    
    print("\n=== MIA 攻击完成 ===")
    print(f"结果已保存到 {csv_path}")
    
    # 打印结果
    print("\nMIA 攻击完整指标 (均值 ± 标准差):")
    for model in summary_results:
        sig_label = significance.get(model, {}).get('label', '')
        print(f"\n  {model} {sig_label}:")
        for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
            print(f"    {metric}: {summary_results[model][metric]['mean']:.4f} ± {summary_results[model][metric]['std']:.4f}")
    
    # 打印统计检验结果
    if significance:
        print("\n统计检验结果 (双侧 t 检验 vs ANN, 仅对比 Accuracy):")
        for model in significance:
            print(f"  {model}: t={significance[model]['t_stat']:.4f}, p={significance[model]['p_value']:.4f} {significance[model]['label']}")

if __name__ == '__main__':
    main()
