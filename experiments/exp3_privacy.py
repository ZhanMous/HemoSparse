# -*- coding: utf-8 -*-
"""
实验3：SNN 稀疏性与隐私保护能力关联性验证
- 证明「参数冗余度降低→隐私泄露风险降低」
- 成员推理攻击 (MIA, Threshold-based Black-box)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, RESULTS_DIR, set_seed
from data.dataloader import get_blood_mnist_loaders
from models.snn_model import SNN
from models.dense_snn_model import DenseSNN
from models.ann_model import ANN
from spikingjelly.activation_based import functional

def threshold_mia(model, train_loader, test_loader, is_snn=True):
    """
    基于置信度阈值的黑盒成员推理测试 (MIA)
    利用模型对训练集(成员)和测试集(非成员)的输出置信度分布差异
    返回: MIA攻击准确率
    """
    model.eval()
    train_confidences = []
    test_confidences = []
    
    with torch.no_grad():
        # 获取成员 (训练集) 置信度
        for i, (inputs, _) in enumerate(train_loader):
            if i > 20: break # 取样评估即可，节约时间
            inputs = inputs.to(DEVICE)
            if is_snn: functional.reset_net(model)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            train_confidences.extend(max_probs.cpu().numpy())
            
        # 获取非成员 (测试集) 置信度
        for i, (inputs, _) in enumerate(test_loader):
            if i > 20: break
            inputs = inputs.to(DEVICE)
            if is_snn: functional.reset_net(model)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            test_confidences.extend(max_probs.cpu().numpy())
            
    # 计算最优分界阈值 (简单起见，取两者中位数中间值)
    threshold = (np.median(train_confidences) + np.median(test_confidences)) / 2.0
    
    # 预测：大于阈值认为是成员(1)，小于认为是非成员(0)
    y_true = np.concatenate([np.ones(len(train_confidences)), np.zeros(len(test_confidences))])
    y_pred = np.concatenate([
        (np.array(train_confidences) >= threshold).astype(int),
        (np.array(test_confidences) >= threshold).astype(int)
    ])
    
    mia_acc = accuracy_score(y_true, y_pred)
    # 差分隐私下理论值接近0.5(盲猜)，普通模型通常在0.6-0.8左右
    return mia_acc, np.mean(train_confidences), np.mean(test_confidences)

def run_experiment_3_privacy():
    print("\n" + "="*60)
    print("实验3：SNN 稀疏性与隐私保护能力关联性验证 (MIA)")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed()
    
    # 需要先有训练好的模型权重，这里假如初始化后表现差异
    # 真正严格测试需要在跑完 train.py 后加载 best checkpoints
    # 实验代码先搭建好测试逻辑
    
    models = {
        'SNN (Sparse)': SNN(T=20).to(DEVICE),
        'Dense_SNN': DenseSNN(T=20).to(DEVICE),
        'ANN': ANN().to(DEVICE)
    }
    
    test_loader_snn, train_loader_snn, _, _ = get_blood_mnist_loaders(batch_size=32, T=20, mode='snn')
    test_loader_ann, train_loader_ann, _, _ = get_blood_mnist_loaders(batch_size=32, T=20, mode='ann')
    
    # 交换一下变量名以供评估
    train_snn = train_loader_snn
    test_snn = test_loader_snn
    train_ann = train_loader_ann
    test_ann = test_loader_ann
    
    results = []
    
    for name, model in models.items():
        print(f"\n[隐私测试] {name} ...")
        is_snn = 'SNN' in name
        
        train_l = test_loader_snn if is_snn else test_loader_ann
        test_l = train_loader_snn if is_snn else train_loader_ann
        
        # 运行 MIA 攻击
        mia_acc, train_conf, test_conf = threshold_mia(model, train_l, test_l, is_snn=is_snn)
        
        res = {
            'Model': name,
            'MIA_Accuracy': mia_acc,
            'Train_Confidence_Mean': train_conf,
            'Test_Confidence_Mean': test_conf,
            'Conf_Gap': train_conf - test_conf
        }
        results.append(res)
        
        print(f"  MIA 攻击准确率: {mia_acc*100:.2f}% (越近50%越安全) | 置信度差: {res['Conf_Gap']:.4f}")
        
    df = pd.DataFrame(results)
    save_path = os.path.join(RESULTS_DIR, 'exp3_privacy_mia.csv')
    df.to_csv(save_path, index=False)
    print(f"\n实验3完成！结果已保存至 {save_path}")

if __name__ == '__main__':
    run_experiment_3_privacy()
