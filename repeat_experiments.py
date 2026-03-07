"""
重复实验脚本
用于执行5次独立实验，获得统计结果（均值±标准差）
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

def run_single_experiment():
    """运行单次完整实验"""
    print("Running single experiment...")
    
    # 运行训练
    train_result = subprocess.run([sys.executable, 'train.py'], 
                                  capture_output=True, text=True, cwd='.')
    if train_result.returncode != 0:
        print(f"Training failed: {train_result.stderr}")
        return None
    
    # 运行实验
    exp_results = {}
    for exp_num in [1, 2, 3]:  # exp1_sparsity, exp2_power, exp3_privacy
        exp_file = f'experiments/exp{exp_num}_*.py'
        # 查找对应的实验文件
        if exp_num == 1:
            exp_script = 'experiments/exp1_sparsity.py'
        elif exp_num == 2:
            exp_script = 'experiments/exp2_power.py'
        else:
            exp_script = 'experiments/exp3_privacy.py'
            
        exp_result = subprocess.run([sys.executable, exp_script], 
                                    capture_output=True, text=True, cwd='.')
        if exp_result.returncode != 0:
            print(f"Experiment {exp_num} failed: {exp_result.stderr}")
            return None
        exp_results[f'exp{exp_num}'] = exp_result.stdout
    
    # 运行可视化
    viz_result = subprocess.run([sys.executable, 'data/visualize_data.py'], 
                                capture_output=True, text=True, cwd='.')
    if viz_result.returncode != 0:
        print(f"Visualization failed: {viz_result.stderr}")
        return None
    
    # 读取结果文件
    try:
        # 读取训练结果
        train_summary = pd.read_csv('outputs/results/training_summary.csv')
        
        # 读取功耗实验结果
        power_result = pd.read_csv('outputs/results/exp2_power_and_latency.csv')
        
        # 读取隐私实验结果
        privacy_result = pd.read_csv('outputs/results/exp3_privacy_mia.csv')
        
        # 读取稀疏性实验结果
        sparsity_result = pd.read_csv('outputs/results/exp1_sparsity_quantification.csv')
        
        return {
            'train_summary': train_summary,
            'power_result': power_result,
            'privacy_result': privacy_result,
            'sparsity_result': sparsity_result
        }
    except Exception as e:
        print(f"Failed to read results: {e}")
        return None


def run_repeated_experiments(n_runs=5):
    """运行重复实验"""
    print(f"Starting {n_runs} repeated experiments...")
    
    all_results = []
    
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")
        
        # 清除上次运行的输出
        import shutil
        outputs_dir = Path('outputs')
        if outputs_dir.exists():
            shutil.rmtree(outputs_dir)
        
        # 运行单次实验
        result = run_single_experiment()
        if result is not None:
            all_results.append(result)
            print(f"Run {i+1} completed successfully")
        else:
            print(f"Run {i+1} failed, skipping...")
    
    if not all_results:
        print("No successful runs completed!")
        return
    
    # 计算统计结果
    print("\nComputing statistics...")
    
    # 提取训练结果
    test_accs = []
    for result in all_results:
        test_accs.append(result['train_summary']['Test_Acc'].values[0])
    
    # 提取隐私结果
    mia_accs = []
    for result in all_results:
        mia_accs.append(result['privacy_result']['MIA_Accuracy'].values[0])
    
    # 提取功耗结果
    energy_per_sample = []
    for result in all_results:
        energy_per_sample.append(result['power_result']['Energy_per_Sample_mJ'].values[0])
    
    # 输出统计结果
    print("\n=== STATISTICS ===")
    print(f"Test Accuracy: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
    print(f"MIA Accuracy: {np.mean(mia_accs):.3f} ± {np.std(mia_accs):.3f}")
    print(f"Energy per Sample: {np.mean(energy_per_sample):.3f} ± {np.std(energy_per_sample):.3f} mJ")
    
    # 保存统计结果
    stats_df = pd.DataFrame({
        'Metric': ['Test Accuracy', 'MIA Accuracy', 'Energy per Sample (mJ)'],
        'Mean': [np.mean(test_accs), np.mean(mia_accs), np.mean(energy_per_sample)],
        'Std': [np.std(test_accs), np.std(mia_accs), np.std(energy_per_sample)]
    })
    
    stats_df.to_csv('outputs/results/repeat_experiment_stats.csv', index=False)
    print("\nStatistics saved to outputs/results/repeat_experiment_stats.csv")


if __name__ == '__main__':
    run_repeated_experiments(5)