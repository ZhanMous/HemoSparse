# -*- coding: utf-8 -*-
"""
HemoSparse 自动化全流程执行脚本
一键运行：环境检查 -> 数据可视化 -> 模型训练 -> 三个核心实验 -> 全维度可视化出图
"""

import os
import sys

def run_script(script_path, desc):
    print(f"\n[{desc}] 正在执行 {script_path} ...")
    ret = os.system(f"{sys.executable} {script_path}")
    if ret != 0:
        print(f"❌ [{desc}] 失败，中止全流程！")
        sys.exit(1)
    print(f"✅ [{desc}] 成功完成！")

def main():
    print("="*60)
    print("🚀 HemoSparse: SNN稀疏性量化与隐私保护双重增益验证 🚀")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 基础环境测试与数据可视化
    run_script(os.path.join(base_dir, 'data', 'visualize_data.py'), "Phase 1: 数据集泊松编码验证与可视化")
    
    # 2. 模型训练 (这步极其耗时，如果已有模型可注释掉)
    run_script(os.path.join(base_dir, 'train.py'), "Phase 3: 训练三组对照模型(SNN, DenseSNN, ANN)")
    print("\n⚠️ 提示：为了自动演示，当前跳过完整训练流程 `train.py`。如需完整训练，请取消 run_all.py 第28行注释。")
    
    # 3. 核心实验 (理论上依赖训练结果，如果无结果则使用未收敛模型测试流程)
    run_script(os.path.join(base_dir, 'experiments', 'exp1_sparsity.py'), "Phase 4.1: SNN 稀疏性量化实验")
    run_script(os.path.join(base_dir, 'experiments', 'exp2_power.py'), "Phase 4.2: RTX 4070 专属硬件功耗验证实验")
    run_script(os.path.join(base_dir, 'experiments', 'exp3_privacy.py'), "Phase 5: 隐私保护MIA攻击验证实验")
    
    # 4. 全维度可视化
    run_script(os.path.join(base_dir, 'visualization', 'plot_sparsity.py'), "Phase 6.1: 生成稀疏性图表")
    run_script(os.path.join(base_dir, 'visualization', 'plot_power.py'), "Phase 6.2: 生成功耗与延迟图表")
    run_script(os.path.join(base_dir, 'visualization', 'plot_privacy.py'), "Phase 6.3: 生成隐私抗性图表")
    run_script(os.path.join(base_dir, 'visualization', 'plot_radar.py'), "Phase 6.4: 生成综合雷达对比图")

    print("\n" + "="*60)
    print("🎉 全流程执行完毕！所有图表和CSV数据已保存至 HemoSparse/outputs/ 目录。")
    print("="*60)

if __name__ == '__main__':
    main()
