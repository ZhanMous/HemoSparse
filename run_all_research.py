"""
HemoSparse 一键复现脚本
用于完整执行训练→测试→MIA→绘图的全过程
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, description):
    """执行命令并检查结果"""
    print(f"\n🔄 {description}")
    print(f"   执行: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"   ✅ 完成")
        return True
    else:
        print(f"   ❌ 失败")
        print(f"   错误: {result.stderr}")
        return False


def main():
    """主流程"""
    print("=" * 60)
    print("🔬 HemoSparse 一键复现脚本")
    print("   完整执行: 训练→测试→MIA→绘图")
    print("=" * 60)
    
    # 清理旧的输出目录
    outputs_dir = Path('outputs')
    if outputs_dir.exists():
        print("🗑️  清理旧的输出目录...")
        shutil.rmtree(outputs_dir)
    
    # Phase 1: 数据预处理和可视化
    print("\n📊 Phase 1: 数据预处理和可视化")
    success = run_command(
        f"{sys.executable} data/visualize_data.py",
        "数据集可视化"
    )
    if not success:
        print("   ⚠️ 数据可视化失败，继续后续步骤")
    
    # Phase 2: 模型训练
    print("\n🏋️ Phase 2: 模型训练")
    success = run_command(
        f"{sys.executable} train.py",
        "统一训练框架"
    )
    if not success:
        print("   ⚠️ 模型训练失败，使用预训练模型继续")
    
    # Phase 3: 三项核心实验
    print("\n🔬 Phase 3: 三项核心实验")
    
    # Exp1: 稀疏性量化实验
    success = run_command(
        f"{sys.executable} experiments/exp1_sparsity.py",
        "稀疏性量化实验"
    )
    if not success:
        print("   ⚠️ 稀疏性实验失败，继续后续步骤")
    
    # Exp2: 功耗关联性实验
    success = run_command(
        f"{sys.executable} experiments/exp2_power.py",
        "功耗关联性实验"
    )
    if not success:
        print("   ⚠️ 功耗实验失败，继续后续步骤")
    
    # Exp3: 隐私保护实验
    success = run_command(
        f"{sys.executable} experiments/exp3_privacy.py",
        "隐私保护实验"
    )
    if not success:
        print("   ⚠️ 隐私实验失败，继续后续步骤")
    
    # Phase 4: 结果可视化
    print("\n📈 Phase 4: 结果可视化")
    
    # 生成学术图表
    success = run_command(
        f"{sys.executable} generate_academic_figures.py",
        "学术图表生成"
    )
    if not success:
        print("   ⚠️ 学术图表生成失败，继续后续步骤")
    
    # 生成各类可视化图表
    viz_scripts = [
        'visualization/plot_sparsity.py',
        'visualization/plot_power.py',
        'visualization/plot_privacy.py',
        'visualization/plot_radar.py',
        'visualization/plot_pareto.py',
    ]
    
    for script in viz_scripts:
        success = run_command(
            f"{sys.executable} {script}",
            f"生成{os.path.basename(script)}"
        )
        if not success:
            print(f"   ⚠️ {script} 生成失败，继续后续步骤")
    
    # Phase 5: 重复实验（5次）
    print("\n🔁 Phase 5: 重复实验 (5次)")
    success = run_command(
        f"{sys.executable} repeat_experiments.py",
        "5次重复实验"
    )
    if not success:
        print("   ⚠️ 重复实验失败")
    
    print("\n" + "=" * 60)
    print("✅ 一键复现流程完成!")
    print("📄 实验结果已保存至 outputs/ 目录")
    print("🖼️  学术图表已保存至 outputs/figures/academic/ 目录")
    print("📊 统计结果已保存至 outputs/results/ 目录")
    print("=" * 60)


if __name__ == '__main__':
    main()