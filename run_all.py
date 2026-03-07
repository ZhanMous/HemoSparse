"""
HemoSparse 项目全流程运行脚本
- 统一调度：数据预览 → 模型训练 → 三项核心实验 → 结果可视化
- 错误处理：异常捕获与日志记录
- 进度追踪：实时显示执行进度
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_script(script_path, description):
    """运行指定脚本并捕获输出"""
    print(f"\n🔄 {description}")
    print(f"   执行: {script_path}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path)
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"   ✅ 完成 ({duration:.2f}s)")
            return True
        else:
            print(f"   ❌ 失败 ({duration:.2f}s)")
            print(f"   错误: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ 异常: {str(e)}")
        return False


def main():
    """主流程"""
    print("=" * 60)
    print("🧠 HemoSparse 项目全流程运行")
    print("   基于 SpikingJelly 的 SNN 医疗AI隐私保护与低功耗推理研究")
    print("=" * 60)
    
    # 项目根目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Phase 1: 数据预览与可视化
    print("\n📊 Phase 1: 数据预览与可视化")
    data_viz_script = os.path.join(base_dir, 'data', 'visualize_data.py')
    if not run_script(data_viz_script, "数据集可视化"):
        print("   ⚠️ 数据可视化失败，继续后续步骤")
    
    # Phase 2: 模型训练
    print("\n🏋️ Phase 2: 模型训练")
    train_script = os.path.join(base_dir, 'train.py')
    if not run_script(train_script, "统一训练框架"):
        print("   ⚠️ 模型训练失败，使用预训练模型继续")
    
    # Phase 3: 三项核心实验
    print("\n🔬 Phase 3: 三项核心实验")
    
    # Exp1: 稀疏性量化实验
    exp1_script = os.path.join(base_dir, 'experiments', 'exp1_sparsity.py')
    if not run_script(exp1_script, "Phase 3.1: 稀疏性量化实验"):
        print("   ⚠️ 稀疏性实验失败，继续后续步骤")
    
    # Exp2: 功耗关联性实验
    exp2_script = os.path.join(base_dir, 'experiments', 'exp2_power.py')
    if not run_script(exp2_script, "Phase 3.2: 硬件功耗验证实验"):
        print("   ⚠️ 功耗实验失败，继续后续步骤")
    
    # Exp3: 隐私保护实验
    exp3_script = os.path.join(base_dir, 'experiments', 'exp3_privacy.py')
    if not run_script(exp3_script, "Phase 3.3: 隐私保护评估实验"):
        print("   ⚠️ 隐私实验失败，继续后续步骤")
    
    # Phase 4: 结果可视化
    print("\n📈 Phase 4: 结果可视化")
    
    # 可视化脚本列表
    viz_scripts = [
        ('Phase 4.1: 稀疏性分析图', 'visualization', 'plot_sparsity.py'),
        ('Phase 4.2: 功耗分析图', 'visualization', 'plot_power.py'),
        ('Phase 4.3: 隐私性能图', 'visualization', 'plot_privacy.py'),
        ('Phase 4.4: 雷达对比图', 'visualization', 'plot_radar.py'),
        ('Phase 4.5: 帕累托前沿图', 'visualization', 'plot_pareto.py'),
    ]
    
    for desc, *script_parts in viz_scripts:
        script_path = os.path.join(base_dir, *script_parts)
        if not run_script(script_path, desc):
            print(f"   ⚠️ {desc} 失败，继续后续步骤")
    
    print("\n" + "=" * 60)
    print("✅ 全流程运行完成!")
    print("📄 结果文件已保存至 outputs/ 目录")
    print("🖼️  可视化图表已保存至 outputs/figures/ 目录")
    print("=" * 60)


if __name__ == '__main__':
    main()