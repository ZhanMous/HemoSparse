#!/bin/bash
# HemoSparse 一键运行脚本
# 执行完整的实验流程

set -e

echo "=========================================="
echo "  HemoSparse 完整实验流程"
echo "=========================================="

# 创建输出目录
mkdir -p outputs
mkdir -p outputs/figures

# 1. 生成 IEEE 三线表（示例结果）
echo ""
echo "步骤 1/5: 生成 IEEE 三线表"
python generate_ieee_tables.py

# 2. 训练模型
echo ""
echo "步骤 2/5: 训练模型 (SNN, DenseSNN, ANN)"
echo "注意：这需要较长时间，5次重复实验"
python train.py

# 3. 执行 MIA 攻击
echo ""
echo "步骤 3/5: 执行 MIA 攻击"
python mia_attack.py

# 4. 运行稀疏度消融实验
echo ""
echo "步骤 4/5: 运行稀疏度消融实验"
python ablation_study.py

# 5. 生成学术图表
echo ""
echo "步骤 5/5: 生成学术图表"
python generate_academic_figures.py

echo ""
echo "=========================================="
echo "  所有实验完成！"
echo "=========================================="
echo ""
echo "结果文件位置："
echo "- 训练结果: outputs/training_summary.csv"
echo "- MIA 攻击结果: outputs/mia_results.csv"
echo "- 消融实验结果: outputs/ablation_results.csv"
echo "- 学术图表: outputs/figures/"
echo "- 完整论文: FINAL_RESEARCH_PAPER.md"
echo ""
