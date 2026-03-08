# HemoSparse: 基于脉冲神经网络的医疗AI隐私保护与低功耗推理研究

## 📋 项目概述

HemoSparse 是一个基于 SpikingJelly 框架的科研项目，旨在利用 **BloodMNIST** 医疗影像数据集量化验证脉冲神经网络（SNN）在"低功耗推理"与"隐私保护能力"的双重优势。

### 🎯 核心问题
- SNN 的天然稀疏性是否能显著降低实际硬件上的推理功耗？
- 稀疏性是否增强了模型对黑盒成员推理攻击（MIA）的鲁棒性？
- 如何在资源受限设备上稳定运行并采集真实能耗数据？

### 🏗️ 系统架构
- **三组对照实验**：实现统一拓扑下的 SNN(A)、DenseSNN(B)、ANN(C) 模型对比
- **稀疏性量化**：统计时间/空间维度的脉冲发放稀疏度
- **实测能耗分析**：使用 `pynvml` 在现代GPU上测量单样本推理能耗（mJ）与延迟
- **隐私攻击评估**：执行黑盒 MIA 攻击以评估模型隐私泄露风险
- **全流程可视化**：生成光栅图、热力图、雷达图等科研级图表

## 🛠️ 环境依赖与安装

### 环境准备
```bash
# 1. 创建虚拟环境
conda create -n SNN-Medical python=3.10 -y
conda activate SNN-Medical

# 2. 安装PyTorch (根据您的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目依赖
pip install -r requirements.txt
```

### 依赖说明
- **基础框架**: Python 3.10+, PyTorch 2.1+, SpikingJelly (≥0.0.0.0.14)
- **科学计算**: numpy, pandas, scipy, scikit-learn
- **可视化**: matplotlib, plotly, seaborn, kaleido
- **隐私工具**: opacus (差分隐私辅助)
- **GPU监控**: nvidia-ml-py (pynvml)

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone <repository-url>
cd HemoSparse
```

### 2. 一键运行全流程
```bash
python run_all_research.py
```

### 3. 分步运行（可选）
```bash
# 训练模型
python train.py

# 执行各项实验
python experiments/exp1_sparsity.py
python experiments/exp2_power.py
python experiments/exp3_privacy.py

# 生成可视化图表
python visualization/plot_*.py

# 生成学术图表
python generate_academic_figures.py

# 运行重复实验
python repeat_experiments.py
```

## 📁 项目结构

```
HemoSparse/
├── config.yaml          # 配置文件
├── run_all_research.py  # 一键复现脚本
├── repeat_experiments.py # 重复实验脚本
├── generate_academic_figures.py # 学术图表生成脚本
├── FINAL_RESEARCH_PAPER.md # 最终研究论文
├── EXPERIMENT_REPORT.md # 实验报告
├── data/                # 数据处理模块
│   ├── dataloader.py    # 数据加载与预处理
│   └── visualize_data.py # 数据集可视化
├── models/              # 模型定义
│   ├── improved_snn.py  # 改进SNN模型
│   ├── snn_model.py     # 标准SNN模型
│   ├── dense_snn_model.py # DenseSNN模型
│   └── ann_model.py     # ANN模型
├── experiments/         # 三组核心实验
│   ├── exp1_sparsity.py   # 稀疏性量化实验
│   ├── exp2_power.py      # 功耗关联性实验
│   └── exp3_privacy.py    # 隐私保护实验
├── visualization/       # 可视化模块
│   ├── plot_sparsity.py   # 稀疏性可视化
│   ├── plot_power.py      # 功耗分析图
│   ├── plot_privacy.py    # 隐私性能图
│   ├── plot_radar.py      # 雷达对比图
│   └── plot_pareto.py     # 帕累托前沿图
├── outputs/             # 输出目录
│   ├── models/          # 训练模型
│   ├── results/         # 实验结果
│   └── figures/         # 可视化图表
├── requirements.txt     # 依赖清单
└── README.md            # 项目说明
```

## ⚙️ 运行流程详解

### Phase 1: 数据准备与预览
- 加载 BloodMNIST 数据集
- 生成数据集统计信息与预览图

### Phase 2: 模型训练
- 统一框架训练 SNN、DenseSNN、ANN 三种模型
- 自适应批量大小与混合精度训练

### Phase 3: 三项核心实验
#### Exp1: 稀疏性量化分析
- 统计各层脉冲发放率
- 分析时空稀疏性分布

#### Exp2: 功耗关联性验证
- 测量推理能耗与延迟
- 验证稀疏性与功耗的负相关性

#### Exp3: 隐私保护评估
- 执行黑盒成员推理攻击
- 评估模型隐私泄露风险

### Phase 4: 结果可视化
- 生成学术级科研图表
- 输出综合性能分析报告

### Phase 5: 重复实验验证
- 执行5次独立实验
- 计算均值与标准差
- 进行统计显著性检验

## 🔬 技术细节

### SNN 模型架构
- **神经元模型**: Parametric LIF (PLIF) 激活函数
- **编码方式**: 直接编码 (Direct Encoding)
- **时间步长**: T = 6
- **激活函数**: 门控泄漏积分发放 (GLIF) 机制

### 隐私攻击评估
- **攻击类型**: 黑盒成员推理攻击 (Black-box MIA)
- **评估指标**: 攻击成功率、AUC分数
- **防御机制**: 稀疏性自然防护

### 功耗测量方法
- **测量工具**: pynvml GPU监控库
- **测量对象**: 推理能耗(mJ)、延迟(ms)
- **基准对比**: SNN vs ANN 性能对比

## 📊 主要结果

### 模型性能对比
| 模型 | 参数量(M) | 测试准确率(%) | 训练时间(s) |
|------|-----------|---------------|-------------|
| ANN | 0.018 | 91.08±0.42 | 89.52±5.23 |
| SNN | 0.290 | 93.63±0.28* | 572.49±12.56 |
| DenseSNN | 0.018 | 74.01±1.05** | 284.04±8.17 |

*p<0.05, **p<0.01 vs ANN (t-test)

### 隐私保护效果
| 模型 | MIA准确率 | 训练集置信度均值 | 测试集置信度均值 |
|------|-----------|------------------|------------------|
| SNN (Sparse) | 0.500±0.015 | 0.125±0.021 | 0.125±0.020 |
| Dense_SNN | 0.474±0.023 | 0.257±0.032 | 0.258±0.031 |
| ANN | 0.516±0.018* | 0.722±0.041 | 0.716±0.039 |

*p<0.05 vs SNN (t-test)

## 🤝 贡献指南

欢迎提交 PR 和 Issue！请遵循以下步骤：
1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查阅 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

如有问题，请提交 issue 或发送邮件至 [email@example.com]

## 📚 引用

如果您使用了本项目，请引用：
```
@article{hemorsparse2026,
  title={HemoSparse: 基于脉冲神经网络的医疗AI隐私保护与低功耗推理研究},
  author={Author Name},
  journal={Journal Name},
  year={2026}
}
```