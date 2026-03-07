# HemoSparse: SNN 稀疏性量化与隐私保护双重增益验证

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)
![SpikingJelly](https://img.shields.io/badge/SpikingJelly-Latest-orange.svg)

本项目基于 **BloodMNIST** (MedMNIST v2) 医疗影像数据集，使用 **SpikingJelly** 框架全流程开发。核心目标是量化验证**脉冲神经网络(SNN)** 天然的脉冲稀疏性对 **「低功耗推理」** 和 **「隐私保护能力(抗MIA攻击)」** 的双重增益。

> **⚠️ 专属硬件优化**：本项目代码已针对 **NVIDIA RTX 4070 笔记本配置 (8GB GDDR6显存)** 进行深度适配，包含混合精度(AMP)、自适应Batch Size调整与GPU功耗实测逻辑。

---

## 🌟 项目核心亮点
1. **统一拓扑的三组严谨对照**
   - **实验组 A (SNN)**: SpikingJelly多步模式 `LIFNode`，保留天然稀疏性。
   - **对照组 B (DenseSNN)**: 极低阈值强制触发脉冲，破坏稀疏性（100%发放率）。
   - **对照组 C (ANN)**: 替换所有LIF为 `ReLU`，剥离时间维度，作为基础性能与功耗基线。
2. **多维度量化实验**
   - 实验 1：量化不同超参数(T, 阈值)下的时间/空间稀疏度。
   - 实验 2：基于 `pynvml` 在 4070 笔记本上实测单样本推理功耗 (mJ) 与延迟。
   - 实验 3：量化验证SNN对抗黑盒成员推理攻击(MIA)的隐私保护能力。
3. **全链路科研级可视化**
   - 数据预处理：类分布、特征热力图、**脉冲光栅图(Raster Plot)**。
   - 结果分析：稀疏度随Epoch变化、实测能效柱状图、特征-功耗相关性、**综合维度的雷达图**。

---

## 🛠️ 环境依赖与安装 (RTX 4070 笔记本指南)

推荐使用 `conda` 创建纯净环境：

```bash
# 1. 创建并激活环境
conda create -n SNN-Medical python=3.10 -y
conda activate SNN-Medical

# 2. 安装核心库 (请根据您的本机的CUDA版本调整)
pip install torch torchvision

# 3. 安装 SpikingJelly, MedMNIST 及其他图表生成依赖
pip install -r requirements.txt
```

---

## 🚀 快速开始与一键运行

本项目提供了一键式全流程运行脚本 `run_all.py`，它会自动执行：
1. 数据集下载、预处理与可视化图表生成。
2. 调用 `experiments/` 目录下的三个量化实验。
3. 调用 `visualization/` 目录汇总生成所有的结果分析图表。

```bash
# 运行全部验证实验与作图脚本（耗时主要取决于您是否开启了模型训练）
python run_all.py
```
*注：为了便于快速演示，`run_all.py` 中默认屏蔽了极度耗时的模型主体训练阶段。若需全流程从零跑，请取消该文件中对 `train.py` 调用的注释。*

---

## 📂 核心代码结构与功能

```text
HemoSparse/
├── config.py                 # 全局配置参数 (含 4070 显存自适应策略, 路径管理)
├── requirements.txt          # 项目运行环境清单
├── run_all.py                # 一键运行入口
├── data/                     # 数据处理模块
│   ├── dataloader.py         # DataLoader (含泊松编码, AMP适配, SNN/ANN多模式)
│   └── visualize_data.py     # 绘制数据类别、热力图、光栅图等
├── models/                   # 模型定义 (SpikingJelly 框架实现)
│   ├── snn_model.py          # 标准SNN (实验组A)
│   ├── dense_snn_model.py    # 稠密SNN (对照组B, 破坏稀疏性)
│   ├── ann_model.py          # 标准ANN (对照组C)
│   └── sparsity_hooks.py     # SpikingJelly 稀疏度挂载采集器
├── experiments/              # 核心验证实验
│   ├── exp1_sparsity.py      # 实验1：稀疏性统计实验
│   ├── exp2_power.py         # 实验2：pynvml 4070 实测能耗实验
│   └── exp3_privacy.py       # 实验3：MIA黑盒隐私攻击实验
├── visualization/            # 数据大屏与图表生成
│   ├── plot_sparsity.py      # 生成稀疏度动态折线图, 箱线图等
│   ├── plot_power.py         # 生成功耗柱状对比, 相关性散点图等
│   ├── plot_privacy.py       # 生成 MIA 攻击成功率等对比图
│   └── plot_radar.py         # 生成综合评估性能三维雷达图
└── outputs/                  # 运行生成物目录 (包含日志、模型权重、生成的PNG图表与CSV数据)
```

---

## 🔍 RTX 4070 进阶运行说明与排障
- **显存保护机制**：`config.py` 内置了自适应调整 `batch_size` 逻辑。如果您在训练时抛出 `CUDA OOM (Out Of Memory)` 异常，它会自动将批次缩小；如果显存充足，则扩大以加速训练。
- **功耗采集失效**：`pynvml` 依赖于完整安装 Nvidia GPU 驱动。如果无法获取功率，请在终端执行 `nvidia-smi` 确保驱动正常工作。
- **训练收敛慢**：SNN 采用的是泊松脉冲输入，且时间步 `T=20` 具备较强的随机性，通常需要运行比 ANN 更长的 `Epochs` (推荐>50) 才能完全达到其准确率上限。
