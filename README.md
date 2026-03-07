<!-- 项目标题 -->
# 🧠 HemoSparse 项目

> **SNN 医疗AI隐私保护与低功耗推理研究**
> 
> 基于 SpikingJelly 框架，利用 BloodMNIST 影像数据集量化验证脉冲神经网络（SNN）在"低功耗推理"与"隐私保护能力"的双重优势
> 
> **⚠️ 硬件优化**：本项目代码已针对现代GPU架构进行深度适配，包含混合精度(AMP)、自适应Batch Size调整与GPU功耗实测逻辑。

<!-- 项目徽章 -->
<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red?logo=pytorch&logoColor=white)
![SpikingJelly](https://img.shields.io/badge/spikingjelly-≥0.0.0.0.14-yellow?logo=github&logoColor=white)
![License](https://img.shields.io/github/license/your-repo/your-project)

</div>

<!-- 项目简介 -->
## 📋 项目简介

**HemoSparse** 是一个基于 SpikingJelly 框架的科研项目，旨在利用 **BloodMNIST** 医疗影像数据集量化验证脉冲神经网络（SNN）在"低功耗推理"与"隐私保护能力"的双重优势。

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

<!-- 安装说明 -->
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
- **基础框架**: Python 3.9+, PyTorch 2.1+, SpikingJelly (≥0.0.0.0.14)
- **科学计算**: numpy, pandas, scipy, scikit-learn
- **可视化**: matplotlib, plotly, seaborn, kaleido
- **隐私工具**: opacus (差分隐私辅助)
- **GPU监控**: nvidia-ml-py (pynvml)

<!-- 快速开始 -->
## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone <repository-url>
cd HemoSparse
```

### 2. 一键运行全流程
```bash
python run_all.py
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
```

<!-- 项目结构 -->
## 📁 项目结构

```
HemoSparse/
├── config.py          # 全局配置与超参管理
├── run_all.py         # 一键运行全流程入口
├── train.py           # 统一训练框架
├── data/              # 数据处理模块
│   ├── dataloader.py  # 数据加载与预处理
│   └── visualize_data.py # 数据集可视化
├── models/            # 模型定义
│   └── improved_snn.py # 改进SNN模型
├── experiments/       # 三组核心实验
│   ├── exp1_sparsity.py   # 稀疏性量化实验
│   ├── exp2_power.py      # 功耗关联性实验
│   └── exp3_privacy.py    # 隐私保护实验
├── visualization/     # 可视化模块
│   ├── plot_sparsity.py   # 稀疏性可视化
│   ├── plot_power.py      # 功耗分析图
│   ├── plot_privacy.py    # 隐私性能图
│   ├── plot_radar.py      # 雷达对比图
│   └── plot_pareto.py     # 帕累托前沿图
├── outputs/           # 输出目录
│   ├── models/        # 训练模型
│   ├── results/       # 实验结果
│   └── figures/       # 可视化图表
└── requirements.txt   # 依赖清单
```

<!-- 运行流程 -->
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

<!-- 技术细节 -->
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

<!-- 贡献指南 -->
## 🤝 贡献指南

欢迎提交 PR 和 Issue！请遵循以下步骤：
1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

<!-- 许可证 -->
## 📄 许可证

本项目采用 MIT 许可证 - 查阅 [LICENSE](LICENSE) 文件了解详情

<!-- 进阶说明 -->
## 🔍 通用运行说明与排障

### 硬件兼容性
- **最低配置**: 4GB GPU显存, 8GB RAM
- **推荐配置**: 8GB+ GPU显存, 16GB+ RAM
- **支持平台**: Linux, Windows, macOS (CPU模式)

### 性能优化
- **混合精度**: 启用AMP加速训练
- **批大小自适应**: 显存不足时自动降低batch_size
- **内存清理**: 周期性清空GPU缓存

### 常见问题
Q: 出现CUDA OOM错误怎么办？  
A: 系统会自动降低batch_size，也可手动减小config.py中的BATCH_SIZE

Q: 无法安装SpikingJelly？  
A: 尝试使用官方源: pip install spikingjelly

Q: 功耗测试失败？  
A: 检查NVIDIA驱动与pynvml安装，或在CPU模式下跳过功耗测试