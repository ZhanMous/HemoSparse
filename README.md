<!-- 项目标题 -->
# 🧠 HemoSparse 项目

> **SNN 医疗AI隐私保护与低功耗推理研究**
> 
> 基于 SpikingJelly 框架，利用 BloodMNIST 影像数据集量化验证脉冲神经网络（SNN）在"低功耗推理"与"隐私保护能力"的双重优势
> 
> **⚠️ 硬件优化**：本项目代码已针对现代GPU架构进行深度适配，包含混合精度(AMP)、梯度累积与裁剪、余弦退火学习率调度、non-blocking数据传输等优化，充分利用GPU资源，降低CPU负载。

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
- **全流程可视化**：生成IEEE标准学术图表
- **统计检验**：双侧t检验验证结果显著性

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
- **可视化**: matplotlib, seaborn
- **隐私工具**: scikit-learn (用于MIA攻击)
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
chmod +x run_all.sh
./run_all.sh
```

### 3. 分步运行（可选）
```bash
# 生成 IEEE 三线表（示例结果）
python generate_ieee_tables.py

# 训练模型
python train.py

# 执行 MIA 攻击
python mia_attack.py

# 运行稀疏度消融实验
python ablation_study.py

# 生成学术图表
python generate_academic_figures.py
```

<!-- 项目结构 -->
## 📁 项目结构

```
HemoSparse/
├── models.py                 # 整合模型定义（SNN、DenseSNN、ANN）
├── train.py                  # 统一训练框架（5次重复实验，GPU优化）
├── mia_attack.py             # MIA 攻击实现（含统计检验）
├── ablation_study.py         # 稀疏度消融实验（含MIA准确率）
├── generate_ieee_tables.py   # IEEE 三线表生成
├── generate_academic_figures.py # 学术图表生成
├── utils.py                  # 工具函数（统计检验、表格生成）
├── run_all.sh                # 一键运行全流程
├── config.yaml               # 超参数配置
├── requirements.txt          # 依赖清单
├── README.md                 # 项目说明
├── FINAL_RESEARCH_PAPER.md   # 完整投稿版论文
├── data/                     # 数据处理模块
│   ├── dataloader.py         # 数据加载与预处理
│   └── __init__.py
├── models/                   # 旧版模型定义（保留参考）
├── outputs/                  # 输出目录
│   ├── checkpoints/          # 模型检查点
│   ├── results/              # 实验结果
│   └── figures/              # 可视化图表
└── __pycache__/              # Python缓存
```

<!-- 核心文件说明 -->
## 📄 核心文件说明

### 模型定义 (models.py)
- **SNN**: MS-ResNet结构，PLIF神经元（α为可学习参数），时间步T=6，保持稀疏计算
- **DenseSNN**: 与SNN同结构同阈值，关闭稀疏计算，强制全张量稠密计算
- **ANN**: 同拓扑结构，ReLU激活，无时间维度
- **参数量**: SNN/DenseSNN = 117,248 = 0.117M

### 训练脚本 (train.py)
- 5次独立重复实验
- **GPU性能优化**:
  - 混合精度训练 (AMP)
  - 梯度累积 (Gradient Accumulation)
  - 梯度裁剪 (Gradient Clipping)
  - AdamW 优化器
  - 余弦退火学习率调度
  - non-blocking 数据传输
  - cudnn 性能优化
- 记录: test_acc / training_time / power / latency

### MIA攻击脚本 (mia_attack.py)
- 5个影子模型
- 攻击模型: Logistic Regression
- 输入特征: max confidence + entropy
- 过拟合ANN验证攻击有效性（MIA准确率≥0.72）
- 双侧t检验统计显著性

### 消融实验脚本 (ablation_study.py)
- 遍历 v_threshold = [0.5, 0.75, 1.0, 1.5]
- 记录: sparsity / test_acc / MIA_acc
- 5次独立重复实验

<!-- 论文修正要点 -->
## 📝 论文修正要点 (P0)

本项目已完成所有学术硬伤修复：

### 1. 统一参数量
- ✅ SNN 总参数量：117,248 = 0.117M
- ✅ 修正表格中 0.290M → 0.117M
- ✅ 加注释：SNN 比 ANN 多的参数来自 PLIF 可学习衰减系数 α，结构完全对等

### 2. 重写 DenseSNN 定义
- ✅ DenseSNN 与 SNN 结构、阈值、训练完全一样
- ✅ 唯一区别：关闭稀疏计算优化，强制全张量稠密计算
- ✅ 禁止用 v_threshold=0.001 这种破坏模型的写法

### 3. 补充 MIA 攻击有效性验证
- ✅ 加一个过拟合 ANN（无 weight_decay，100 epoch）
- ✅ 要求其 MIA 攻击准确率 ≥ 0.72
- ✅ 加到论文表格里，证明攻击有效

### 4. 补充稀疏度梯度消融表
- ✅ 变量：T = 6, v_threshold = [0.5, 0.75, 1.0, 1.5]
- ✅ 输出表格：稀疏度 / 测试 acc / MIA acc（带均值 ± 标准差）

### 5. 统一术语
- ✅ Dense_SNN → 全部改为 DenseSNN

### 6. 参考文献
- ✅ 参考文献 [7] 换成 SNN + 隐私 / MIA 相关论文

### 7. PLIF 公式补全
- ✅ 注明：α 是可学习参数，对应膜时间常数 τ

<!-- 运行流程详解 -->
## ⚙️ 运行流程详解

### Phase 1: 生成示例表格
- 生成 IEEE 三线表示例结果
- 用于快速预览论文格式

### Phase 2: 模型训练
- 统一框架训练 SNN、DenseSNN、ANN 三种模型
- 混合精度训练与GPU性能优化
- 5次独立重复实验

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
- 过拟合ANN验证攻击有效性

### Phase 4: 消融实验
- 稀疏度梯度消融
- 遍历不同阈值
- 记录稀疏度、准确率、MIA准确率

### Phase 5: 结果可视化
- 生成IEEE标准学术图表
- 输出综合性能分析报告

<!-- 技术细节 -->
## 🔬 技术细节

### SNN 模型架构
- **神经元模型**: Parametric LIF (PLIF) 激活函数，α为可学习参数
- **编码方式**: 直接编码 (Direct Encoding)
- **时间步长**: T = 6
- **替代梯度**: ATan函数

### 隐私攻击评估
- **攻击类型**: 黑盒成员推理攻击 (Black-box MIA)
- **影子模型**: 5个同架构模型
- **攻击模型**: Logistic Regression
- **评估指标**: 攻击准确率、统计显著性
- **防御机制**: 稀疏性自然防护

### 功耗测量方法
- **测量工具**: pynvml GPU监控库
- **测量对象**: 推理能耗(mJ)、延迟(ms)
- **基准对比**: SNN vs ANN 性能对比

### GPU性能优化
- **混合精度训练**: 启用AMP加速
- **梯度累积**: 支持大batch训练
- **梯度裁剪**: 防止梯度爆炸
- **AdamW优化器**: 权重衰减分离
- **余弦退火**: 学习率平滑下降
- **Non-blocking**: 异步数据传输
- **cudnn.benchmark**: 自动选择最优算法

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
- **混合精度**: 默认启用AMP加速训练
- **批大小**: 可在train.py中调整BATCH_SIZE
- **内存清理**: 代码中已包含GPU缓存清理逻辑

### 常见问题
Q: 出现CUDA OOM错误怎么办？  
A: 减小train.py中的BATCH_SIZE，或启用梯度累积（增大ACCUMULATION_STEPS）

Q: 无法安装SpikingJelly？  
A: 尝试使用官方源: `pip install spikingjelly`

Q: 功耗测试失败？  
A: 检查NVIDIA驱动与pynvml安装，或在CPU模式下跳过功耗测试

Q: 如何只运行部分实验？  
A: 直接运行对应的Python脚本，不需要运行完整的run_all.sh

<!-- 项目交付物 -->
## 📦 最终交付物

1. **修复好的完整投稿版论文**：`FINAL_RESEARCH_PAPER.md`
2. **可一键复现的项目代码**：完整的Python脚本和配置
3. **所有实验结果表格（含统计）**：outputs/*.csv
4. **5张学术标准图表**：outputs/figures/
5. **README.md + requirements.txt**：项目说明和依赖清单

---

**项目版本**: HemoSparse v2.0  
**最后更新**: 2026年3月7日
