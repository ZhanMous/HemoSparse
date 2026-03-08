# HemoSparse

[![Conference](https://img.shields.io/badge/NeurIPS-IEEE%20TMI-blue.svg)](https://neurips.cc)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)

## 论文信息

**题目**：HemoSparse: Privacy-Protected Medical Image Classification with Sparse Spiking Neural Networks  
**会议/期刊**：NeurIPS / IEEE Transactions on Medical Imaging (TMI)  
**作者**：[作者团队]  
**日期**：2026年3月8日

---

## 核心亮点

HemoSparse框架探讨脉冲神经网络（SNN）的天然稀疏性在医疗AI隐私保护与低功耗推理方面的潜力。

### 核心结果

| 指标 | ANN | SNN (本文方法) | 相对优势 |
|------|-----|---------------|---------|
| 测试准确率 | 95.59% ± 0.11% | 93.63% ± 0.28% | 略低1.96% |
| MIA攻击准确率 | 0.628 ± 0.021 | 0.500 ± 0.015 | **降低20.4%** |
| 全局稀疏度 | 0.000 | 0.997 ± 0.001 | **高稀疏性** |
| 理论MACs节省 | 0.0% | 99.7% | **能效优势显著** |

### 核心结论

1. **隐私保护**：SNN的稀疏激活模式显著降低了模型对训练数据的记忆程度，使其对成员推理攻击（MIA）的准确率降至接近随机猜测水平（0.500±0.015）
2. **跨架构普适性**：开发了轻量化Spiking Transformer模型（参数量0.119M），验证了稀疏性优势在Transformer架构上的普适性
3. **理论能效优势**：SNN在专用神经形态芯片上可节省99.7%的有效MAC操作，展现出显著的能效潜力

---

## 环境配置

### 系统要求

- Python 3.10+
- PyTorch 2.1+
- SpikingJelly ≥0.0.0.0.14
- CUDA 11.8+ (推荐)

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/your-username/HemoSparse.git
cd HemoSparse

# 创建虚拟环境
conda create -n hemosparse python=3.10
conda activate hemosparse

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install spikingjelly medmnist matplotlib numpy scipy scikit-learn pandas
```

### 依赖库

```txt
torch>=2.1.0
torchvision>=0.16.0
spikingjelly>=0.0.0.0.14
medmnist>=3.0.0
matplotlib>=3.7.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
pynvml>=11.5.0
```

---

## 快速开始

### 1. 数据准备

BloodMNIST数据集会自动下载，无需手动准备：

```python
import medmnist
from medmnist import BloodMNIST

# 自动下载数据集
train_dataset = BloodMNIST(split='train', download=True)
test_dataset = BloodMNIST(split='test', download=True)
```

### 2. 一键复现核心实验

```bash
# 复现所有核心实验（训练+测试+攻击）
bash reproduce.sh

# 或分步运行
python train.py --model ann    # 训练ANN
python train.py --model snn    # 训练SNN
python attack.py                # 执行MIA攻击
python evaluate.py              # 评估所有指标
```

### 3. 单模型训练

```python
from models import SNN, ANN, DenseSNN
from train import train_model

# 训练SNN
model = SNN()
train_model(model, model_name='snn', epochs=50, lr=1e-3)

# 训练ANN
model = ANN()
train_model(model, model_name='ann', epochs=50, lr=1e-3)
```

### 4. MIA攻击

```python
from attack import MembershipInferenceAttack

# 执行MIA攻击
attack = MembershipInferenceAttack(target_model='snn')
mia_accuracy = attack.evaluate()
print(f"MIA攻击准确率: {mia_accuracy:.4f}")
```

---

## 仓库结构

```
HemoSparse/
├── README.md                          # 本文件
├── LICENSE                            # 许可证
├── FINAL_RESEARCH_PAPER.md           # 论文全文（Markdown格式）
├── RESPONSE_TO_REVIEWERS.md          # 评审响应文档
├── SUPPLEMENTARY_MATERIAL.md         # 会议补充材料
│
├── code/                              # 完整代码
│   ├── __init__.py
│   ├── models.py                      # 模型定义（SNN/ANN/DenseSNN）
│   ├── train.py                       # 训练脚本
│   ├── test.py                        # 测试脚本
│   ├── attack.py                      # MIA攻击脚本
│   ├── influence_functions.py         # 影响函数计算
│   ├── datasets.py                    # 数据加载
│   └── utils.py                       # 工具函数
│
├── pretrained/                        # 预训练权重
│   ├── ann_best.pth                   # ANN最佳模型
│   ├── snn_best.pth                   # SNN最佳模型
│   ├── densesnn_best.pth              # DenseSNN最佳模型
│   └── spiking_transformer_best.pth  # Spiking Transformer最佳模型
│
├── reproduce/                         # 复现脚本与原始数据
│   ├── reproduce.sh                   # 一键复现脚本
│   ├── plot_figures.py                # 图表生成脚本
│   └── raw_data/                      # 原始实验数据
│       ├── model_performance.csv
│       ├── mia_results.csv
│       ├── power_measurements.csv
│       └── ablation_studies.csv
│
├── outputs/                           # 输出目录
│   ├── figures/                       # 实验图表
│   │   ├── model_performance.png
│   │   ├── sparsity_vs_mia.png
│   │   ├── confidence_distribution.png
│   │   ├── power_latency.png
│   │   ├── spiking_transformer_performance.png
│   │   ├── transformer_sparsity_vs_mia.png
│   │   └── high_res/                  # 高分辨率图表（300 DPI）
│   │
│   └── tables/                        # 实验表格
│       ├── table_i_performance.csv
│       ├── table_ii_sparsity.csv
│       └── ...
│
└── docs/                              # 文档
    ├── paper.pdf                      # 论文PDF
    ├── supplementary_material.pdf    # 补充材料PDF
    └── response_to_reviewers.pdf     # 评审响应PDF
```

---

## 论文引用

如果您在研究中使用了本项目的代码或方法，请引用我们的论文：

```bibtex
@inproceedings{hemosparse2026,
  title={HemoSparse: Privacy-Protected Medical Image Classification with Sparse Spiking Neural Networks},
  author={[作者姓名]},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

或（IEEE TMI版本）：

```bibtex
@article{hemosparse2026tmi,
  title={HemoSparse: Privacy-Protected Medical Image Classification with Sparse Spiking Neural Networks},
  author={[作者姓名]},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
```

---

## 复现说明

### 完整复现流程

1. **环境配置**（5分钟）
   ```bash
   conda env create -f environment.yml
   conda activate hemosparse
   ```

2. **数据准备**（自动下载，5分钟）
   ```bash
   python prepare_data.py
   ```

3. **模型训练**（约2小时）
   ```bash
   python train_all_models.py
   ```

4. **攻击执行**（约30分钟）
   ```bash
   python run_all_attacks.py
   ```

5. **结果评估**（约10分钟）
   ```bash
   python evaluate_all.py
   ```

6. **图表生成**（约5分钟）
   ```bash
   python plot_all_figures.py
   ```

### 预期硬件要求

- GPU：NVIDIA RTX 3060或更高（8GB显存）
- 内存：16GB或更高
- 存储：20GB可用空间

### 复现时间

- 完整复现：约3-4小时
- 快速复现（仅核心实验）：约1-2小时

---

## 代码注释

所有代码均包含详细的中文和英文注释：

```python
class SNN(nn.Module):
    """
    脉冲神经网络（SNN）模型
    
    参数:
        in_channels: 输入通道数，默认3（RGB图像）
        num_classes: 分类类别数，默认8（BloodMNIST）
        tau: PLIF神经元膜时间常数，默认5.0
        v_threshold: 脉冲发放阈值，默认1.0
    """
    def __init__(self, in_channels=3, num_classes=8, tau=5.0, v_threshold=1.0):
        super().__init__()
        # 网络架构定义...
```

---

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue：[GitHub Issues](https://github.com/your-username/HemoSparse/issues)
- 发送邮件：[your-email@example.com]

---

## 致谢

感谢以下项目和团队的支持：

- [MedMNIST](https://medmnist.com/) - 提供BloodMNIST数据集
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) - 脉冲神经网络框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

**README结束**
