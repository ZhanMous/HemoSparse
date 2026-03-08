# HemoSparse

HemoSparse 是一个围绕 BloodMNIST 的研究型代码库，用于比较稀疏脉冲神经网络、非稀疏对照模型与 ANN 在准确率、隐私攻击鲁棒性与理论能效上的差异。

## 当前仓库范围

这个仓库当前包含的是研究代码和论文材料，不是一个通用 Python 包。实际可运行的核心入口如下：

- train.py：训练 SNN、DenseSNN、ANN，并生成训练摘要。
- mia_attack.py：运行成员推理攻击实验。
- p1_ablation_studies.py：执行 P1 消融实验。
- control_variable_ablation.py：执行控制变量实验。
- calculate_flops.py：计算理论计算量。
- generate_academic_figures.py、generate_public_figures.py、generate_ieee_tables.py：生成图表与表格。

## 环境要求

- Python 3.10+
- PyTorch 2.1
- torchvision 0.16
- SpikingJelly 0.0.14
- medmnist 3.0.0

推荐直接安装 requirements.txt 中锁定的版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

训练全部主模型：

```bash
python train.py
```

从某个模型开始继续训练：

```bash
python train.py DenseSNN
```

启用严格可复现模式：

```bash
python train.py SNN --deterministic
```

运行基础测试：

```bash
pytest
```

执行隐私攻击实验：

```bash
python mia_attack.py
```

## 目录说明

```text
HemoSparse/
├── config.py
├── models.py
├── train.py
├── mia_attack.py
├── calculate_flops.py
├── p1_ablation_studies.py
├── control_variable_ablation.py
├── memorization_analysis.py
├── data/
│   └── dataloader.py
├── tests/
│   └── test_smoke.py
├── outputs/
│   ├── *.csv
│   ├── *.pth
│   └── figures/
└── *.md
```

## 仓库约定

- outputs 目录默认存放运行产物。
- data/raw 目录可能包含自动下载的数据。
- 根目录 Markdown 文件主要用于论文、补充材料与投稿记录。
- 仓库中的测试以“快速形状校验”和“轻量模型一致性检查”为主，不覆盖完整训练流程。

## 已知限制

- 功耗测量依赖 NVIDIA NVML；若本机不可用，训练摘要中的 power 字段会显示为 N/A。
- 当前多个实验脚本仍然是研究脚本风格，而不是可复用库接口。
- outputs 中已有结果文件属于实验产物，不保证与当前代码完全同步。

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
