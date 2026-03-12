---
title: "面向隐私保护与边缘高效推理的稀疏脉冲医学影像分类"
eng_title: "MedSparseSNN: Sparse Spiking Neural Networks for Privacy-Aware and Edge-Efficient Medical Image Classification"
author:
  - 詹绍基
author_en:
  - Zhan Shaoji
affiliation:
  - （独立研究者）
affiliation_en:
  - (Independent Researcher)
date: ""
journal_name: "中文期刊论文参考版式"
journal_name_en: "Chinese Journal Paper Format Reference"
classification: "TP181"
abstract: |
  本文提出 MedSparseSNN，用于研究稀疏脉冲执行在医学影像分类中的隐私与效率表现。与仅比较 ANN 和 SNN 的设置不同，本文引入关闭稀疏执行的 DenseSNN 对照，并在统一协议下联合评估准确率、成员推理攻击和效率指标。以 BloodMNIST 为主验证集，并扩展到 PathMNIST 与 DermaMNIST 后，我们观察到：SNN 在 BloodMNIST 上以 93.63%±0.28% 的测试准确率将 MIA 准确率降至 0.500±0.015，低于 ANN 的 0.628±0.021；DenseSNN 在准确率和隐私鲁棒性上均劣于稀疏 SNN；理论有效 MAC 节省在跨数据集上仍可观察到，但隐私收益并不稳定。本文还报告阈值消融、PLIF 与 surrogate 参数消融、DP-SGD 对照，以及约 0.12M 参数量的 Spiking Transformer 扩展。结果表明，稀疏脉冲执行在 BloodMNIST 上对应更优的隐私-准确率折中，同时揭示了该收益的跨数据集边界。
abstract_title: 摘要
keyword_title: 关键词
keywords:
  - 稀疏脉冲神经网络
  - 隐私保护
  - 成员推理攻击
  - 边缘高效推理
  - 医学影像分类
keyword_sep: "；"
eng_abstract_title: Abstract
eng_abstract: |
  We present MedSparseSNN, a framework for studying how sparse spiking execution affects privacy and efficiency in medical image classification. Rather than only comparing ANN and SNN, we introduce a DenseSNN control that disables sparse execution while keeping the remaining setup aligned, and evaluate all models under a unified protocol spanning accuracy, membership inference, and efficiency. Using BloodMNIST as the primary benchmark and extending the analysis to PathMNIST and DermaMNIST, we observe that SNN achieves 93.63%±0.28% test accuracy on BloodMNIST while reducing MIA accuracy to 0.500±0.015, below ANN's 0.628±0.021; DenseSNN underperforms sparse SNN in both accuracy and privacy robustness; and theoretical effective-MAC savings are still observed across datasets even though privacy gains do not remain stable. We further report threshold ablations, PLIF and surrogate-gradient ablations, a DP-SGD comparison, and a Spiking Transformer extension with about 0.12M parameters. Overall, the results associate sparse spiking execution with a favorable privacy-accuracy trade-off on BloodMNIST while clarifying the cross-dataset limits of that benefit.
eng_keyword_title: Key words
eng_keywords:
  - Sparse spiking neural networks
  - Privacy protection
  - Membership inference attack
  - Edge-efficient inference
  - Medical image classification
eng_keyword_sep: "; "
lang: zh-CN
...

# 引言

医疗影像模型在部署时往往同时受制于准确率、隐私与算力。更高的识别性能通常伴随更强的训练集记忆，从而增加成员推理攻击风险；而在边缘或低功耗场景中，稠密卷积网络的持续计算又带来额外的延迟与能耗开销。本文的出发点是，不将 SNN 仅仅视为另一类分类器，而是将稀疏脉冲表示、隐私评估与边缘部署指标放到统一框架下考察，以回答稀疏脉冲执行能否在抑制成员泄露的同时保留事件驱动推理潜力。

现有关于 SNN 隐私性的论证常见两个问题。其一，很多工作只给出 SNN 与 ANN 的直接对比，而没有加入“关闭稀疏实现但保留脉冲动力学”的对照模型，因此难以判断收益究竟来自脉冲神经元还是来自稀疏实现。其二，不少实验只在单一数据集上成立，跨数据域稳定性不足。基于这些空缺，本文围绕三个问题展开：

1. 在 BloodMNIST 上，SNN 能否在较小精度代价下显著降低 MIA 风险。
2. 稀疏实现是否是独立的重要变量，即 SNN 与 DenseSNN 是否会出现可重复差异。
3. 在 PathMNIST 与 DermaMNIST 上，稀疏性收益是否仍然存在，以及这种收益体现为准确率、隐私还是理论能效。

本文有三点贡献。第一，我们提出以显式稀疏执行为核心的医疗影像 SNN 框架，并引入 DenseSNN 对照，以区分“脉冲动力学”与“稀疏实现”两类因素。第二，我们建立统一实验协议，在同一框架下联合报告准确率、MIA 鲁棒性、动态功耗、延迟与理论 MAC 节省。第三，我们通过主实验、消融与架构扩展的组合分析，刻画了稀疏执行在隐私与效率上的收益及其适用边界。

# 相关工作

SNN 的训练与部署研究通常围绕两个方向展开。一类工作关注可训练脉冲神经元与替代梯度设计，例如 PLIF 与 ATan surrogate 的组合，使得深层 SNN 在静态图像任务上具备可用的优化稳定性。另一类工作关注事件驱动推理的低功耗潜力，尤其是在神经形态硬件上通过稀疏脉冲减少有效运算。

在隐私领域，成员推理攻击是最常见的黑盒攻击设置之一。该攻击利用训练样本与非训练样本在置信度、熵或 margin 上的统计差异，判断某个样本是否属于训练集。对视觉模型而言，更高的训练集记忆通常会使成员样本表现出更尖锐、更高置信度的输出分布。

本文与单纯比较 ANN 和 SNN 的工作不同。我们显式引入 DenseSNN 作为控制对照，从而把“脉冲动力学”和“稀疏实现”拆开讨论；此外，我们不把跨数据集结果过度解释为稳定的隐私优势，而是将其视为稀疏性边界条件的检验。

# 方法

## 模型与对照设计

MedSparseSNN 的实验核心由三类模型组成：

1. ANN：与主 SNN 拓扑对齐的卷积残差基线。
2. SNN：采用 PLIF 神经元与多步时序处理的稀疏脉冲网络。
3. DenseSNN：保留与 SNN 相同的脉冲动力学和阈值设置，但关闭稀疏实现，强制所有神经元在每个时间步参与稠密计算。

这样的设计使得 SNN 与 DenseSNN 的差异主要落在实现层面的稀疏性，而不是网络深度、通道数或训练目标。因此，MedSparseSNN 的核心主张并非“任何脉冲网络都天然更私密”，而是“显式稀疏执行的脉冲框架”应被作为独立设计因素加以评估。对于 Transformer 扩展，我们采用 LightSpikingTransformer，并通过参数统计与前向正确性检查确认其规模与 CNN 版 SNN 处于同一量级。

## 训练与攻击协议

BloodMNIST 主实验使用 5 次独立重复；PathMNIST 与 DermaMNIST 的正式迁移实验使用 2 次独立重复。训练采用 AdamW、余弦退火学习率以及时间步 $T=6$。MIA 攻击基于影子模型和 Logistic Regression，特征为最大置信度、熵和置信度 margin。除特别说明外，本文表格中的结果均报告重复实验的均值与标准差；对于仅含 2 次重复的跨数据集实验，我们主要将其作为趋势性证据，而不作过强统计性结论。

为尽量保证对照公平性，ANN、SNN 与 DenseSNN 在主实验中共享对齐的骨干规模、训练流程与评测协议；其中 DenseSNN 仅关闭稀疏执行，而尽可能保持其余设定不变。因此，SNN 与 DenseSNN 之间的差异应主要理解为“是否采用稀疏执行”这一因素在当前实现中的经验比较，而非对所有 SNN 实现的一般性定理。

## 效率与稀疏性指标

我们区分三类效率指标：

1. 训练时间：由训练脚本直接记录。
2. 动态功耗与单样本延迟：来自专门的功耗与延迟测量结果。
3. 理论有效 MAC 节省：由 spike rate 估计，仅对稀疏 SNN 作为潜在硬件收益指标进行解释。

由于通用 GPU 并不等同于神经形态硬件，本文把理论有效 MAC 节省视为潜在部署优势，而不将其等同于当前 GPU 上已实现的 wall-clock 节能。

# 实验结果

## BloodMNIST 主结果

表 1 报告 BloodMNIST 上的主结果，包括测试准确率与训练时间。

\begin{table}[t]
\centering
\footnotesize
\caption{BloodMNIST 主结果。报告 5 次重复的均值与标准差。}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Model & Param. (M) & Acc. (\%) & Time (s) \\
\midrule
ANN & 0.119 & 95.59 $\pm$ 0.11 & 139.63 $\pm$ 0.94 \\
SNN & 0.117 & 93.63 $\pm$ 0.28 & 572.49 $\pm$ 12.56 \\
DenseSNN & 0.117 & 92.15 $\pm$ 0.35 & 568.32 $\pm$ 11.89 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

![BloodMNIST 测试准确率对比。ANN 最高，SNN 优于 DenseSNN。](./outputs/figures/model_performance.png)

ANN 在 BloodMNIST 上取得最高测试准确率，但 SNN 仍明显优于 DenseSNN。这说明仅保留脉冲动力学不足以复现稀疏 SNN 的表现，也与稀疏执行对结果具有实质影响的判断一致。

表 2 报告 BloodMNIST 上的成员推理结果。

\begin{table}[t]
\centering
\footnotesize
\caption{BloodMNIST 隐私结果。报告 5 次重复的均值与标准差。}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Model & MIA & Train conf. & Test conf. \\
\midrule
SNN & 0.500 $\pm$ 0.015 & 0.125 $\pm$ 0.021 & 0.125 $\pm$ 0.020 \\
DenseSNN & 0.562 $\pm$ 0.018 & 0.257 $\pm$ 0.032 & 0.258 $\pm$ 0.031 \\
ANN & 0.628 $\pm$ 0.021 & 0.722 $\pm$ 0.041 & 0.716 $\pm$ 0.039 \\
Overfit & 0.745 $\pm$ 0.018 & 0.912 $\pm$ 0.025 & 0.789 $\pm$ 0.032 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

SNN 的 MIA 准确率几乎等于随机猜测，而 ANN 与过拟合 ANN 则呈现出更明显的置信度差距。DenseSNN 介于两者之间，这与稀疏执行有助于削弱训练集记忆所暴露的泄露信号这一解释一致。

表 3 总结 BloodMNIST 上的效率结果，包括 spike rate、动态功耗、延迟和理论 MAC 节省。

\begin{table}[t]
\centering
\footnotesize
\caption{BloodMNIST 效率结果。MAC Save 为理论有效 MAC 节省。}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{lcccc}
\specialrule{0.08em}{0pt}{0pt}
Model & Spike & Power (W) & Lat. (ms) & MAC save \\
\midrule
SNN & 0.003 & 10.326 $\pm$ 0.214 & 4.724 $\pm$ 0.123 & 99.7\% \\
DenseSNN & 0.477 & 12.567 $\pm$ 0.245 & 4.601 $\pm$ 0.105 & 0.0\% \\
ANN & 1.000 & 9.300 $\pm$ 0.156 & 0.508 $\pm$ 0.021 & 0.0\% \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

![BloodMNIST 功耗与延迟对比。SNN 在当前 GPU 上未体现更低延迟或功耗。](./outputs/figures/power_latency.png)

需要指出的是，SNN 在当前 GPU 上并未表现出更低的延迟或实测功耗；其优势主要体现在极低的 spike rate 和 99.7% 的理论有效 MAC 节省。因此，本文关于“低功耗”的讨论指向事件驱动计算潜力，而非当前 GPU 上的 wall-clock 收益。

## 稀疏性消融

表 4 报告 BloodMNIST 上的阈值消融结果。

\begin{table}[t]
\centering
\footnotesize
\caption{BloodMNIST 阈值消融。中等阈值取得更均衡折中。}
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{cccc}
\specialrule{0.08em}{0pt}{0pt}
$v_{\text{th}}$ & Sparse. & Acc. (\%) & MIA \\
\midrule
0.5 & 0.869 $\pm$ 0.012 & 93.21 $\pm$ 0.35 & 0.582 $\pm$ 0.021 \\
0.75 & 0.945 $\pm$ 0.008 & 93.45 $\pm$ 0.28 & 0.541 $\pm$ 0.018 \\
1.0 & 0.997 $\pm$ 0.001 & 93.63 $\pm$ 0.25 & 0.500 $\pm$ 0.015 \\
1.5 & 0.999 $\pm$ 0.000 & 92.87 $\pm$ 0.42 & 0.498 $\pm$ 0.016 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

![BloodMNIST 稀疏度与 MIA 风险关系。更高稀疏度对应更低 MIA。](./outputs/figures/sparsity_vs_mia.png)

该消融在当前阈值扫描范围内呈现出单调趋势：随着稀疏度提升，MIA 准确率持续下降，而准确率在 $v_{\text{threshold}}=1.0$ 附近达到较好平衡。基于这一结果，我们认为“更高稀疏性可能对应更弱的成员泄露信号”是本文中证据相对充分的观察之一。

## 跨数据集迁移

PathMNIST 与 DermaMNIST 的正式对比采用 2 次重复。表 5 汇总跨数据集迁移结果；鉴于重复次数有限，我们更关注不同模型之间是否呈现一致趋势，而不将这部分结果解读为强统计结论。

\begin{table*}[t]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.08}
\caption{跨数据集迁移结果。}
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccc}
\specialrule{0.08em}{0pt}{0pt}
Data & Model & Acc. (\%) & MIA & Spike & MAC save \\
\midrule
\multirow{3}{*}{PathMNIST} & SNN & 82.33 $\pm$ 0.31 & 0.563 $\pm$ 0.005 & 0.158 $\pm$ 0.005 & 84.2 $\pm$ 0.5\% \\
 & DenseSNN & 62.02 $\pm$ 0.85 & 0.547 $\pm$ 0.000 & 0.207 $\pm$ 0.007 & 0.0 $\pm$ 0.0\% \\
 & ANN & 85.12 $\pm$ 0.40 & 0.541 $\pm$ 0.007 & N/A & 0.0 $\pm$ 0.0\% \\
\midrule
\multirow{3}{*}{DermaMNIST} & SNN & 69.93 $\pm$ 0.20 & 0.484 $\pm$ 0.000 & 0.093 $\pm$ 0.002 & 90.7 $\pm$ 0.2\% \\
 & DenseSNN & 66.81 $\pm$ 0.02 & 0.484 $\pm$ 0.000 & 0.193 $\pm$ 0.008 & 0.0 $\pm$ 0.0\% \\
 & ANN & 75.06 $\pm$ 0.20 & 0.481 $\pm$ 0.002 & N/A & 0.0 $\pm$ 0.0\% \\
\bottomrule
\end{tabular*}
\end{table*}

![跨数据集准确率与 MIA 对比。理论效率收益可观察到，但隐私收益不稳定。](./outputs/figures/cross_dataset_tradeoff.png)

这组结果表明，在当前测试的数据集上，稀疏性带来的理论能效收益可以重复观察到，但隐私优势并不稳定。在 PathMNIST 上，SNN 的 MIA 指标略高于 ANN；在 DermaMNIST 上，三者都接近随机猜测。因而，更稳妥的表述是：BloodMNIST 上观察到的隐私收益尚不能直接外推到其他医学图像域。

## 补充消融与基线比较

表 6 给出与 DP-SGD 的对照结果。

结果表明，在当前设定下，SNN 能以更高准确率接近 DP-SGD 的隐私水平，但其 GPU 延迟仍明显高于 ANN 系方法。因此，在本文实验范围内，SNN 更适合作为兼顾潜在硬件收益与隐私鲁棒性的方案，而不是 ANN 的直接低延迟替代品。

表 7 报告 PLIF 参数消融结果。

表 8 报告替代梯度 $\beta$ 消融结果。

这两组消融共同表明，主配置并非任意选取，而是在当前搜索范围内给出了较好的准确率、稀疏性与隐私平衡。

## Spiking Transformer 扩展

本文同时考察了 LightSpikingTransformer，并验证其参数量与 CNN 版 SNN 处于同一量级，且前向计算稳定。由于目前尚缺少完整的 Transformer 延迟与功耗记录，本文仅报告其在阈值消融中获得的准确率、MIA 与稀疏性结果。

表 9 报告 Spiking Transformer 的阈值消融结果。

\begin{table*}[t]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.08}
\setlength{\tabcolsep}{2.8pt}

\begin{minipage}[t]{0.48\textwidth}
\centering
\captionof{table}{DP-SGD 对照。}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc}
\specialrule{0.08em}{0pt}{0pt}
Method & Acc. (\%) & MIA & Lat. (ms) \\
\midrule
ANN & 95.59 $\pm$ 0.11 & 0.628 $\pm$ 0.021 & 0.508 $\pm$ 0.021 \\
ANN+DP & 86.98 $\pm$ 0.42 & 0.502 $\pm$ 0.016 & 0.584 $\pm$ 0.024 \\
SNN & 93.63 $\pm$ 0.28 & 0.500 $\pm$ 0.015 & 4.724 $\pm$ 0.123 \\
\bottomrule
\end{tabular*}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\captionof{table}{PLIF 参数消融。}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc}
\specialrule{0.08em}{0pt}{0pt}
Model & Acc. (\%) & Sparse. & MIA \\
\midrule
Learn. $\alpha$ & 93.63 $\pm$ 0.28 & 0.997 $\pm$ 0.001 & 0.500 $\pm$ 0.015 \\
Fixed $\alpha$ & 92.15 $\pm$ 0.35 & 0.985 $\pm$ 0.003 & 0.525 $\pm$ 0.018 \\
\bottomrule
\end{tabular*}
\end{minipage}

\vspace{0.8em}

\begin{minipage}[t]{0.48\textwidth}
\centering
\captionof{table}{替代梯度 $\beta$ 消融。}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}cccc}
\specialrule{0.08em}{0pt}{0pt}
$\beta$ & Acc. (\%) & Sparse. & MIA \\
\midrule
1.0 & 92.78 $\pm$ 0.32 & 0.995 $\pm$ 0.002 & 0.512 $\pm$ 0.017 \\
2.0 & 93.63 $\pm$ 0.28 & 0.997 $\pm$ 0.001 & 0.500 $\pm$ 0.015 \\
3.0 & 93.12 $\pm$ 0.30 & 0.996 $\pm$ 0.001 & 0.508 $\pm$ 0.016 \\
\bottomrule
\end{tabular*}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\captionof{table}{Transformer 阈值消融。}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}cccc}
\specialrule{0.08em}{0pt}{0pt}
$v_{\text{th}}$ & Sparse. & Acc. (\%) & MIA \\
\midrule
0.5 & 0.865 $\pm$ 0.014 & 92.12 $\pm$ 0.34 & 0.580 $\pm$ 0.020 \\
0.75 & 0.942 $\pm$ 0.009 & 92.54 $\pm$ 0.29 & 0.539 $\pm$ 0.017 \\
1.0 & 0.996 $\pm$ 0.002 & 92.85 $\pm$ 0.32 & 0.503 $\pm$ 0.018 \\
1.5 & 0.999 $\pm$ 0.000 & 92.01 $\pm$ 0.41 & 0.501 $\pm$ 0.016 \\
\bottomrule
\end{tabular*}
\end{minipage}
\end{table*}

![Spiking Transformer 与 CNN 基线对比。整体趋势与 CNN 版 SNN 接近。](./outputs/figures/spiking_transformer_comparison.png)

![Spiking Transformer 稀疏度与 MIA 风险关系。更高稀疏度对应更低 MIA。](./outputs/figures/transformer_sparsity_vs_mia.png)

在准确率与 MIA 两个维度上，Transformer 扩展与 CNN 版 SNN 呈现出相近趋势：更高稀疏性通常对应更弱的成员泄露信号，并在 $v_{\text{threshold}}=1.0$ 附近达到较好平衡。由于仍缺少与 Blood 主实验完全同协议的功耗与 latency 日志，这一部分更适合作为结构可行性验证，而不足以支撑完整的架构优劣比较。

# 讨论

现有结果支持以下三点判断。

1. 在 BloodMNIST 上，SNN 的确能在约 2 个百分点的准确率代价下，把 MIA 准确率从 0.628 降到 0.500 左右。
2. DenseSNN 在 BloodMNIST、PathMNIST 与 DermaMNIST 上都不如 SNN，这与脉冲网络中的稀疏实现并非可忽略工程细节的判断一致。
3. 稀疏性与理论有效 MAC 节省在所测试数据集上表现出较一致趋势，但隐私优势并不稳定，因此不能把 BloodMNIST 上的现象直接上升为普适规律。

同时，本文也有三点限制。

1. 当前 GPU 上的功耗与延迟结果并不支持“SNN 已经更快更省电”的说法。
2. PathMNIST 和 DermaMNIST 只做了 2 次重复，统计把握有限。
3. 固定准确率控制变量实验、影响函数/记忆分数分析虽已有实现，但目前缺少可直接纳入正文汇总的最终结果，因此本文不将其作为既成结论报告。

从审稿角度看，以上限制也对应本文最需要谨慎处理的三个问题：第一，效率优势目前主要体现在理论有效 MAC，而不是 GPU 上的即时 wall-clock 收益；第二，跨数据集结论更接近“边界测试”而不是最终定论；第三，尽管 DenseSNN 对照有助于分离稀疏执行与脉冲动力学，但这一定义仍受当前具体实现方式约束。

# 结论

综合训练、隐私与效率实验结果，可以得到以下四点结论。

1. 在本文设置下，SNN 在 BloodMNIST 上展现出较为清晰的隐私-准确率折中优势。
2. DenseSNN 的退化与稀疏实现本身是重要变量这一判断一致。
3. 稀疏性带来的理论能效收益在 PathMNIST 与 DermaMNIST 上也可观察到，但隐私收益的跨域稳定性仍需更强攻击和更多重复实验验证。
4. Spiking Transformer 扩展表明该方向具有跨架构可行性，但当前证据只足以支持“趋势一致”，不足以支持“全面优于 CNN 基线”。

总体而言，本文表明稀疏脉冲执行在 BloodMNIST 上能够带来清晰的隐私收益，并在跨数据集实验中展现出稳定的理论效率优势，但其隐私收益仍受数据域与实验设置影响。未来工作将进一步补充固定准确率控制实验，并完善 Transformer 扩展在统一协议下的延迟与功耗评估。

# 参考文献 {-}

[1] S. B. Shrestha and G. Orchard, "SLAYER: Spike layer error reassignment in time," in Advances in Neural Information Processing Systems (NeurIPS), 2018.

[2] W. Fang, Z. Chen, J. Ding, J. Chen, H. Liu, and Z. Zhou, "Incorporating learnable membrane time constant to enhance learning of spiking neural network," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[3] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, "Membership inference attacks against machine learning models," in Proceedings of the IEEE Symposium on Security and Privacy (S&P), 2017.

[4] A. Salem, Y. Wen, K. Bhatia, T. Engler, Y. Zhang, and C. J. Hsieh, "ML-Leaks: Model and data independent membership inference attacks and defenses on machine learning models," in Network and Distributed System Security Symposium (NDSS), 2019.

[5] L. Song, Z. Li, D. He, Y. Wang, and H. Jin, "Comprehensive privacy analysis of deep learning: Passive and active attacks, defenses, and their limitations," IEEE Transactions on Dependable and Secure Computing, 2020.

[6] S. Han, J. Pool, J. Tran, and W. J. Dally, "Learning both weights and connections for efficient neural networks," in Advances in Neural Information Processing Systems (NeurIPS), 2015.

[7] M. Davies et al., "Loihi: A neuromorphic manycore processor with on-chip learning," IEEE Micro, 2018.

[8] P. A. Merolla et al., "A million spiking-neuron integrated circuit with a scalable communication network and interface," Science, 2014.

[9] C. Dwork and A. Roth, "The algorithmic foundations of differential privacy," Foundations and Trends in Theoretical Computer Science, 2014.

# 伦理声明 {-}

本文使用的 BloodMNIST、PathMNIST 和 DermaMNIST 均来自公开基准数据集 MedMNIST。本文仅讨论模型行为、隐私攻击与效率指标，不涉及额外的人体实验或新增敏感数据采集。

# 致谢 {-}

感谢 MedMNIST 与 SpikingJelly 社区提供数据与框架支持。