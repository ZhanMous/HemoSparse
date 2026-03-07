# -*- coding: utf-8 -*-
"""
HemoSparse 全局配置文件
针对 RTX 4070 笔记本 (8GB GDDR6) 专属优化
"""

import os
import random
import torch
import numpy as np

# ============================================================
# 1. 随机种子（保证实验100%可复现）
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# 2. 设备与硬件检测
# ============================================================
def get_device():
    """自动检测CUDA可用性，默认cuda:0"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[HemoSparse] 使用GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return device
    else:
        print("[HemoSparse] 警告: CUDA不可用，使用CPU（训练将极慢）")
        return torch.device('cpu')

DEVICE = get_device()

# ============================================================
# 3. 数据集配置
# ============================================================
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
DATA_FLAG = 'bloodmnist'
NUM_CLASSES = 8
IMG_SIZE = 28
IN_CHANNELS = 3  # BloodMNIST 为 RGB 3通道

# ============================================================
# 4. 训练超参数（适配 4070 笔记本）
# ============================================================
# 时间步
DEFAULT_T = 20       # 默认时间步
MAX_T = 50           # 最大时间步

# 批次大小（显存自适应）
DEFAULT_BATCH_SIZE = 16
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 32

# 训练轮次
NUM_EPOCHS = 50

# 学习率
SNN_LR = 0.05        # SNN 使用 SGD
ANN_LR = 1e-3        # ANN 使用 Adam
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9       # SGD momentum

# LIF 神经元参数
LIF_TAU = 2.0
LIF_V_THRESHOLD = 1.0

# 混合精度训练
USE_AMP = True

# 数据加载
NUM_WORKERS = 4
PIN_MEMORY = True

# ============================================================
# 5. 显存自适应调整
# ============================================================
def get_adaptive_batch_size(default=DEFAULT_BATCH_SIZE):
    """根据 GPU 剩余显存自适应调整 batch_size"""
    if not torch.cuda.is_available():
        return default

    torch.cuda.empty_cache()
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)
    free_mem = total_mem - reserved_mem

    if free_mem < 3.0:  # 剩余显存不足3GB
        batch_size = MIN_BATCH_SIZE
        print(f"[HemoSparse] 显存紧张 ({free_mem:.1f}GB剩余)，batch_size降至 {batch_size}")
    elif free_mem > 6.0:  # 显存充足
        batch_size = MAX_BATCH_SIZE
        print(f"[HemoSparse] 显存充足 ({free_mem:.1f}GB剩余)，batch_size升至 {batch_size}")
    else:
        batch_size = default
        print(f"[HemoSparse] 显存正常 ({free_mem:.1f}GB剩余)，batch_size = {batch_size}")

    return batch_size

# ============================================================
# 6. 输出路径管理
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

def ensure_dirs():
    """创建所有输出目录"""
    for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

ensure_dirs()

# ============================================================
# 7. 功耗测试配置
# ============================================================
POWER_SAMPLE_INTERVAL = 0.01    # pynvml 采样间隔 (秒)
POWER_WARMUP_ITERS = 10         # GPU 预热推理次数
POWER_TEST_ITERS = 100          # 功耗测试推理次数

# ============================================================
# 8. 隐私攻击配置
# ============================================================
MIA_SHADOW_EPOCHS = 10          # 影子模型训练轮次
MEA_SUBSTITUTE_EPOCHS = 10      # 替代模型训练轮次
MEA_QUERY_BUDGET = 5000         # 模型提取查询预算

# ============================================================
# 9. 可视化配置
# ============================================================
FIG_DPI = 300                   # 图表分辨率 (适配打印/汇报)
FIG_FORMAT = 'png'
FONT_SIZE = 12

# 颜色方案（三组对照）
COLOR_SNN = '#2196F3'       # 蓝色 - 标准SNN
COLOR_DENSE = '#FF9800'     # 橙色 - 稠密SNN
COLOR_ANN = '#4CAF50'       # 绿色 - ANN
COLORS = [COLOR_SNN, COLOR_DENSE, COLOR_ANN]
LABELS = ['标准SNN (A)', '稠密SNN (B)', 'ANN (C)']

print(f"[HemoSparse] 配置加载完成 | 设备: {DEVICE} | T={DEFAULT_T} | Epochs={NUM_EPOCHS}")
