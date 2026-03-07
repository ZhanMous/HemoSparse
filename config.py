# -*- coding: utf-8 -*-
"""
HemoSparse 全局配置文件
针对现代GPU架构 (8GB GDDR6) 专属优化
包含混合精度(AMP)、自适应Batch Size调整与GPU功耗实测逻辑
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')  # 非GUI后端，适合服务器环境

# ===========================================
# 🧪 实验配置
# ===========================================
SEED = 42  # 随机种子
DATA_FLAG = 'bloodmnist'  # 数据集标识
PROJECT_NAME = 'HemoSparse'  # 项目名称
T = 6  # 时间步长
EPOCHS = 50  # 训练轮数
INIT_LR = 1e-3  # 初始学习率
BATCH_SIZE = 64  # 默认批次大小
WEIGHT_DECAY = 1e-4  # 权重衰减
STEP_SIZE = 20  # 学习率衰减步长
GAMMA = 0.1  # 学习率衰减系数
NUM_CLASSES = 8
IMG_SIZE = 28
IN_CHANNELS = 3  # BloodMNIST 为 RGB 3通道
MAX_T = 20  # 最大时间步
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 32
LIF_TAU = 2.0
LIF_V_THRESHOLD = 1.0

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 主设备
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[{PROJECT_NAME}] 使用GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print(f"[{PROJECT_NAME}] 警告: CUDA不可用，使用CPU（训练将极慢）")

# ============================================================
# 3. 数据集配置
# ============================================================
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
DATA_FLAG = 'bloodmnist'
NUM_CLASSES = 8
IMG_SIZE = 28
IN_CHANNELS = 3  # BloodMNIST 为 RGB 3通道

# ===========================================
# 📈 可视化配置
# ===========================================
FIG_SIZE = (10, 6)  # 图表默认尺寸
FIG_DPI = 300  # 图表分辨率
COLOR_SNN = '#2196F3'       # 蓝色 - 标准SNN
COLOR_DENSE = '#FF9800'     # 橙色 - 稠密SNN
COLOR_ANN = '#4CAF50'       # 绿色 - ANN
COLORS = [COLOR_SNN, COLOR_DENSE, COLOR_ANN]
LABELS = ['标准SNN (A)', '稠密SNN (B)', 'ANN (C)']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # 标记样式

# ===========================================
# ⚡ 性能优化配置
# ===========================================
USE_AMP = True  # 是否启用混合精度训练
PIN_MEMORY = True  # DataLoader是否启用pin_memory
NUM_WORKERS = 4  # DataLoader进程数
GRAD_CLIP_NORM = 1.0  # 梯度裁剪阈值


# ===========================================
# 🔋 功耗测量配置
# ===========================================
MEASURE_POWER = True  # 是否测量功耗 (需要pynvml支持)
POWER_DEVICE_ID = 0  # GPU设备ID