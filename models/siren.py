"""
siren.py — SIREN (Sinusoidal Representation Networks) 模块。

SIREN 使用周期性正弦激活函数替代 ReLU，能够更好地捕捉高频空间梯度。
特别适合几何坐标编码，在振动分析中可精确表示共振模态的空间变化。

参考:
    Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
    NeurIPS 2020

与标准 MLP 的关键区别:
    - 使用 sin(w0 * x) 激活函数
    - 特殊的权重初始化策略确保梯度传播稳定
    - w0 控制网络的频率带宽 (越大 → 更能捕捉高频细节)
"""

import torch
import torch.nn as nn
import math


class Sine(nn.Module):
    """
    正弦激活函数: y = sin(w0 * x)

    参数:
        w0: 频率缩放因子。默认 30.0 (SIREN 论文推荐值)。
            更大的 w0 允许网络捕捉更高频的信号。
            对于振动分析中陡峭的共振峰，较大的 w0 有优势。
    """

    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """
    单个 SIREN 全连接层 = Linear + Sine 激活。

    自动应用适合 sine 激活的权重初始化。
    """

    def __init__(self, in_features: int, out_features: int,
                 w0: float = 30.0, is_first: bool = False):
        """
        参数:
            in_features:  输入维度
            out_features: 输出维度
            w0:           sine 频率因子
            is_first:     是否为 SIREN 网络的第一层
                          (第一层需要不同的初始化范围)
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.sine = Sine(w0)
        self.is_first = is_first
        self._init_weights()

    def _init_weights(self):
        """
        SIREN 特有的权重初始化。

        第一层:  w ~ U(-1/in_features, 1/in_features)
               确保 sin(w*x) 在输入空间每个方向上约有一个周期
        后续层: w ~ U(-sqrt(6/in_features)/w0, sqrt(6/in_features)/w0)
               保持激活值的标准差与深度无关
        """
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.sine.w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return self.sine(self.linear(x))


class SirenMLP(nn.Module):
    """
    多层 SIREN 网络。

    用于将空间坐标 (x, y, z) 编码为高频空间特征。
    相比 ReLU MLP，SIREN 能更好地保留坐标的精细几何信息。

    使用方式:
        trunk = SirenMLP(in_dim=3, hidden_dim=256, out_dim=256, n_layers=4)
        spatial_features = trunk(points)  # (B, N, out_dim)
    """

    def __init__(self, in_dim: int = 3, hidden_dim: int = 256,
                 out_dim: int = 256, n_layers: int = 4,
                 w0: float = 30.0, w0_first: float = 30.0):
        """
        参数:
            in_dim:     输入坐标维度 (2 或 3)
            hidden_dim: 隐藏层宽度
            out_dim:    输出特征维度
            n_layers:   总层数 (含输入和输出)
            w0:         正弦激活频率因子 (通用)
            w0_first:   第一层的频率因子 (可独立设置)
        """
        super().__init__()

        layers = []
        for i in range(n_layers):
            is_first = (i == 0)
            current_w0 = w0_first if is_first else w0
            layer_in = in_dim if i == 0 else hidden_dim
            layer_out = out_dim if i == n_layers - 1 else hidden_dim

            layers.append(SirenLayer(layer_in, layer_out, current_w0, is_first))
            # 所有层 (含最后一层) 均使用 Sine 激活, 这是 SIREN 的标准设计

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
