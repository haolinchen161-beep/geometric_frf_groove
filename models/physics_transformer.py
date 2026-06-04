"""
physics_transformer.py — 物理偏置自注意力 Transformer。

核心创新: 在标准多头注意力的 logits 上添加三种物理偏置:
  1. 空间衰减偏置: -α·log(1 + d_ij / L_char)
     源于 2D 弹性静力学 Green 函数 ~ log(r), 近处节点互相影响更强

  2. 边界耦合偏置: f_BC(is_fixed_i, is_fixed_j, logK_i, logK_j)
     共享弹簧约束的节点间增强注意力 — 边界条件通过结构传播

  3. 材料均匀性偏置: -β·|wave_speed_i - wave_speed_j|
     相同材料 → 相同波速 → 相同动力学状态 → 更强的振动耦合

物理依据:
  结构动力学是非局部问题 — 节点 i 的位移取决于 ALL 节点的力,
  通过柔度矩阵传播。自注意力允许节点间建立隐式的柔度关系。
  物理偏置将注意力从 epoch 0 就约束到力学合理模式,
  在少样本场景下大幅降低有效假设空间。

Transformer 架构细节:
  - Pre-LayerNorm: 先 norm 后 attention/FFN (训练更稳定)
  - 逐样本独立注意力: 避免跨样本信息泄露, 同时降低显存
  - 支持可变节点数: 每个样本独立处理, 通过 batch_idx 循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 物理偏置自注意力
# ============================================================

class PhysicsBiasedAttention(nn.Module):
    """
    带三种物理偏置的多头自注意力。

    标准 attention logits = Q @ K^T / sqrt(d_k)
    增强后 = 标准 + spatial_bias + bc_bias + material_bias

    偏置生成: 轻量级 MLP, 将 pairwise 物理特征映射为 per-head 标量偏置。
    乘上小的缩放因子 (0.05-0.1) 确保偏置不会主导注意力,
    而是作为软引导。
    """

    def __init__(self, dim: int = 256, num_heads: int = 8,
                 dropout: float = 0.1, bias_scale_spatial: float = 0.05,
                 bias_scale_bc: float = 0.1, bias_scale_mat: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.bias_scale_spatial = bias_scale_spatial
        self.bias_scale_bc = bias_scale_bc
        self.bias_scale_mat = bias_scale_mat

        # QKV 投影 (合并以减少参数)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 物理偏置生成器 (轻量级, 输入 pairwise 特征 → per-head 标量)
        # 空间偏置:  pairwise 距离 (1,) → num_heads 个标量
        self.spatial_bias_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, num_heads)
        )
        # BC 耦合偏置: |Δis_fixed|, |ΔlogK| (2,) → num_heads
        self.bc_bias_mlp = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, num_heads)
        )
        # 材料偏置: |Δwave_speed| (1,) → num_heads
        self.mat_bias_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, num_heads)
        )

    def forward(self, x, points, point_features):
        """
        Args:
            x:              (N_b, dim) 节点 token
            points:         (N_b, 3)   坐标 (归一化后)
            point_features: (N_b, 7)   逐节点特征
        Returns:
            out: (N_b, dim) 注意力增强后的 token
        """
        N, D = x.shape

        # QKV 投影 + reshape 为多头
        qkv = self.qkv(x)  # (N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (H, N, d_h)
        k = k.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # 标准注意力 logits
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (H, N, N)

        # ========================================
        # 物理偏置 1: 空间衰减
        # ========================================
        # pairwise 距离矩阵
        dist = torch.cdist(points, points, p=2)  # (N, N)
        # 特征长度: 使用当前样本的最大距离
        L_char = dist.max().clamp(min=1e-6)
        spatial_feat = torch.log1p(dist / L_char).unsqueeze(-1)  # (N, N, 1)
        spatial_bias = self.spatial_bias_mlp(spatial_feat)  # (N, N, H)
        spatial_bias = spatial_bias.permute(2, 0, 1)  # (H, N, N)
        # 负号: 远处衰减; small scale: 软引导
        attn_logits = attn_logits - spatial_bias * self.bias_scale_spatial

        # ========================================
        # 物理偏置 2: 边界条件耦合
        # ========================================
        # is_fixed (索引 3) + log10(K) (索引 4)
        bc_i = point_features[:, [3, 4]]  # is_fixed, log10(K)
        bc_diff = (bc_i.unsqueeze(0) - bc_i.unsqueeze(1)).abs()  # (N, N, 2)
        bc_bias = self.bc_bias_mlp(bc_diff)  # (N, N, H)
        bc_bias = bc_bias.permute(2, 0, 1)  # (H, N, N)
        # BC 相似 → 注意力增强 (正号, 因为 BC 耦合应当促进注意力)
        attn_logits = attn_logits + bc_bias * self.bias_scale_bc

        # ========================================
        # 物理偏置 3: 材料均匀性
        # ========================================
        # wave_speed = sqrt(E/rho)
        E_r = point_features[:, 0]    # E/E_base
        rho_r = point_features[:, 2]  # rho/rho_base
        wave_speed = torch.sqrt(torch.abs(E_r / (rho_r + 1e-6)))  # (N,)
        ws_diff = (wave_speed.unsqueeze(0) - wave_speed.unsqueeze(1)).abs().unsqueeze(-1)  # (N, N, 1)
        mat_bias = self.mat_bias_mlp(ws_diff)  # (N, N, H)
        mat_bias = mat_bias.permute(2, 0, 1)  # (H, N, N)
        # 材料差异大 → 注意力衰减 (负号)
        attn_logits = attn_logits - mat_bias * self.bias_scale_mat

        # ========================================
        # Softmax + Value 加权
        # ========================================
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (H, N, d_h)
        out = out.transpose(0, 1).reshape(N, D)  # (N, D)
        return self.proj(out)


# ============================================================
# Transformer 层 (Pre-LN)
# ============================================================

class TransformerLayer(nn.Module):
    """
    单个 Transformer 层: Pre-LN Attention + Pre-LN FFN。

    Pre-LayerNorm 架构:
      x = x + Attention(LN(x))
      x = x + FFN(LN(x))

    相比 Post-LN, Pre-LN 在训练初期梯度更稳定,
    对深层网络和物理偏置注意力尤为重要。
    """

    def __init__(self, dim: int = 256, num_heads: int = 8,
                 ff_expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsBiasedAttention(dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        ff_hidden = dim * ff_expand
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden, dim),
        )
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, points, point_features):
        """
        Args:
            x:              (N_b, dim)
            points:         (N_b, 3)
            point_features: (N_b, F)
        Returns:
            x: (N_b, dim) 更新后的 token
        """
        # Pre-LN Attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, points, point_features)
        x = self.dropout1(x)
        x = residual + x

        # Pre-LN FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x

        return x


# ============================================================
# 物理偏置 Transformer 编码器
# ============================================================

class PhysicsTransformerEncoder(nn.Module):
    """
    物理偏置自注意力编码器。

    逐样本独立处理: 每个样本的节点在自己的注意力空间内交互。
    这避免了跨样本信息泄露, 也自然支持可变节点数。

    架构:
      - n_layers 个 TransformerLayer
      - 每个 layer 包含 PhysicsBiasedAttention + FFN
      - Pre-LayerNorm 确保训练稳定性
    """

    def __init__(self, dim: int = 256, num_heads: int = 8,
                 n_layers: int = 3, ff_expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads, ff_expand, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, points, point_features, batch):
        """
        Args:
            x:              (total_N, dim) 拼接后的节点 token
            points:         (total_N, 3)   归一化坐标
            point_features: (total_N, F)   逐节点特征
            batch:          (total_N,)     批次索引
        Returns:
            x: (total_N, dim) Transformer 增强后的 token
        """
        outputs = []
        B = int(batch.max().item()) + 1

        for b in range(B):
            mask = batch == b
            x_b = x[mask]               # (N_b, dim)
            pts_b = points[mask]        # (N_b, 3)
            feat_b = point_features[mask]  # (N_b, F)

            for layer in self.layers:
                x_b = layer(x_b, pts_b, feat_b)

            outputs.append(x_b)

        return torch.cat(outputs, dim=0)
