"""
geometry_encoder.py — 几何结构编码器。

将纯 SIREN Trunk 升级为三条特征流融合:
  1. SIREN: 高频空间编码 (保留自旧架构)
  2. PointMLP: 逐节点材料+边界条件编码 (无池化, 保留BC位置信息)
  3. LocalGeometryEncoder (EdgeConv): k-NN 局部几何结构捕获 (筋厚度、凹槽曲率)

三条流拼接后线性融合 → node_tokens (N, token_dim)

物理依据:
  模态振型 φ_k(x) 取决于局部刚度分布。凹槽将厚度从 10mm 减至 4mm,
  弯曲刚度降至 (4/10)³=6.4%。仅凭坐标无法区分薄筋与厚基体节点。
  k-NN 局部编码器提供直接的局部几何证据, 对预测模态振型幅值至关重要。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph

from .siren import SirenMLP


# ============================================================
# 工具函数: scatter_mean (PyG 2.6 兼容)
# ============================================================

def scatter_mean(src, index, dim=0, dim_size=None):
    """按 index 对 src 做均值聚合。PyG 2.6 中 scatter_mean 不在 nn 中, 用原生 PyTorch 实现。"""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out = src.new_zeros(dim_size, src.shape[1])
    out.index_add_(0, index, src)
    count = src.new_zeros(dim_size, 1)
    count.index_add_(0, index, src.new_ones(src.shape[0], 1))
    count.clamp_(min=1)
    return out / count


# ============================================================
# 工具函数: 逐样本坐标归一化
# ============================================================

def normalize_per_sample(points, batch):
    """
    每个样本独立归一化坐标到 [-1, 1]。
    与 ModalFRFModel 中的归一化逻辑一致。

    Args:
        points: (total_N, 3) 拼接后的点坐标
        batch:  (total_N,)    批次索引
    Returns:
        pts_norm: (total_N, 3) 归一化后的坐标
    """
    pts_norm_list = []
    for b in range(int(batch.max().item()) + 1):
        mask = batch == b
        p_b = points[mask]
        lo = p_b.min(dim=0, keepdim=True)[0]
        hi = p_b.max(dim=0, keepdim=True)[0]
        pts_norm_list.append((p_b - lo) / (hi - lo + 1e-8) * 2.0 - 1.0)
    return torch.cat(pts_norm_list, dim=0)


# ============================================================
# 改动1a: 局部几何编码器 (EdgeConv 风格)
# ============================================================

class LocalGeometryEncoder(nn.Module):
    """
    k-NN 局部几何结构编码器。

    对每个节点, 在 k 近邻上做 EdgeConv 风格的消息传递:
      edge_feat = concat[feat_src, feat_dst - feat_src]
      node_feat = scatter_mean(MLP(edge_feat))

    捕获: 局部曲率、筋厚度、凹槽边缘接近度、表面法向变化。
    k=16 → 对 6mm 网格覆盖约 24mm 半径 (≈4 个网格单元)。
    """

    def __init__(self, in_dim: int, out_dim: int, k: int = 16):
        """
        Args:
            in_dim:  输入特征维度 (coord_dim + feat_dim, 例如 3+7=10)
            out_dim: 输出特征维度
            k:       k-NN 邻居数
        """
        super().__init__()
        self.k = k
        # EdgeConv MLP: edge = [feat_row, feat_col - feat_row]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim * 2, out_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, points, point_features, batch):
        """
        Args:
            points:         (N, 3) 坐标 (可使用归一化后的坐标)
            point_features: (N, F) 逐节点特征
            batch:          (N,)   批次索引
        Returns:
            node_local: (N, out_dim) 局部几何嵌入
        """
        # 拼接坐标与特征作为 EdgeConv 输入
        node_feat = torch.cat([points, point_features], dim=-1)  # (N, 3+F)

        # k-NN 图 (使用归一化坐标, 更关注拓扑而非绝对尺度)
        edge_index = knn_graph(points, k=self.k, batch=batch, loop=False)

        row, col = edge_index[0], edge_index[1]

        # Edge 特征: [feat_i, feat_j - feat_i]
        feat_diff = node_feat[col] - node_feat[row]
        edge_feat = torch.cat([node_feat[row], feat_diff], dim=-1)

        # MLP + 均值聚合回节点
        edge_emb = self.mlp(edge_feat)  # (E, out_dim)
        node_local = scatter_mean(edge_emb, row, dim=0, dim_size=points.shape[0])

        return node_local


# ============================================================
# 改动1b: PointMLP — 逐节点材料+边界条件编码
# ============================================================

class PointMLP(nn.Module):
    """
    逐节点编码材料属性和边界条件, 不进行池化。
    保留 is_fixed, log10(K), log10(C) 等 BC 关键信号的位置信息。

    输入: point_features (N, 7)
    输出: point_emb     (N, 128)
    """

    def __init__(self, in_dim: int = 7, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, point_features):
        return self.net(point_features)


# ============================================================
# 改动1: 完整的几何结构编码器
# ============================================================

class GeometricStructureEncoder(nn.Module):
    """
    三流几何编码器, 替代纯 SIREN Trunk。

    输入:
      points:         (N, 3) 坐标
      point_features: (N, 7) [E, PRXY, DENS, is_active, is_fixed, log10(K), log10(C)]
      batch:          (N,)   批次索引

    三条流:
      1. SIREN (256-dim):      高频空间编码, 捕获模态振型的空间变化
      2. PointMLP (128-dim):   逐节点材料+BC编码, 保留局部BC信息
      3. LocalGeometry (64-dim): k-NN 局部几何结构

    融合: Linear(256+128+64, token_dim) → node_tokens

    输出: node_tokens (N, token_dim)
    """

    def __init__(self,
                 coord_dim: int = 3,
                 feat_dim: int = 7,
                 hidden_dim: int = 256,
                 token_dim: int = 256,
                 siren_layers: int = 4,
                 siren_w0: float = 30.0,
                 k: int = 16,
                 point_mlp_hidden: int = 64,
                 point_mlp_out: int = 128,
                 local_geom_out: int = 64):
        super().__init__()
        self.token_dim = token_dim

        # 流1: SIREN 空间编码 (保留旧架构的核心能力)
        self.siren = SirenMLP(
            in_dim=coord_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=siren_layers,
            w0=siren_w0,
        )

        # 流2: 逐节点材料+BC编码
        self.point_mlp = PointMLP(
            in_dim=feat_dim,
            hidden_dim=point_mlp_hidden,
            out_dim=point_mlp_out,
        )

        # 流3: k-NN 局部几何结构
        self.local_geom = LocalGeometryEncoder(
            in_dim=coord_dim + feat_dim,
            out_dim=local_geom_out,
            k=k,
        )

        # 三流融合
        fusion_in = hidden_dim + point_mlp_out + local_geom_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, token_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, points, point_features, batch):
        """
        Args:
            points:         (total_N, 3) 或 (B, N, 3)
            point_features: (total_N, F) 或 (B, N, F)
            batch:          (total_N,)   批次索引 (可变N时) 或 None (固定N时)
        Returns:
            node_tokens: (total_N, token_dim)
        """
        # 处理固定N (stacked) 格式
        if points.ndim == 3:
            B, N_max, _ = points.shape
            points_flat = points.reshape(-1, 3)
            feats_flat = point_features.reshape(-1, point_features.shape[-1])
            batch_flat = torch.arange(B, device=points.device).repeat_interleave(N_max)
        else:
            points_flat = points
            feats_flat = point_features
            batch_flat = batch
            if batch_flat is None:
                batch_flat = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)

        # 逐样本坐标归一化
        pts_norm = normalize_per_sample(points_flat, batch_flat)

        # 流1: SIREN 空间编码
        spatial_feat = self.siren(pts_norm)  # (N, hidden_dim=256)

        # 流2: 逐节点 MLP
        point_emb = self.point_mlp(feats_flat)  # (N, 128)

        # 流3: k-NN 局部几何 (使用归一化坐标以关注拓扑)
        local_feat = self.local_geom(pts_norm, feats_flat, batch_flat)  # (N, 64)

        # 三流拼接 + 融合
        fused = torch.cat([spatial_feat, point_emb, local_feat], dim=-1)  # (N, 448)
        node_tokens = self.fusion(fused)  # (N, token_dim)

        return node_tokens
