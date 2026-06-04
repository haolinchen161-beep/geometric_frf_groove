"""
groove_transfrf.py — GrooveTransFRF: 几何感知物理 Transformer 用于 FRF 预测。

架构升级: DeepONet-SIREN → GrooveTransFRF

  points(N,3) + pt_feat(N,7)
          │
  ┌───────┴────────┐
  │ GeomEncoder    │ ← [改动1] SIREN + PointMLP + k-NN EdgeConv → (N, 256)
  └───────┬────────┘
          │
  ┌───────┴────────┐
  │ PhysTransformer│ ← [改动2] 2-3层物理偏置自注意力 (非局部通信)
  └───────┬────────┘
          │
    ┌─────┴─────┐
    │           │
┌───┴──┐  ┌────┴─────┐
│head_phi│  │BC-Aware  │ ← [改动3] 可学习模态token交叉注意力替代均值池化
│→φ_k(x)│  │Decoder   │
│(N,K)  │  │→Δω,ζ(B,2K)│
└───┬──┘  └────┬─────┘
    │          │
    └────┬─────┘
         │
  ┌──────┴──────┐
  │ macro+micro │ ← softplus(macro([√E/ρ,logK]))×15000 + tanh(micro(geo))×10000
  │ ζ = softplus×0.004+1e-4 │
  └──────┬──────┘
         │
  ┌──────┴──────┐
  │ PhysicsDec  │ ← [保留] 模态叠加公式 H=Σφ_k(x)φ_k(x_f)/(ω_k²-ω²+j2ζ_kω_kω)
  │ →FRF(N,F,2) │
  └─────────────┘

与 ModalFRFModel 的 forward() 接口完全兼容,
trainer.py, evaluate.py, predict.py 无需修改。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps, knn_interpolate

from .geometry_data import GeometryData
from .geometry_encoder import GeometricStructureEncoder, normalize_per_sample
from .physics_transformer import PhysicsTransformerEncoder
from .bc_aware_decoder import BCAwareModalDecoder
from .physics_decoder import PhysicsDecoder  # 直接复用物理解码器


class GrooveTransFRF(nn.Module):
    """
    几何感知物理 Transformer — FRF 预测。

    输入/输出接口与 ModalFRFModel 完全兼容。
    """

    def __init__(self, coord_dim=3, point_feat_dim=7,
                 hidden_dim=256, token_dim=256, n_modes=2,
                 siren_layers=4, siren_w0=30.0,
                 n_trans_layers=3, num_heads=8, k=16,
                 dropout=0.1,
                 num_super_nodes=512,
                 amp_scale=500000.0, freq_min=1.0, freq_max=5000.0):
        """
        Args:
            coord_dim:        坐标维度 (3)
            point_feat_dim:   逐节点特征维度 (7)
            hidden_dim:       SIREN 隐藏维度
            token_dim:        Transformer token 维度
            n_modes:          模态阶数 K
            siren_layers:     SIREN 层数
            siren_w0:         SIREN 频率因子
            n_trans_layers:   Transformer 层数
            num_heads:        注意力头数
            k:                k-NN 邻居数
            dropout:          Transformer dropout
            num_super_nodes:  FPS 超节点数 (Transformer 仅在此子集上运行,
                              降低 O(N²) 显存占用至 O(M²))
            amp_scale:        FRF 幅值缩放
            freq_min/max:     频率范围 (Hz)
        """
        super().__init__()
        self.n_modes = n_modes
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_super_nodes = num_super_nodes

        # ============================================
        # 改动1: 几何结构编码器 (替代纯 SIREN Trunk)
        # ============================================
        self.geometry_encoder = GeometricStructureEncoder(
            coord_dim=coord_dim,
            feat_dim=point_feat_dim,
            hidden_dim=hidden_dim,
            token_dim=token_dim,
            siren_layers=siren_layers,
            siren_w0=siren_w0,
            k=k,
        )

        # ============================================
        # 改动2: 物理偏置 Transformer
        # ============================================
        self.transformer = PhysicsTransformerEncoder(
            dim=token_dim,
            num_heads=num_heads,
            n_layers=n_trans_layers,
            dropout=dropout,
        )

        # ============================================
        # 改动3: BC-Aware 模态解码器 (替代 global_mean_pool + head_modal)
        # ============================================
        self.modal_decoder = BCAwareModalDecoder(
            token_dim=token_dim,
            n_modes=n_modes,
            num_heads=max(4, num_heads // 2),  # 交叉注意力头数
            dropout=dropout,
        )

        # ============================================
        # 保留: head_phi (从 Transformer token → 模态振型)
        # 相比旧架构稍增强: Token → hidden → n_modes
        # ============================================
        self.head_phi = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, n_modes),
        )

        # ============================================
        # 双流残差 ω 解码: 宏观物理 + 微观几何
        # ============================================
        self.macro_omega = nn.Sequential(          # 宏观流: 真物理量→ω基线
            nn.Linear(2, 64), nn.GELU(),           # 2 = √(E/ρ), avg_logK
            nn.Linear(64, n_modes)
        )
        self.micro_omega = nn.Sequential(          # 微观流: 几何特征→Δω
            nn.Linear(token_dim, 64), nn.GELU(),
            nn.Linear(64, n_modes)
        )
        # 零初始化: 让宏观基线从零开始, 微观修正也从零开始
        for m in [self.macro_omega, self.micro_omega]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.0)
                    nn.init.constant_(layer.bias, 0.0)

        # ============================================
        # 保留: PhysicsDecoder (无参数物理解码器)
        # ============================================
        self.physics = PhysicsDecoder(
            amp_scale=amp_scale,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    def _compute_physics_prior(self, point_features, batch):
        """
        计算物理先验 (B, 2): [√(E/ρ), avg_logK]

        两个真物理量 (与 ω 正相关):
          1. √(E/ρ):       材料波速, ω ∝ √(E/ρ) — 振动理论严格结论
          2. avg_logK(bc):  边界弹簧平均刚度 — ω 随约束增强而增大

        几何信息 (凹槽深度/位置) 由 SIREN+EdgeConv+Transformer 学习,
        不在此处强行编码。Z/H 已作为逐节点特征输入几何编码器。

        Args:
            point_features: (total_N, 7) [E,PRXY,DENS,is_fixed,logK,logC,Z/H]
            batch:          (total_N,)
        Returns:
            physics: (B, 2)
        """
        E_r = point_features[:, 0]
        rho_r = point_features[:, 2]
        logK = point_features[:, 4]

        wave_speed = torch.sqrt(torch.abs(E_r / (rho_r + 1e-6)))

        B = int(batch.max().item()) + 1
        device = point_features.device

        physics_list = []
        for b in range(B):
            mask = batch == b
            ws_b = wave_speed[mask][0]

            bc_mask = logK[mask] > 0
            if bc_mask.any():
                avg_logK = logK[mask][bc_mask].mean()
            else:
                avg_logK = torch.tensor(0.0, device=device)

            physics_list.append(torch.stack([ws_b, avg_logK]))

        return torch.stack(physics_list)  # (B, 2)

    def _prepare_inputs(self, geometry_data):
        """
        将 GeometryData 统一为 (total_N, 3/7) + (total_N,) batch 格式。

        Returns:
            points:         (total_N, 3)
            point_features: (total_N, F)
            batch:          (total_N,)
        """
        points = geometry_data.points
        point_feat = geometry_data.point_features
        batch = geometry_data.batch

        if points.ndim == 3:
            # 固定 N: (B, N, 3) → (B*N, 3)
            B, N_max, _ = points.shape
            points = points.reshape(-1, 3)
            if point_feat is not None:
                point_feat = point_feat.reshape(-1, point_feat.shape[-1])
            batch = torch.arange(B, device=points.device).repeat_interleave(N_max)

        return points, point_feat, batch

    def forward(self, geometry_data, frequencies=None, phi_exc=None):
        """
        前向传播 — 与 ModalFRFModel.forward() 完全兼容。

        流程:
          1. 几何编码 (全分辨率 N)
          2. FPS 子采样 → M 个超节点 (M=num_super_nodes, 默认512)
          3. 物理偏置 Transformer (仅超节点, O(M²) 而非 O(N²))
          4. k-NN 插值超节点特征回全分辨率 → head_phi
          5. BC-Aware 模态解码器 (超节点 → 全局 ω, ζ)
          6. PhysicsDecoder → FRF

        Args:
            geometry_data: GeometryData
            frequencies:   (B, F) 归一化频率 [-1,1] 或 None
            phi_exc:       (B, K) 激励点振型值
        Returns:
            frf, omega, zeta, phi
        """
        points, point_feat, batch = self._prepare_inputs(geometry_data)

        if point_feat is None:
            point_feat = torch.zeros(points.shape[0], 7, device=points.device)
            point_feat[:, 3] = 1.0

        # ============================================
        # 步骤1: 几何结构编码 (全分辨率)
        # ============================================
        node_tokens = self.geometry_encoder(points, point_feat, batch)
        # (total_N, token_dim)

        pts_norm = normalize_per_sample(points, batch)

        # ============================================
        # 步骤2: FPS 子采样 + Transformer (仅超节点)
        # ============================================
        B = int(batch.max().item()) + 1
        device = node_tokens.device

        super_tokens_list = []
        super_pts_list = []
        super_feat_list = []
        super_batch_list = []
        fps_indices_list = []

        for b in range(B):
            mask = batch == b
            N_b = int(mask.sum().item())
            M_b = min(self.num_super_nodes, N_b)

            tokens_b = node_tokens[mask]      # (N_b, D)
            pts_b = pts_norm[mask]            # (N_b, 3)
            feat_b = point_feat[mask]         # (N_b, F)

            # FPS: 选择 M_b 个最具空间代表性的节点
            # fps 输入: (N, 3) 坐标, 可选 batch, ratio
            batch_b = torch.zeros(N_b, dtype=torch.long, device=device)
            ratio = M_b / N_b
            fps_idx = fps(pts_b, batch_b, ratio=ratio)  # (M_b,)

            super_tokens_list.append(tokens_b[fps_idx])
            super_pts_list.append(pts_b[fps_idx])
            super_feat_list.append(feat_b[fps_idx])
            super_batch_list.append(torch.full((M_b,), b, dtype=torch.long, device=device))
            fps_indices_list.append(fps_idx)

        super_tokens = torch.cat(super_tokens_list, dim=0)      # (total_M, D)
        super_pts = torch.cat(super_pts_list, dim=0)            # (total_M, 3)
        super_feat = torch.cat(super_feat_list, dim=0)          # (total_M, F)
        super_batch = torch.cat(super_batch_list, dim=0)        # (total_M,)

        # Transformer 仅在超节点上运行
        super_tokens = self.transformer(super_tokens, super_pts, super_feat, super_batch)
        # (total_M, token_dim)

        # ============================================
        # 步骤3: 插值回全分辨率 + 双路径解码
        # ============================================
        # k-NN 插值: 超节点特征 → 全节点特征
        # knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k=3)
        node_tokens_full = torch.zeros_like(node_tokens)
        for b in range(B):
            mask = batch == b
            s_mask = super_batch == b
            N_b = int(mask.sum().item())

            super_feat_b = super_tokens[s_mask]  # (M_b, D)
            super_pts_b = super_pts[s_mask]      # (M_b, 3)
            node_pts_b = pts_norm[mask]            # (N_b, 3)
            s_batch_b = torch.zeros(super_feat_b.shape[0], dtype=torch.long, device=device)
            n_batch_b = torch.zeros(N_b, dtype=torch.long, device=device)

            interpolated = knn_interpolate(
                super_feat_b, super_pts_b, node_pts_b,
                s_batch_b, n_batch_b, k=3,
            )  # (N_b, D)
            node_tokens_full[mask] = interpolated

        # 局部路径: 模态振型 (全分辨率, 插值特征 → head_phi)
        phi = self.head_phi(node_tokens_full)  # (total_N, K)

        # 全局路径: 模态参数 (超节点 → BC-Aware 交叉注意力池化)
        modal_out = self.modal_decoder(super_tokens, super_batch)  # (B, 2K)

        # ============================================
        # 步骤4: 双流残差 ω + ζ
        # 宏观流: 物理标量→ω基线; 微观流: 几何特征→Δω
        # ============================================
        physics_prior = self._compute_physics_prior(point_feat, batch)  # (B, 2)

        # 微观流: 超节点全局均值池化→几何特征
        geo_feat = torch.zeros(B, self.token_dim, device=device)
        for b in range(B):
            mask = super_batch == b
            geo_feat[b] = super_tokens[mask].mean(dim=0)

        omega_coarse = F.softplus(self.macro_omega(physics_prior)) * 15000.0  # 宏观基线
        omega_fine = torch.tanh(self.micro_omega(geo_feat)) * 10000.0         # 微观修正
        omega = omega_coarse + omega_fine

        zeta = F.softplus(modal_out[:, self.n_modes:]) * 0.004 + 1e-4

        # ============================================
        # 步骤5: 物理重建 FRF
        # ============================================
        if frequencies is not None:
            frf_raw = self.physics(phi, omega, zeta, frequencies, phi_exc,
                                   batch_idx=batch)
            frf = torch.asinh(frf_raw.clamp(-1e4, 1e4))  # 输出在 asinh 空间, 避免梯度消失
        else:
            frf = None

        return frf, omega, zeta, phi
