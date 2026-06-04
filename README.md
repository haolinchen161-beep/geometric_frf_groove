# geometric_frf_groove — 基于几何的频响函数预测

输入 3D 几何 + 边界条件 → GrooveTransFRF (几何感知物理 Transformer) → 模态参数 (ω, ζ, φ) → 物理公式重建 FRF。

## 1. 目录结构

```
├── models/
│   ├── groove_transfrf.py      主模型: GrooveTransFRF
│   ├── frf_model.py             模型工厂 build_geometric_model()
│   ├── geometry_encoder.py      几何结构编码器 (SIREN + PointMLP + k-NN EdgeConv)
│   ├── physics_transformer.py   物理偏置自注意力 Transformer
│   ├── bc_aware_decoder.py      BC感知模态解码器 (交叉注意力池化)
│   ├── physics_decoder.py       无参数物理解码器 (模态叠加→FRF)
│   ├── geometry_data.py         GeometryData 数据容器
│   └── siren.py                 SIREN 正弦激活 (w0=30)
├── data/
│   └── dataset.py               HDF5 数据集 (per-sample-group) + collate
├── training/
│   ├── losses.py                modal_loss + frf_loss
│   ├── trainer.py               五阶段训练循环 + 评估
│   └── augmentations.py         数据增强 (坐标/特征噪声, 节点dropout, 频率子采样)
├── ansys/
│   ├── generate_3d_test.py      ANSYS MAPDL 数据生成 (凹槽工件)
│   ├── data/                    train/val/test.h5
│   └── mesh_viz/                网格截图
└── sample/
    ├── run_validation.py        训练入口
    ├── evaluate.py              评估 + 保存 final_results.npz
    ├── 测试.py                  查看原始 FRF
    ├── 对比图.py                预测 vs 真实对比图
    ├── predict.py               推理
    └── output/                  checkpoint + 图表 + npz
```

## 2. 架构

```
输入: points(N,3) + point_features(N,7) + frequencies(B,F) + phi_exc(B,K)
                    │
    ┌───────────────┴───────────────┐
    │  GeometricStructureEncoder    │  SIREN + PointMLP + k-NN EdgeConv
    │  → node_tokens (N, 256)       │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │  FPS 子采样 (N → M=512)       │  降低 O(N²) → O(M²)
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │  PhysicsTransformer (3层)     │  物理偏置自注意力 (空间衰减+BC耦合+材料均匀性)
    │  → super_tokens (M, 256)     │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │  k-NN 插值 (M → N)           │  恢复全分辨率
    └───────────────┬───────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    ┌────┴────┐          ┌─────┴──────┐
    │head_phi │          │BC-Aware    │  可学习模态token交叉注意力
    │→φ_k(x)  │          │ModalDecoder│
    │(N,K)    │          │→Δω,ζ (B,2K)│
    └────┬────┘          └─────┬──────┘
         │                     │
    ┌────┴─────────────────────┴──────┐
    │  ω = softplus(skip)×20000 + tanh(Δω)×8000  │
    │  ζ = softplus(Δζ)×0.004 + 1e-4            │
    └────────────────┬────────────────┘
                     │
    ┌────────────────┴────────────────┐
    │  PhysicsDecoder (无参数)         │  H=Σφ_k(x)φ_k(x_f)/(ω_k²-ω²+j2ζ_kω_kω)
    │  → FRF (N, F, 2) [Re, Im]      │
    └─────────────────────────────────┘
```

## 3. 数据

### ANSYS 凹槽工件

| 参数 | 值 |
|------|-----|
| 工件尺寸 | 160×60×10mm (铝7075, E=71.7GPa, ρ=2810) |
| 凹槽 | 5/6/7 方案, 深度随机 30-60% 厚度 |
| 装夹 | 4角 XYZ 弹簧 + 3侧面 Y 向弹簧 (COMBIN14) |
| 弹簧刚度 | K∈[1e6, 1e8] N/m, 随机 |
| 弹簧阻尼 | C∈[10, 500] N·s/m, 随机 |
| 阻尼 | ζ=材料0.002 + 边界耗散 (物理公式) |
| 模态 | 前2阶, Y向振型 |
| 网格 | 6mm 自由四面体 (SOLID187), ~4k 节点/样本 |
| 频率 | 40点自适应网格 |

### HDF5 格式 (per-sample-group)

```
/sample_0/
├── points         (N₀, 3)       节点坐标 [x, y, z] (m)
├── point_frf      (N₀, F₀, 2)  复数频响函数 [Re, Im]
├── frequencies    (F₀,)         频率采样点 (Hz)
├── point_features (N₀, 7)       逐节点特征:
│                                [E/E_base, PRXY, ρ/ρ_base, is_active, is_fixed, log10(K), log10(C)]
├── modal_omega    (K,)          固有圆频率 (rad/s)
├── modal_zeta     (K,)          阻尼比
├── modal_phi      (N₀, K)       模态振型 φ_k(x)
└── modal_phi_exc  (K,)          激励点振型值
```

## 4. 训练

### 五阶段策略

| 阶段 | Epoch | 冻结 | 损失 | 目的 |
|------|-------|------|------|------|
| 1: Warmup | 0-300 | Transformer, head_phi | modal_loss | 建立几何编码 |
| 2: Attn预热 | 300-600 | head_phi | modal + 轻FRF | 引入非局部通信 |
| 3: FRF对齐 | 600-1200 | head_phi | modal + 强FRF | 物理约束对齐ω,ζ |
| 4: 联合微调 | 1200-2000 | 无 | modal + 强FRF | 端到端精修 |
| 5: LR冷却 | 2000-2500 | 无 | modal + 强FRF | 最终收敛 |

### 超参数

| 参数 | 值 |
|------|-----|
| 模型 | GrooveTransFRF (~2.5M params) |
| token_dim / num_heads | 256 / 8 |
| n_trans_layers / FPS nodes | 3 / 512 |
| k-NN / dropout | 16 / 0.1 |
| 损失权重 | rel_ω×500K + rel_ζ×5K + φ×1K + FRF×30 |
| 优化器 | AdamW, lr=3e-4, wd=8e-5, CosineAnnealing |
| 梯度裁剪 | Transformer=1.0, SIREN=5.0, Modal=2.0 |
| batch_size | 4 |

## 5. 快速开始

```bash
# 生成数据 (需 ANSYS MAPDL license)
F:/pytorch_cuda12/python.exe ansys/generate_3d_test.py

# 查看原始 FRF
F:/pytorch_cuda12/python.exe sample/测试.py

# 训练
F:/pytorch_cuda12/python.exe sample/run_validation.py

# 评估
F:/pytorch_cuda12/python.exe sample/evaluate.py

# 对比图
F:/pytorch_cuda12/python.exe sample/对比图.py
```
