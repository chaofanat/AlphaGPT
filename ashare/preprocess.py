# -*- coding: utf-8 -*-
"""截面中性化链（Phase 2，对齐课题 PREPROCESS 口径）：

MAD 去极值 → z-score → 申万一级行业 + log 总市值中性化（截面回归取残差）
→ 残差 z-score。

作用于公式的截面分数（而非特征原子）：训练 reward 与终选 IC 都在"剥离了
行业与规模暴露"的残差上计算，挖掘器无法靠行业/市值 beta 得分。

实现按日期预构建设计矩阵 X [V, K]（行业哑变量 + z 后 log 市值 + 截距）并
缓存 pinv(X)^T；逐批次两次小矩阵乘（[B,V]×[V,K]×[K,V]）完成中性化。
X 列不满秩（哑变量和 = 截距）由 pinv 的最小范数解吸收，残差仍唯一。
"""
import numpy as np
import torch

from .config import DEVICE, WINSOR_MAD_CLIP


class CrossSectionNeutralizer:
    """industry / log_mktcap: [N, Tg]（面板原数组；行业为字符串，市值可 NaN）。"""

    def __init__(self, industry, log_mktcap, valid, device=None):
        self.device = device or DEVICE
        valid = np.asarray(valid, dtype=bool)
        industry = np.asarray(industry, dtype=object)
        log_mv = np.asarray(log_mktcap, dtype=np.float64)

        names = sorted({ind for ind in industry[valid] if ind})   # 全面板行业集（含 "" 缺失列）
        self.ind_names = names + ["__missing__"]
        n_dates = valid.shape[1]

        self.X, self.XpT = [], []
        for t in range(n_dates):
            v = valid[:, t]
            V = int(v.sum())
            if V < 30:
                self.X.append(None)
                self.XpT.append(None)
                continue
            ind_t = industry[v, t].astype(object)
            cols = [torch.tensor((ind_t == name).astype(np.float32)) for name in names]
            cols.append(torch.tensor((ind_t == "").astype(np.float32)))   # 缺失行业独立列
            mv = log_mv[v, t].copy()
            fill = np.nanmedian(mv) if np.isfinite(mv).any() else 0.0     # 市值缺失 → 截面中位数
            mv = np.where(np.isfinite(mv), mv, fill)
            mv = mv - mv.mean()
            sd = mv.std()
            cols.append(torch.tensor((mv / (sd + 1e-9)).astype(np.float32)))  # z 后 log 市值
            cols.append(torch.ones(V))
            X = torch.stack(cols, dim=1).to(self.device)                  # [V, K]
            self.X.append(X)
            self.XpT.append(torch.linalg.pinv(X).T.contiguous())          # [V, K]

    def apply(self, scores, t):
        """scores [B, V] → 中性化残差 z [B, V]（该日无设计矩阵则原样返回 z）。"""
        X, XpT = self.X[t], self.XpT[t]
        if X is None:
            return _zscore(_winsor_mad_z(scores.to(torch.float32)))
        y = _winsor_mad_z(scores.to(XpT.dtype))                         # MAD 去极值 + z
        beta = y @ XpT                                                   # [B, K]
        resid = y - beta @ X.T                                           # [B, V]
        return _zscore(resid)


def _winsor_mad_z(x: torch.Tensor) -> torch.Tensor:
    """逐行：MAD 去极值（±WINSOR_MAD_CLIP 倍 MAD）→ z-score。"""
    med = x.median(dim=1, keepdim=True).values
    mad = (x - med).abs().median(dim=1, keepdim=True).values + 1e-9
    z = torch.clamp((x - med) / mad, -WINSOR_MAD_CLIP, WINSOR_MAD_CLIP)
    return _zscore(z)


def _zscore(x: torch.Tensor) -> torch.Tensor:
    """逐行 z-score；退化行（std < 1e-4，如纯 beta 分数被完全剥离后只剩
    ~1e-6 级投影噪声）返回零向量——否则会把数值噪声放大成 O(1) 伪信号。"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    z = (x - mean) / (std + 1e-9)
    return torch.where(std > 1e-4, z, torch.zeros_like(x))
