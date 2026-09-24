# -*- coding: utf-8 -*-
"""中性化链试金石（纯本地合成数据）：

1. 纯行业 beta 分数 → 残差与行业无关，IC ≈ 0
2. 纯市值 beta 分数 → 残差与市值无关，IC ≈ 0
3. 混合分数（行业 + 市值 + 信号）→ 残差 ≈ 信号成分，IC 由信号驱动

注：链末端有残差 z-score，会把"完全被剥离后的数值噪声"放大成 O(1) 的
z 噪声，故"杀死"的判据用相关性/IC 而非残差幅值。"""
import numpy as np
import torch

from ashare.evaluator import ICEvaluator

N, T = 240, 3
INDS = ["银行", "白酒", "半导体", "医药", "钢铁"]


def _make(rng_seed, label_signal=None):
    rng = np.random.default_rng(rng_seed)
    ind = rng.choice(INDS, size=N)
    # 均匀分布（无尾部）：MAD 去极值不触发截断，纯线性 → 可精确检验中性化；
    # 正态尾部的 winsor 截断是链的有意行为（去极值在中性化之前），不在此测
    mv = rng.uniform(9.0, 11.0, size=(N, T))
    labels = (label_signal if label_signal is not None
              else rng.normal(size=(N, T))).astype(np.float32)
    ev = ICEvaluator(np.zeros((N, 6, T), dtype=np.float32), labels,
                     np.ones((N, T), dtype=bool),
                     industry=np.tile(ind[:, None], (1, T)), log_mktcap=mv,
                     neutralize=True, device=torch.device("cpu"))
    ind_code = np.searchsorted(INDS, ind).astype(np.float32)
    return ev, ind_code, mv


def test_industry_beta_killed():
    ev, ind_code, _ = _make(11)
    scores = torch.tensor(np.tile(ind_code[:, None], (1, T)), dtype=torch.float32)
    resid = ev.neut.apply(scores[:, 0].unsqueeze(0), 0)[0]
    # 纯行业分数被完全剥离：残差 z 为零向量（IC 的秩并列 → IC=0）
    assert resid.abs().max().item() < 1e-6, "行业 beta 残差未清零"
    ics = ev.ic_series(scores).numpy()
    assert abs(np.nanmean(ics)) < 0.05


def test_size_beta_killed():
    ev, _, mv = _make(12)
    scores = torch.tensor((mv - mv.mean(axis=0)).astype(np.float32))
    resid = ev.neut.apply(scores[:, 0].unsqueeze(0), 0)[0]
    assert resid.abs().max().item() < 1e-6, "市值 beta 残差未清零"
    ics = ev.ic_series(scores).numpy()
    assert abs(np.nanmean(ics)) < 0.05


def test_mixed_signal_preserved():
    rng = np.random.default_rng(13)
    eps = rng.normal(size=(N, T)).astype(np.float32)      # 真信号（标签由它驱动）
    ev, ind_code, mv = _make(13, label_signal=eps)
    scores = torch.tensor((3.0 * np.tile(ind_code[:, None], (1, T))
                           + 2.0 * (mv - mv.mean(axis=0)) + 0.3 * eps).astype(np.float32))
    resid = ev.neut.apply(scores[:, 0].unsqueeze(0), 0)[0].numpy()
    corr = np.corrcoef(resid, eps[:, 0])[0, 1]
    assert corr > 0.95, f"混合分数的信号成分丢失: corr={corr:.3f}"
    # 残差不应再携带 beta
    assert abs(np.corrcoef(resid, ind_code)[0, 1]) < 0.05
    assert abs(np.corrcoef(resid, mv[:, 0])[0, 1]) < 0.05
    ics = ev.ic_series(scores).numpy()
    assert np.nanmean(ics) > 0.3, f"中性化后信号 IC 过低: {np.nanmean(ics):.3f}"
