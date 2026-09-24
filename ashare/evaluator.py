# -*- coding: utf-8 -*-
"""截面 RankIC 评估器（Phase 1 朴素版 reward）。

公式输出 [N, Tg] → 逐网格日：池内 ∩ 标签有效 的 Spearman 秩相关
（对 t8 池内超额收益）→ 所选网格日的均值。

秩相关用 torch 平均秩（并列取平均）实现，与 scipy.stats.spearmanr 数值
一致（tests/test_evaluator.py 对账）；标签秩在初始化时一次性预计算。
"""
import numpy as np
import torch

from .config import MIN_POOL_IC


def avg_rank(x: torch.Tensor) -> torch.Tensor:
    """沿 dim=1 升序赋 1..n 的平均秩；NaN/Inf 位秩为 NaN。x: [B, N]。"""
    B, N = x.shape
    valid = torch.isfinite(x)
    filled = torch.where(valid, x, torch.full_like(x, float("inf")))
    sv, si = torch.sort(filled, dim=1)                    # inf（无效）排最后
    pos = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)
    pos = pos.unsqueeze(0).expand(B, N)
    finite = torch.isfinite(sv)
    # 并列组边界 → 组首/组尾位置 → 平均秩
    ones = torch.ones(B, 1, dtype=torch.bool, device=x.device)
    start_b = torch.cat([ones, sv[:, 1:] != sv[:, :-1]], dim=1)
    end_b = torch.cat([sv[:, :-1] != sv[:, 1:], ones], dim=1)
    first = torch.cummax(torch.where(start_b, pos, torch.zeros_like(pos)), dim=1).values
    # 组尾位置 = "右侧最近的组尾标记"：翻转后用 (N+1-pos) 单调递增使 cummax 等价于右向 ffill
    c_end = torch.where(end_b, (N + 1) - pos, torch.zeros_like(pos))
    last = (N + 1) - torch.flip(torch.cummax(torch.flip(c_end, [1]), dim=1).values, [1])
    rank_sorted = (first + last) / 2.0 * finite           # 无效组秩置 0，随后统一 NaN
    out = torch.zeros_like(x)
    out.scatter_(1, si, rank_sorted)
    return out.where(valid, torch.full_like(out, float("nan")))


def _pearson(a: torch.Tensor, b: torch.Tensor, min_n: int) -> torch.Tensor:
    """逐行 Pearson（b 广播到 a 的行数）；行内有效样本 < min_n 返回 NaN。"""
    ok = torch.isfinite(a) & torch.isfinite(b)
    cnt = ok.sum(dim=1)
    af = torch.where(ok, a, torch.zeros_like(a))
    bf = torch.where(ok, b, torch.zeros_like(b))
    ma = af.sum(dim=1) / cnt.clamp(min=1)
    mb = bf.sum(dim=1) / cnt.clamp(min=1)
    da = torch.where(ok, a - ma[:, None], torch.zeros_like(a))
    db = torch.where(ok, b - mb[:, None], torch.zeros_like(b))
    cov = (da * db).sum(dim=1)
    var = torch.sqrt((da * da).sum(dim=1) * (db * db).sum(dim=1))
    ic = cov / (var + 1e-12)
    return torch.where(cnt >= min_n, ic, torch.full_like(ic, float("nan")))


class ICEvaluator:
    """从物化面板构建 IC 评估器。labels NaN 与 mask 外的样本不参与。"""

    def __init__(self, features, labels, mask, device=None):
        from .config import DEVICE
        self.device = device or DEVICE
        self.features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        labels = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

        self.valid = mask & torch.isfinite(labels)          # [N, Tg]
        # 池内超额：逐列减去当日有效样本均值（课题 excess='pool' 口径）
        cnt = self.valid.sum(dim=0).clamp(min=1)
        mean = torch.where(self.valid, labels, torch.zeros_like(labels)).sum(dim=0) / cnt
        self.excess = torch.where(self.valid, labels - mean[None, :],
                                  torch.full_like(labels, float("nan")))
        # 标签秩一次性预计算（标签固定）；逐日有效索引缓存
        self.label_rank = avg_rank(self.excess.T).T          # [N, Tg]
        self.date_idx = [torch.nonzero(self.valid[:, t], as_tuple=False).squeeze(1)
                         for t in range(self.valid.shape[1])]
        self.usable_dates = [t for t, idx in enumerate(self.date_idx)
                             if idx.numel() >= MIN_POOL_IC]

    def ic_series(self, scores, dates=None) -> torch.Tensor:
        """scores [B, N, Tg]（或 [N, Tg]）→ 各日 IC [B, D]（无效日 NaN）。"""
        single = scores.dim() == 2
        sc = scores.unsqueeze(0) if single else scores
        use = self.usable_dates if dates is None else dates
        out = torch.full((sc.shape[0], len(use)), float("nan"), device=self.device)
        for k, t in enumerate(use):
            idx = self.date_idx[t]
            s = sc[:, idx, t]                                # [B, V]
            e = self.label_rank[idx, t].unsqueeze(0).expand(sc.shape[0], -1)
            out[:, k] = _pearson(avg_rank(s), e, MIN_POOL_IC)
        return out.squeeze(0) if single else out

    def evaluate(self, scores, dates=None) -> torch.Tensor:
        """scores [B, N, Tg] → reward [B]（所选日期 IC 均值，全 NaN 日期则 -10）。"""
        ics = self.ic_series(scores, dates)
        nan_mask = torch.isfinite(ics)
        denom = nan_mask.sum(dim=1).clamp(min=1)
        mean_ic = torch.where(nan_mask, ics, torch.zeros_like(ics)).sum(dim=1) / denom
        return torch.where(torch.isfinite(mean_ic), mean_ic,
                           torch.full_like(mean_ic, -10.0))

    def evaluate_columns(self, scores, dates) -> torch.Tensor:
        """scores [B, N, len(dates)]，最后一维与 dates（原始日期索引）对齐 → reward [B]。

        训练主循环只为抽中的日期子集执行栈机/保留分数，走此入口省内存。"""
        B = scores.shape[0]
        ics = torch.full((B, len(dates)), float("nan"), device=self.device)
        for k, t in enumerate(dates):
            idx = self.date_idx[t]
            s = scores[:, idx, k]                        # [B, V]
            e = self.label_rank[idx, t].unsqueeze(0).expand(B, -1)
            ics[:, k] = _pearson(avg_rank(s), e, MIN_POOL_IC)
        nan_mask = torch.isfinite(ics)
        denom = nan_mask.sum(dim=1).clamp(min=1)
        mean_ic = torch.where(nan_mask, ics, torch.zeros_like(ics)).sum(dim=1) / denom
        return torch.where(torch.isfinite(mean_ic), mean_ic,
                           torch.full_like(mean_ic, -10.0))


def evaluator_from_panel(panel=None):
    """便捷构造：从 npz 面板（或磁盘加载）构建 ICEvaluator。"""
    if panel is None:
        from .materialize import load_panel
        panel = load_panel()
    return ICEvaluator(panel["features"], panel["labels"], panel["mask"])
