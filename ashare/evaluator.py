# -*- coding: utf-8 -*-
"""截面 RankIC 评估器（Phase 2：中性化链 + ICIR/分层 + 分段裁决）。

公式输出 [N, Tg] → 逐网格日：池内 ∩ 标签有效 子集上，分数过中性化链
（MAD→z→行业+log市值→残差z，可选）后与 t8 池内超额做 Spearman 秩相关。

秩相关用 torch 平均秩（并列取平均）实现，与 scipy.stats.spearmanr 数值
一致（tests/test_evaluator.py 对账）；标签秩在初始化时一次性预计算。
"""
import numpy as np
import torch

from .config import DEVICE, MIN_POOL_IC, NEUTRALIZE, REWARD_MODE

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


def icir(ics: torch.Tensor) -> torch.Tensor:
    """IC 序列的均值/标准差（沿最后一维；全 NaN 或零方差安全）。"""
    ok = torch.isfinite(ics)
    cnt = ok.sum(dim=-1).clamp(min=1)
    mean = torch.where(ok, ics, torch.zeros_like(ics)).sum(dim=-1) / cnt
    d = torch.where(ok, ics - mean[..., None], torch.zeros_like(ics))
    std = torch.sqrt((d * d).sum(dim=-1) / cnt)
    val = mean / (std + 1e-9)
    return torch.where(cnt > 0, val, torch.full_like(val, float("nan")))


def ic_tstat(ics: torch.Tensor) -> torch.Tensor:
    """IC 序列 t 统计量 = ICIR×√n（网格日间隔=持有期，窗口无重叠，普通 t 即可）。"""
    ok = torch.isfinite(ics)
    cnt = ok.sum(dim=-1).clamp(min=1)
    mean = torch.where(ok, ics, torch.zeros_like(ics)).sum(dim=-1) / cnt
    d = torch.where(ok, ics - mean[..., None], torch.zeros_like(ics))
    std = torch.sqrt((d * d).sum(dim=-1) / cnt)
    t = mean / (std + 1e-9) * torch.sqrt(cnt.to(torch.float32))
    return torch.where(cnt > 1, t, torch.full_like(t, float("nan")))


class ICEvaluator:
    """从物化面板构建 IC 评估器。labels NaN 与 mask 外的样本不参与。

    industry / log_mktcap 传入（面板原数组）且 neutralize 开启时，分数在
    IC 计算前过中性化链（见 preprocess.py）。"""

    def __init__(self, features, labels, mask, industry=None, log_mktcap=None,
                 neutralize=None, device=None):
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

        self.neut = None
        use_neut = NEUTRALIZE if neutralize is None else neutralize
        if use_neut and industry is not None:
            from .preprocess import CrossSectionNeutralizer
            self.neut = CrossSectionNeutralizer(industry, log_mktcap,
                                                self.valid.cpu().numpy(),
                                                device=self.device)

    # ---- 核心：单日 IC ----

    def _day_ic(self, s: torch.Tensor, t: int) -> torch.Tensor:
        """s [B, V]（该日有效子集分数）→ [B] Spearman IC（分数过中性化链）。"""
        if self.neut is not None:
            s = self.neut.apply(s, t)
        e = self.label_rank[self.date_idx[t], t].unsqueeze(0).expand(s.shape[0], -1)
        return _pearson(avg_rank(s), e, MIN_POOL_IC)

    def ic_series(self, scores, dates=None) -> torch.Tensor:
        """scores [B, N, Tg]（或 [N, Tg]）→ 各日 IC [B, D]（无效日 NaN）。"""
        single = scores.dim() == 2
        sc = scores.unsqueeze(0) if single else scores
        use = self.usable_dates if dates is None else dates
        out = torch.full((sc.shape[0], len(use)), float("nan"), device=self.device)
        for k, t in enumerate(use):
            out[:, k] = self._day_ic(sc[:, self.date_idx[t], t], t)
        return out.squeeze(0) if single else out

    def ics_columns(self, scores, dates) -> torch.Tensor:
        """scores [B, N, len(dates)]，最后一维与 dates（原始日期索引）对齐 → [B, D]。

        训练主循环只为抽中的日期子集执行栈机/保留分数，走此入口省内存。"""
        B = scores.shape[0]
        out = torch.full((B, len(dates)), float("nan"), device=self.device)
        for k, t in enumerate(dates):
            out[:, k] = self._day_ic(scores[:, self.date_idx[t], k], t)
        return out

    # ---- reward 与报告指标 ----

    def reward_on_columns(self, scores, dates, mode=None) -> torch.Tensor:
        """按 REWARD_MODE 聚合各日 IC → reward [B]；全 NaN 行 -10。"""
        mode = mode or REWARD_MODE
        ics = self.ics_columns(scores, dates)
        ok = torch.isfinite(ics)
        cnt = ok.sum(dim=1).clamp(min=1)
        mean = torch.where(ok, ics, torch.zeros_like(ics)).sum(dim=1) / cnt
        if mode == "icir":
            rew = icir(ics)
        elif mode == "ic+icir":
            rew = mean + icir(ics)
        else:                                                # "ic"
            rew = mean
        return torch.where(torch.isfinite(rew), rew, torch.full_like(rew, -10.0))

    def evaluate(self, scores, dates=None) -> torch.Tensor:
        """scores [B, N, Tg] → mean IC [B]（兼容入口，等价 mode='ic'）。"""
        ics = self.ic_series(scores, dates)
        ok = torch.isfinite(ics)
        mean = torch.where(ok, ics, torch.zeros_like(ics)).sum(dim=1) / ok.sum(dim=1).clamp(min=1)
        return torch.where(torch.isfinite(mean), mean, torch.full_like(mean, -10.0))

    def evaluate_columns(self, scores, dates) -> torch.Tensor:
        """evaluate 的日期子列版（Phase 1 兼容入口，mean IC）。"""
        ics = self.ics_columns(scores, dates)
        ok = torch.isfinite(ics)
        mean = torch.where(ok, ics, torch.zeros_like(ics)).sum(dim=1) / ok.sum(dim=1).clamp(min=1)
        return torch.where(torch.isfinite(mean), mean, torch.full_like(mean, -10.0))

    @torch.no_grad()
    def _quintile_means(self, scores, dates) -> torch.Tensor | None:
        """逐日五分位层均池内超额（分数过中性化后分层），对 dates 取均值 → [5]。"""
        rows = []
        for t in dates:
            idx = self.date_idx[t]
            s = scores[idx, t]
            if self.neut is not None:
                s = self.neut.apply(s.unsqueeze(0), t).squeeze(0)
            e = self.excess[idx, t]
            r = torch.argsort(torch.argsort(s)).float()
            q = torch.clamp((r / len(idx) * 5).long(), 0, 4)
            rows.append(torch.stack([e[q == k].mean() for k in range(5)]))
        if not rows:
            return None
        return torch.stack(rows).mean(dim=0)

    @torch.no_grad()
    def layer_spread(self, scores, dates) -> float:
        """五分位 top-bottom 平均池内超额（百分比），对 dates 取均值。

        scores [N, Tg] 单公式全日期分数；排序在原始分数上（中性化不改变
        单调变换下的分层——如启用中性化，此处与 IC 口径一致的秩）。"""
        means = self._quintile_means(scores, dates)
        if means is None:
            return float("nan")
        return float(means[4] - means[0])

    @torch.no_grad()
    def layer_monotonicity(self, scores, dates) -> float:
        """分层单调性：层序(1..5) 与层均超额的 Spearman（穿越层级的秩序性）。"""
        means = self._quintile_means(scores, dates)
        if means is None or not bool(torch.isfinite(means).all()):
            return float("nan")
        order = torch.arange(1, 6, device=means.device, dtype=torch.float32)
        return float(_pearson(order.unsqueeze(0), means.unsqueeze(0), 5)[0])


def evaluator_from_panel(panel=None):
    """便捷构造：从 npz 面板（或磁盘加载）构建 ICEvaluator。"""
    if panel is None:
        from .materialize import load_panel
        panel = load_panel()
    return ICEvaluator(panel["features"], panel["labels"], panel["mask"],
                       industry=panel["industry"], log_mktcap=panel["log_mktcap"])
