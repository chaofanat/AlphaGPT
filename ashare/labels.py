# -*- coding: utf-8 -*-
"""t8 标签（本地复刻课题契约）：

T+1 开盘买入 → T+8 收盘卖出；T+1 无 bar（停牌买不进）→ 丢弃；T+8 无 bar
（持有期内停牌）→ 用窗口内最后一根收盘价（冻结价近似）。收益为复权口径：
跨日段按 close/pre_close 链累乘（pre_close 由交易所按除权除息公布）；
pre_close 异常（缺失/≤0/inf）回退不复权口径。
"""
import numpy as np

from .config import HORIZON


def label_panel(open_px, close_px, pre_close_px, grid, date_index):
    """向量化计算 [N, Tg] 标签（百分比收益，NaN = 丢弃）。

    open_px / close_px / pre_close_px: DataFrame index=symbol, columns=日历日
    （停牌日为 NaN）；grid: 采样日列表；date_index: {date: 列位置}。
    """
    o = open_px.to_numpy(dtype=np.float64)
    c = close_px.to_numpy(dtype=np.float64)
    pc = pre_close_px.to_numpy(dtype=np.float64)
    n, n_days = o.shape
    out = np.full((n, len(grid)), np.nan, dtype=np.float64)

    for j, d in enumerate(grid):
        t = date_index[d]
        lo, hi = t + 1, t + 1 + HORIZON      # 窗口 [T+1, T+8]
        if hi > n_days:
            continue                          # 前向窗口不足（最近的采样日）
        ow = o[:, lo:hi]
        cw = c[:, lo:hi]
        pw = pc[:, lo:hi]

        t1_ok = np.isfinite(ow[:, 0]) & (ow[:, 0] > 0)   # T+1 必须有 bar
        valid_c = np.isfinite(cw)
        any_c = valid_c.any(axis=1)
        # 出场价：窗口内最后一根有效收盘（冻结价近似）
        last_pos = np.where(any_c,
                            valid_c.shape[1] - 1 - np.argmax(valid_c[:, ::-1], axis=1), 0)
        exit_c = cw[np.arange(n), last_pos]
        # 复权链：ratios = close_i / pre_close_i（i 为窗口内第 2 根起）；任一异常 → 回退
        ratios = cw[:, 1:] / pw[:, 1:]
        chain_ok = (np.isfinite(ratios).all(axis=1)
                    & (ratios > 0).all(axis=1)
                    & np.isfinite(cw[:, 0]) & (cw[:, 0] > 0))
        safe_ratios = np.where(np.isfinite(ratios), ratios, 1.0)
        growth_chain = (cw[:, 0] / ow[:, 0]) * np.prod(safe_ratios, axis=1)
        growth_raw = exit_c / ow[:, 0]
        growth = np.where(chain_ok, growth_chain, growth_raw)

        ret = (growth - 1.0) * 100.0
        ok = t1_ok & any_c & np.isfinite(ret)
        out[ok, j] = ret[ok]
    return out
