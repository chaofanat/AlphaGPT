# -*- coding: utf-8 -*-
"""6 个 A 股截面特征（初始词表原子）：

REV5      5 日对数收益（close/pre_close 链，内嵌除权还原）——反转族
BIAS60    复权价对 60 日均线偏离——中期动量
TURNOVER  20 日平均换手率（快照 turn_rate）——流动性/投机情绪
VRATIO    当日成交额 / 20 日均成交额——量能异常
VOL20     20 日对数收益标准差——已实现波动
EP        1/pe_ttm（负 PE 置 NaN）——价值

日线面板计算 → 抽网格日 → 逐截面 robust-z（MAD，±5 截断，NaN→0）。
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = ("REV5", "BIAS60", "TURNOVER", "VRATIO", "VOL20", "EP")


def daily_features(close_px, pre_close_px, amount_px, turn_px):
    """日线轴量价特征（DataFrame dict，index=symbol, columns=日历日，停牌 NaN）。

    REV5/BIAS60/VOL20 用复权收益链（log(close/pre_close)），跨除权日不失真；
    BIAS60 的分母均线在复权价格序列（比值口径，与缩放无关）上计算。
    内部转置为 [日历日 × 股票] 布局沿默认轴滚动（pandas 3 移除了 axis=1）。"""
    close = close_px.T.astype(float)          # [日历日, 股票]
    pc = pre_close_px.T.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lret = np.log(close / pc)
    lret = lret.where((close > 0) & (pc > 0))

    feats = {}
    feats["REV5"] = lret.rolling(5, min_periods=5).sum().T

    # 上市前 NaN；此后停牌日按 0 收益处理（价格冻结近似），再累乘成复权价序列
    listed = close.notna().cummax()
    adj = np.exp(lret.fillna(0.0).where(listed).cumsum())
    ma60 = adj.rolling(60, min_periods=60).mean()
    feats["BIAS60"] = (adj / ma60 - 1.0).T

    turn = turn_px.T.astype(float)
    feats["TURNOVER"] = turn.rolling(20, min_periods=15).mean().T

    amt = amount_px.T.astype(float)
    amt_ma = amt.rolling(20, min_periods=15).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["VRATIO"] = (amt / amt_ma).where(amt_ma > 0).T

    feats["VOL20"] = lret.rolling(20, min_periods=15).std().T
    return feats


def ep_panel(grid, pe_lookup, index):
    """EP：逐网格日 1/pe_ttm（pe<=0 或缺失 → NaN）。返回 DataFrame [N, Tg]。"""
    cols = {}
    for d in grid:
        pe = pe_lookup.get(d)
        if pe is None:
            pe = pd.Series(dtype=float)
        pe = pe.reindex(index)
        cols[d] = pd.Series(np.where(pe > 0, 1.0 / pe, np.nan), index=index)
    return pd.DataFrame(cols, index=index)


def cross_section_robust_z(df, mask):
    """逐列截面 robust-z：MAD 缩放、±5 截断、NaN→0、mask 外强制 0。"""
    x = df.to_numpy(dtype=np.float64)
    m = mask.to_numpy(dtype=bool)
    out = np.zeros_like(x)
    for j in range(x.shape[1]):
        v = x[m[:, j], j]
        v = v[np.isfinite(v)]
        if len(v) < 20:
            continue
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-9
        col = np.clip((x[:, j] - med) / mad, -5.0, 5.0)
        col = np.nan_to_num(col, nan=0.0, posinf=5.0, neginf=-5.0)
        col[~m[:, j]] = 0.0
        out[:, j] = col
    return out


def build_grid_features(daily_feat, ep_grid_px, grid, mask_df):
    """组装 [N, 6, Tg]：量价特征抽网格列 + EP，raw 保留、z 版供 VM。"""
    index = mask_df.index
    raw_cols = {}
    for name in ("REV5", "BIAS60", "TURNOVER", "VRATIO", "VOL20"):
        df = daily_feat[name]
        raw_cols[name] = df.reindex(index=index, columns=grid)
    raw_cols["EP"] = ep_grid_px.reindex(index=index, columns=grid)

    raw = np.stack([raw_cols[name].to_numpy(dtype=np.float64)
                    for name in FEATURE_NAMES], axis=1)        # [N, 6, Tg]
    z = np.stack([cross_section_robust_z(raw_cols[name], mask_df)
                  for name in FEATURE_NAMES], axis=1)
    return raw, z
