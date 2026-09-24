# -*- coding: utf-8 -*-
"""面板物化：GMtest 缓存 → ashare/cache/panel.npz（训练/评估唯一数据入口）。

一次物化产出（Phase 1 口径）：
- features     [N, 6, Tg] float32  逐截面 robust-z 后的网格日特征（VM 输入）
- features_raw [N, 6, Tg] float32  原始值（试金石对账 / Phase 2 中性化起点）
- labels       [N, Tg]    float32  t8 百分比收益（NaN = 丢弃）
- mask         [N, Tg]    bool     课题宇宙（可交易池）
- industry     [N, Tg]    object   申万一级行业名（Phase 2 中性化用）
- log_mktcap   [N, Tg]    float32  log 总市值（Phase 2 中性化用）
- symbols      [N]        object / grid_dates [Tg] object
"""
import json

import numpy as np
import pandas as pd

from . import grid as G
from .config import CACHE_DIR, META_PATH, PANEL_PATH
from .features import FEATURE_NAMES, build_grid_features, daily_features, ep_panel
from .gm_bridge import (_norm_eob, get_history_with_today,
                        get_industry_sw_cached, get_mktcap_cached,
                        get_valuation_cached, load_snapshots)
from .labels import label_panel
from .universe import build_pools

BAR_FIELDS = "symbol,eob,open,high,low,close,pre_close,volume,amount"  # ⊆ BAR_COLS，命中磁盘缓存


def _pivot(bars, value, index, columns):
    df = bars.pivot(index="symbol", columns="eob", values=value)
    return df.reindex(index=index, columns=columns)


def materialize(end_date=None, verbose=True):
    def log(msg):
        if verbose:
            print(msg, flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    end = end_date or G.safe_end_date()
    grid = G.sampling_grid(end)
    fetch_start = G.fetch_start_date()
    all_days = [d for d in G.trading_days() if fetch_start <= d <= end]
    log(f"网格: {grid[0]} ~ {grid[-1]}（{len(grid)} 个采样日）｜日线窗口: {fetch_start} ~ {end}（{len(all_days)} 日）")
    if len(grid) < 10:
        raise RuntimeError(f"采样日过少({len(grid)})，检查日历与窗口")

    # ---- 快照：换手率日面板 + 宇宙过滤 ----
    log("Step 1/5: 加载快照（turn_rate / ST / 次新 / 板块）...")
    snapshots = load_snapshots(all_days)
    if snapshots.empty:
        raise RuntimeError("快照数据为空：请确认 GMTEST_HOME 与其磁盘缓存")
    turn_px = snapshots.pivot(index="symbol", columns="trade_date", values="turn_rate")

    # 预池（不含 bar 双保险）：用于圈定 bar 拉取的股票并集
    from .config import MAIN_BOARD, NEW_STOCK_DAYS
    from datetime import datetime, timedelta
    union = set()
    for d in grid:
        snap = snapshots[snapshots["trade_date"] == d]
        threshold = (datetime.strptime(d, "%Y-%m-%d")
                     - timedelta(days=NEW_STOCK_DAYS)).date()
        keep = snap[(snap["board"] == MAIN_BOARD)
                    & (~snap["is_st"].astype(bool))
                    & (~snap["sec_name"].str.contains("ST|退", na=False))]
        union |= set(keep.loc[keep["listed_date"].dt.date < threshold, "symbol"])
    symbols = sorted(union)
    log(f"并集股票池: {len(symbols)} 只（网格日快照过滤后的 union）")

    # ---- 日线（全部走 GMtest 磁盘缓存；缺失段自动网络补拉）----
    log("Step 2/5: 拉取日线面板（OHLC + pre_close + amount）...")
    bars = get_history_with_today(symbols, fetch_start, end, fields=BAR_FIELDS)
    if bars is None or bars.empty:
        raise RuntimeError("日线面板为空：请确认 GMtest 缓存/网络")
    bars["eob"] = _norm_eob(bars["eob"])
    bars = bars.drop_duplicates(subset=["symbol", "eob"], keep="last")

    open_px = _pivot(bars, "open", symbols, all_days)
    close_px = _pivot(bars, "close", symbols, all_days)
    pre_close_px = _pivot(bars, "pre_close", symbols, all_days)
    amount_px = _pivot(bars, "amount", symbols, all_days)
    turn_sub = turn_px.reindex(index=symbols, columns=all_days)

    # ---- 最终宇宙（含当日 bar / 一字板过滤）----
    log("Step 3/5: 构建课题宇宙（主板/剔ST/剔次新/剔停牌/剔一字板）...")
    pools = build_pools(grid, bars, snapshots[snapshots["trade_date"].isin(grid)])
    grid = [d for d in grid if d in pools]          # 空池日剔除
    mask = pd.DataFrame(
        {d: pd.Series(symbols, index=symbols).isin(pools[d]) for d in grid},
        index=symbols)
    avg = int(mask.to_numpy().sum() / max(1, len(grid)))
    log(f"采样池: {len(grid)} 个采样日, 池均 {avg} 只")

    date_index = {d: i for i, d in enumerate(all_days)}

    # ---- 特征 ----
    log("Step 4/5: 计算特征（日线面板 → 网格日 → 截面 robust-z）...")
    daily = daily_features(close_px, pre_close_px, amount_px, turn_sub)
    pe_lookup = {}
    for d in grid:
        val = get_valuation_cached(d, fields=["pe_ttm"])
        pe_lookup[d] = None if val is None else val["pe_ttm"]
    ep_px = ep_panel(grid, pe_lookup, symbols)
    raw, z = build_grid_features(daily, ep_px, grid, mask)

    # ---- 标签 + Phase 2 附带通道 ----
    log("Step 5/5: 计算 t8 标签与 Phase 2 通道（行业 / log 市值）...")
    labels = label_panel(open_px, close_px, pre_close_px, grid, date_index)

    industry = np.full((len(symbols), len(grid)), "", dtype=object)
    log_mktcap = np.full((len(symbols), len(grid)), np.nan, dtype=np.float64)
    for j, d in enumerate(grid):
        ind = get_industry_sw_cached(d)
        if ind is not None:
            aligned = ind.reindex(symbols)
            industry[:, j] = aligned.fillna("").to_numpy(dtype=object)
        mv = get_mktcap_cached(d)
        if mv is not None:
            vals = mv.reindex(symbols).to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore"):
                log_mktcap[:, j] = np.where(vals > 0, np.log(vals), np.nan)

    valid_labels = np.isfinite(labels) & mask.to_numpy()
    log(f"标签有效率: {valid_labels.sum() / max(1, mask.to_numpy().sum()):.1%}")

    np.savez_compressed(
        PANEL_PATH,
        features=z.astype(np.float32),
        features_raw=raw.astype(np.float32),
        labels=labels.astype(np.float32),
        mask=mask.to_numpy(),
        industry=industry,
        log_mktcap=log_mktcap.astype(np.float32),
        symbols=np.array(symbols, dtype=object),
        grid_dates=np.array(grid, dtype=object),
    )
    meta = {
        "fetch_start": fetch_start,
        "end": end,
        "grid": grid,
        "symbols": len(symbols),
        "feature_names": list(FEATURE_NAMES),
        "pool_avg": avg,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"面板已保存: {PANEL_PATH}")
    return meta


def load_panel():
    """加载物化面板（训练/评估入口）。"""
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"{PANEL_PATH} 不存在：先运行 materialize")
    data = np.load(PANEL_PATH, allow_pickle=True)
    return data
