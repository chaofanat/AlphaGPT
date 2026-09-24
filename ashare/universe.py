# -*- coding: utf-8 -*-
"""课题宇宙（本地复刻 GMtest cross_section 契约，不 import 其 models 代码）：

主板(board==MAIN_BOARD) → 剔 ST(is_st 或名称含 ST|退) → 剔次新(上市不满
NEW_STOCK_DAYS 自然日) → 剔当日停牌(当日无 bar 双保险) → 剔当日一字板。
"""
from datetime import datetime, timedelta

import pandas as pd

from .config import MAIN_BOARD, NEW_STOCK_DAYS
from .gm_bridge import load_snapshots


def build_pools(grid, bars, snapshots):
    """逐网格日构建可交易池。

    bars: get_history_with_today 返回的日线 DataFrame（eob 已归一化到 naive 日）
    snapshots: load_snapshots(grid) 快照（含 trade_date/board/is_st/sec_name/listed_date）
    返回 {date: set(symbol)}，空池日跳过。
    """
    pools = {}
    for d in grid:
        day_snap = snapshots[snapshots["trade_date"] == d]
        if day_snap.empty:
            continue
        threshold = (datetime.strptime(d, "%Y-%m-%d")
                     - timedelta(days=NEW_STOCK_DAYS)).date()
        keep = day_snap[
            (day_snap["board"] == MAIN_BOARD)
            & (~day_snap["is_st"].astype(bool))
            & (~day_snap["sec_name"].str.contains("ST|退", na=False))
        ]
        pool = set(keep.loc[keep["listed_date"].dt.date < threshold, "symbol"])

        day = bars[bars["eob"] == pd.Timestamp(d)]
        pool &= set(day["symbol"])          # 当日停牌双保险（快照已剔，无 bar 再剔）
        one_word = day[
            (day["open"] == day["high"])
            & (day["high"] == day["low"])
            & (day["low"] == day["close"])]
        pool -= set(one_word["symbol"])     # 当日一字板买不进
        if pool:
            pools[d] = pool
    return pools
