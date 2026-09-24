# -*- coding: utf-8 -*-
"""采样网格与交易日历（复刻课题契约：锚点固定，start_date 只截断不重锚）。"""
from bisect import bisect_left, bisect_right
from datetime import datetime

from .config import ANCHOR_DATE, REBALANCE_STEP
from .gm_bridge import get_trading_calendar, last_closed_date

_CAL: list | None = None


def trading_days() -> list:
    """锚点前一年至今（+1 年冗余）的交易日升序列表（进程级缓存）。

    多取锚点前一年：特征回看（FEATURE_LOOKBACK_TD）需要锚点前的历史。"""
    global _CAL
    if _CAL is None:
        year = datetime.now().year
        cal = get_trading_calendar(exchange="SHSE",
                                   start_year=int(ANCHOR_DATE[:4]) - 1,
                                   end_year=year + 1)
        _CAL = sorted(x for x in cal["trade_date"][cal["trade_date"] != ""])
    return _CAL


def sampling_grid(end_date: str | None = None) -> list:
    """全课题采样日：>= ANCHOR_DATE 的首个交易日起，每 REBALANCE_STEP 个交易日。"""
    days = trading_days()
    i = bisect_left(days, ANCHOR_DATE)
    grid = days[i::REBALANCE_STEP]
    if end_date is not None:
        grid = [d for d in grid if d <= end_date]
    return grid


def next_trading_days(d: str, n: int) -> list:
    days = trading_days()
    i = bisect_right(days, d)
    return days[i:i + n]


def prev_trading_days(d: str, n: int) -> list:
    days = trading_days()
    i = bisect_left(days, d)
    return days[max(0, i - n):i]


def fetch_start_date() -> str:
    """特征回看起点：锚点前 FEATURE_LOOKBACK_TD 个交易日。"""
    from .config import FEATURE_LOOKBACK_TD
    prev = prev_trading_days(ANCHOR_DATE, FEATURE_LOOKBACK_TD)
    return prev[0] if prev else ANCHOR_DATE


def safe_end_date() -> str:
    """数据窗口右端：最后已收盘日，但不取今天（规避 today-leg 的 current() 调用）。"""
    end = last_closed_date()
    today = datetime.now().strftime("%Y-%m-%d")
    if end >= today:
        prev = prev_trading_days(today, 1)
        end = prev[0] if prev else end
    return end
