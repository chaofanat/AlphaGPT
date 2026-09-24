# -*- coding: utf-8 -*-
"""数据链路试金石（集成测试，需要已物化面板 + GMtest 缓存，缺则跳过）：

1. 单票 REV5：面板 features_raw 对 pandas 直算（bars 复权收益链）
2. 单票 t8 标签：面板 labels 对按课题契约逐窗口直算
"""
import numpy as np
import pandas as pd
import pytest

from ashare.config import GMTEST_HOME, PANEL_PATH

pytestmark = pytest.mark.skipif(
    not PANEL_PATH.exists() or not GMTEST_HOME.exists(),
    reason="需先运行 materialize 且 GMTEST_HOME 存在")


def _load():
    data = np.load(PANEL_PATH, allow_pickle=True)
    import json
    meta = json.loads((PANEL_PATH.parent / "panel_meta.json").read_text(encoding="utf-8"))
    return data, meta


def test_rev5_against_direct_bars():
    from ashare import grid as G
    from ashare.gm_bridge import _norm_eob, get_history_with_today
    data, meta = _load()
    symbols, grid_dates = list(data["symbols"]), list(data["grid_dates"])
    feats = data["features_raw"]
    f_idx = meta["feature_names"].index("REV5")

    rng = np.random.default_rng(0)
    for _ in range(3):
        i = int(rng.integers(len(symbols)))
        sym = symbols[i]
        bars = get_history_with_today(sym, meta["fetch_start"], meta["end"],
                                      fields="symbol,eob,close,pre_close")
        bars["eob"] = _norm_eob(bars["eob"])
        bars = bars.drop_duplicates(subset=["eob"], keep="last").set_index("eob")

        for _ in range(3):
            j = int(rng.integers(len(grid_dates)))
            d = pd.Timestamp(grid_dates[j])
            days = [x for x in sorted(bars.index) if d - pd.Timedelta(days=12) < x <= d]
            if len(days) < 5:
                continue
            lret = np.log(bars.loc[days, "close"] / bars.loc[days, "pre_close"])
            lret = lret[(bars.loc[days, "close"] > 0) & (bars.loc[days, "pre_close"] > 0)]
            expect = lret.iloc[-5:].sum()
            got = feats[i, f_idx, j]
            if np.isfinite(expect):
                assert np.isclose(got, float(expect), atol=1e-4), \
                    f"{sym} @ {grid_dates[j]}: {got} vs {expect}"
            return                                       # 一个合格样本即通过（抽样验证）


def test_t8_label_against_direct_window():
    from ashare import grid as G
    from ashare.gm_bridge import _norm_eob, get_history_with_today
    data, meta = _load()
    symbols, grid_dates = list(data["symbols"]), list(data["grid_dates"])
    labels = data["labels"]
    mask = data["mask"]

    rng = np.random.default_rng(1)
    checked = 0
    while checked < 3:
        i = int(rng.integers(len(symbols)))
        j = int(rng.integers(len(grid_dates) - 2))       # 留前向窗口
        if not mask[i, j] or not np.isfinite(labels[i, j]):
            continue
        sym, T = symbols[i], grid_dates[j]
        nxt = G.next_trading_days(T, 8)
        if len(nxt) < 8:
            continue
        bars = get_history_with_today(sym, T, nxt[-1],
                                      fields="symbol,eob,open,close,pre_close")
        bars["eob"] = _norm_eob(bars["eob"])
        bars = bars.drop_duplicates(subset=["eob"], keep="last").set_index("eob")
        win = bars[(bars.index > pd.Timestamp(T)) & (bars.index <= pd.Timestamp(nxt[-1]))]
        if win.empty or str(win.index[0].date()) != nxt[0]:
            # T+1 无 bar（停牌买不进）→ 面板标签应为 NaN
            assert not np.isfinite(labels[i, j]), f"{sym} @ {T}: T+1 停牌但标签非 NaN"
            checked += 1
            continue
        entry = float(win["open"].iloc[0])
        ratios = win["close"].iloc[1:] / win["pre_close"].iloc[1:]
        growth = (float(win["close"].iloc[0]) / entry) * float(ratios.prod()) \
            if (ratios.notna().all() and np.isfinite(ratios).all() and (ratios > 0).all()) \
            else float(win["close"].iloc[-1]) / entry
        expect = (growth - 1.0) * 100.0
        assert np.isclose(labels[i, j], expect, atol=1e-4), \
            f"{sym} @ {T}: {labels[i, j]} vs {expect}"
        checked += 1
