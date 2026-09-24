# -*- coding: utf-8 -*-
"""与 GMtest 课题缓存对账（只读数据，不 import 其代码）：

比较本地物化宇宙 与 GMtest cross_section 课题自己的采样池缓存
（models/data/cross_section_stocks_20200101_*.json，{date: [symbols]}）
的重合度——两者锚点/步长契约相同，重合度应很高（>95% 量级；
差异主要来自快照缓存补拉时点与一字板判定的 bar 完整性）。

用法：python -m ashare.scripts.validate_vs_gmtest [gmtest_json_path]
"""
import json
import sys
from pathlib import Path

import numpy as np

from ashare.config import GMTEST_HOME, PANEL_PATH


def validate(gmtest_json=None):
    if gmtest_json is None:
        candidates = sorted((GMTEST_HOME / "models" / "data").glob(
            "cross_section_stocks_20200101_*.json"))
        if not candidates:
            print(f"未找到 GMtest 采样池缓存（{GMTEST_HOME / 'models' / 'data'}）")
            return
        gmtest_json = candidates[-1]
    theirs = json.loads(Path(gmtest_json).read_text(encoding="utf-8"))

    data = np.load(PANEL_PATH, allow_pickle=True)
    symbols, grid_dates, mask = (list(data["symbols"]), list(data["grid_dates"]),
                                 data["mask"])
    sym_idx = {s: i for i, s in enumerate(symbols)}

    overlaps, sizes_mine, sizes_theirs = [], [], []
    for d, pool in theirs.items():
        if d not in grid_dates:
            continue
        j = grid_dates.index(d)
        mine = set(symbols[k] for k in np.nonzero(mask[:, j])[0])
        t = set(pool)
        if not t:
            continue
        overlaps.append(len(mine & t) / len(t | mine) if (t | mine) else 1.0)
        sizes_mine.append(len(mine))
        sizes_theirs.append(len(t))

    if not overlaps:
        print("无重叠采样日：检查锚点/步长是否与课题一致（2020-01-01 锚 / 8td）")
        return
    print(f"对账文件: {gmtest_json}")
    print(f"重叠采样日: {len(overlaps)}")
    print(f"Jaccard 重合度: mean={np.mean(overlaps):.4f} "
          f"min={np.min(overlaps):.4f} p10={np.percentile(overlaps, 10):.4f}")
    print(f"池规模: 本地均值 {np.mean(sizes_mine):.0f} vs 课题均值 {np.mean(sizes_theirs):.0f}")


if __name__ == "__main__":
    validate(sys.argv[1] if len(sys.argv) > 1 else None)
