# -*- coding: utf-8 -*-
"""GMtest 数据层唯一 import 点（红线执行处）。

红线：仅 D:\\workspace\\GMtest\\utils 下的代码可作为外部代码导入，其余一切
逻辑在本仓库本地实现。任何模块需要 GMtest 数据时统一从这里转引，禁止在
其他模块直接 `import utils.*`（tests/test_redline.py 源码扫描强制此约定）。

注意：utils 的磁盘缓存锚定在 GMtest 根目录的 models/data/ 下（随 utils 包
位置解析），因此读取/补写都发生在 GMtest 仓库内——已缓存的 5000+ 只股票
日线、快照、估值、行业、市值 parquet 直接复用；缓存缺失走 gm SDK 网络拉取
（GM_TOKEN 有兜底），历史区间不触发掘金终端依赖。
"""
import os
import sys

from .config import GMTEST_HOME

# gm SDK（经 utils 传递导入）会在 C++ 层劫持/关闭 fd 1，使 stdout 静默丢失。
# 在导入前保存原 fd，导入后把 Python stdout 改绑到保存的 fd 上——之后无论
# gm 如何动 fd 1，我们的 print 都走原管道。
_SAVED_STDOUT_FD = os.dup(1)

if str(GMTEST_HOME) not in sys.path:
    sys.path.insert(0, str(GMTEST_HOME))

# ---- 统一转引（调用点保持一行 import）----
from utils.gm_data import get_history_with_today, last_closed_date, _norm_eob  # noqa: E402,F401
from utils.market_data import (load_snapshots,  # noqa: E402,F401
                               get_industry_sw_cached, get_mktcap_cached)
from utils.trading_day import get_trading_calendar  # noqa: E402,F401
from utils.fundamental_data import get_valuation_cached  # noqa: E402,F401

# 恢复可见 stdout（绑到保存的 fd，绕开 gm 对 fd 1 的劫持）
os.dup2(_SAVED_STDOUT_FD, 1)
sys.stdout = open(_SAVED_STDOUT_FD, "w", encoding="utf-8", buffering=1, closefd=False)
