# -*- coding: utf-8 -*-
"""A股截面因子挖掘配置（Phase 1）。"""
import os
from pathlib import Path

import torch

# ---- GMtest 桥（红线：唯一外部代码依赖 D:\\workspace\\GMtest\\utils）----
GMTEST_HOME = Path(os.getenv("GMTEST_HOME", r"D:\workspace\GMtest"))

# ---- 课题契约（复刻 GMtest cross_section 课题，锚点与窗口与其 topic.py 一致）----
ANCHOR_DATE = "2020-01-01"    # 网格锚点：>= 锚点的首个交易日起（固定锚点保证采样日可复现）
HORIZON = 8                   # 持有期：T+1 开盘 → T+8 收盘（交易日）
REBALANCE_STEP = 8            # 采样间隔 = 持有期（仓位不重叠，embargo 天然满足）
NEW_STOCK_DAYS = 120          # 次新阈值（自然日）
MAIN_BOARD = 10100101         # GM board code：沪深主板
FEATURE_LOOKBACK_TD = 150     # 特征回看（交易日，BIAS60 需 60 日 + 冗余）
MIN_POOL_IC = 50              # 单截面参与 IC 计算的最小有效样本数

# ---- 训练超参（在 AlphaGPT model_core 默认上按截面评估成本调整）----
BATCH_SIZE = 2048             # 原版 8192；截面 RankIC 含排序，成本高，先降批
DATES_PER_STEP = 32           # 每步随机抽样的网格日数（REINFORCE 本就随机，日期子样可接受）
TRAIN_STEPS = 1000
MAX_FORMULA_LEN = 12
D_MODEL = 64
ILLEGAL_REWARD = -5.0         # 非法公式（栈机执行失败）
CONSTANT_REWARD = -2.0        # 输出近常量（无截面区分度）
SEED = 42

# ---- Phase 2：契约对齐 ----
TRAIN_END = "2024-01-01"      # 时间外切分：网格日 < TRAIN_END 为训练段，>= 为样本外段
NEUTRALIZE = True             # reward 内启用中性化链（MAD→z→行业+log市值→残差z）
WINSOR_MAD_CLIP = 3.0         # MAD 去极值倍数
REWARD_MODE = "ic"            # "ic" | "icir" | "ic+icir"（训练 reward 口径）
COST_ROUND_TRIP_PCT = 0.30    # 双边换手成本（佣金+印花税+冲击，百分比），净分层用

# ---- Phase 3：出生检验门槛（样本外段一次性判定，搜索全程不可见）----
# 依据见 DESIGN.md：候选不搞 16 选 1，逐条过门槛后全部交付。
BIRTH_MIN_IC = 0.02          # 样本外 IC 均值下限
BIRTH_MIN_TSTAT = 2.0        # 样本外 t 统计量下限（≈ ICIR×√n；日期间隔=持有期，
                             #   无重叠窗口，普通 t 即可，NW 修正近似恒等）
BIRTH_YEARLY_WIN_MIN = 2     # 样本外分年 IC 为正的最少年数（2024/25/26 共 3 年）
BIRTH_MIN_MONO = 0.8         # 五分位单调性（层序×层均收益 Spearman）下限

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 产物与缓存路径 ----
PKG_DIR = Path(__file__).resolve().parent
CACHE_DIR = PKG_DIR / "cache"
OUTPUT_DIR = PKG_DIR / "output"
PANEL_PATH = CACHE_DIR / "panel.npz"
META_PATH = CACHE_DIR / "panel_meta.json"
