# -*- coding: utf-8 -*-
"""因子定义包导出（Phase 3③，契约见 DESIGN.md §4）。

交付形态：定义包而非静态值面板——值面板由消费方按定义自动生成才能随
日期滚动。一次导出产出：

    packages/
    ├── reference_panel.npz          共享参考面板（VM 输入特征为**全网格**
    │                                  [N,6,Tg]——时序算子依赖全历史；
    │                                  mask/行业/市值只取参考日截面）
    └── ashare_formula_<hash8>/
        ├── definition.json          公式 + 词表指纹 + 原子口径 + 算子语义
        │                              + 中性化口径 + 出生检验溯源
        └── expected.npz             本公式在参考日上的标准执行结果
                                       （原始分数 + 池内中性化分数）

对账协议（消费侧）：按 definition.json 实现原子与栈机（或直接移植
vm.py/preprocess.py 的对应逻辑），读 reference_panel.npz 执行，与
expected.npz 断言 allclose(atol=1e-4)。导出时本侧用同一机制自检，保证
包自洽（防序列化/维度错误）；原子层数值对账（raw 行情→特征）为 v2 项。
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .config import (BIRTH_MIN_IC, BIRTH_MIN_MONO, BIRTH_MIN_TSTAT,
                     BIRTH_YEARLY_WIN_MIN, MIN_POOL_IC, OUTPUT_DIR, TRAIN_END,
                     WINSOR_MAD_CLIP)
from .features import FEATURE_NAMES
from .ops import OPS_CONFIG
from .preprocess import CrossSectionNeutralizer
from .vm import StackVM

REF_DATES = 3            # 参考面板网格日数（可用日集的首/中/末，确定性）
CONSISTENCY_ATOL = 1e-4  # 双侧一致性容差


def _formula_hash(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()[:8]


def _load_candidates():
    data = json.loads((OUTPUT_DIR / "formula_candidates.json")
                      .read_text(encoding="utf-8"))
    return data["candidates"]


def _pick_ref_dates(usable: list) -> list:
    if len(usable) <= REF_DATES:
        return list(usable)
    mid = usable[len(usable) // 2]
    return [usable[0], mid, usable[-1]]


def _atom_spec() -> dict:
    shared = ("逐网格日截面：MAD 去极值（中位数/MAD，±5 截断）→ 缺失填 0 → "
              "池外置 0；原子先在日线上计算，再抽取网格日")
    return {
        "layout": "日线面板 [股票, 交易日] → 抽网格日（间隔 8 交易日）",
        "_shared_postprocess": shared,
        "REV5": "5 日对数收益：sum(log(close/pre_close), 最近 5 根日线)；"
                "要求 close>0 且 pre_close>0（close/pre_close 链内嵌除权还原）",
        "BIAS60": "复权价格序列 adj=exp(cumsum(log(close/pre_close))) 相对 "
                  "60 日均线的偏离 adj/MA60(adj)-1；min_periods=60",
        "TURNOVER": "快照 turn_rate 的 20 日均值；min_periods=15",
        "VRATIO": "当日成交额 amount / 20 日均成交额；min_periods=15，分母>0",
        "VOL20": "log(close/pre_close) 的 20 日标准差；min_periods=15",
        "EP": "1/pe_ttm（估值接口，逐网格日）；pe<=0 或缺失 → NaN",
    }


def _ops_spec() -> dict:
    return {
        "layout": "算子作用在 [N 股票, T 网格日] 张量上（时序算子的 1 步 = 8 交易日）",
        "execution": "后缀表达式栈机：特征 token 压入 [N,T] 行；算子按元数弹栈"
                     "（弹出的逆序为参数）；结果逐元素 NaN/Inf 清洗"
                     "（nan→0, +inf→1, -inf→-1）；结束时栈须恰剩 1 个张量",
        "OPS": [{"name": n, "arity": a} for n, _, a in OPS_CONFIG],
        "_semantics": {
            "GATE": "mask=(cond>0).float(); mask*x+(1-mask)*y（三元）",
            "JUMP": "relu(zscore_time(x) - 3)，zscore 沿网格日轴按股票",
            "DECAY": "x + 0.8*lag1(x) + 0.6*lag2(x)",
            "DELAY1": "沿网格日轴右移 1 步，左补零",
            "MAX3": "max(x, lag1(x), lag2(x))",
        },
    }


def _neut_spec() -> dict:
    return {
        "chain": "MAD 去极值(±3 MAD) → z-score → 申万一级行业哑变量 + "
                 "z 后 log 总市值 + 截距的截面回归取残差 → 残差 z-score",
        "degenerate_guard": "残差 std < 1e-4 的行返回零向量（纯 beta 分数被"
                            "完全剥离后不得放大投影噪声）",
        "subsets": "评估用 池∩标签有效 子集；滚动出值用当日池（mask）——"
                   "本包 expected.npz 的 neutralized 为池口径",
        "params": {"winsor_mad_clip": WINSOR_MAD_CLIP},
    }


def _definition(cand: dict, name: str) -> dict:
    ids = cand["formula"]
    return {
        "name": name,
        "kind": "ashare_mined_factor_v1",
        "formula": {"ids": [int(i) for i in ids],
                    "tokens": [FEATURE_NAMES[int(i)] if int(i) < len(FEATURE_NAMES)
                               else OPS_CONFIG[int(i) - len(FEATURE_NAMES)][0]
                               for i in ids]},
        "vocab": {"feature_names": list(FEATURE_NAMES),
                  "operator_offset": len(FEATURE_NAMES),
                  "operators": [{"name": n, "arity": a} for n, _, a in OPS_CONFIG]},
        "atoms": _atom_spec(),
        "vm_semantics": _ops_spec(),
        "neutralization": _neut_spec(),
        "provenance": {
            "train_end": TRAIN_END,
            "gate": cand["gate"],
            "metrics": {"train": cand["train"], "test": cand["test"]},
            "gate_thresholds": {"min_ic": BIRTH_MIN_IC, "min_tstat": BIRTH_MIN_TSTAT,
                                "yearly_win_min": BIRTH_YEARLY_WIN_MIN,
                                "min_mono": BIRTH_MIN_MONO},
        },
        "consistency": {
            "protocol": "读 ../reference_panel.npz，按本文件实现原子与栈机，"
                        "对 features 执行公式，与 expected.npz 的 raw_scores / "
                        "neutralized 断言 allclose",
            "atol": CONSISTENCY_ATOL,
        },
    }


def export_packages(candidates=None, panel=None, out_dir=None, limit=None):
    """导出全部过门槛候选的定义包；返回 [(包名, formula_ids)]。"""
    out_dir = Path(out_dir or (OUTPUT_DIR / "packages"))
    candidates = [c for c in (candidates if candidates is not None else _load_candidates())
                  if c["gate"]["passed"]]
    if limit:
        candidates = candidates[:limit]
    if not candidates:
        raise RuntimeError("无过门槛候选：先训练或检查 formula_candidates.json")
    if panel is None:
        from .materialize import load_panel
        panel = load_panel()

    mask = np.asarray(panel["mask"])
    usable = [t for t in range(mask.shape[1]) if mask[:, t].sum() >= MIN_POOL_IC]
    ref_idx = _pick_ref_dates(usable)
    grid = list(panel["grid_dates"])

    cpu = torch.device("cpu")
    features = torch.tensor(np.asarray(panel["features"]), dtype=torch.float32,
                            device=cpu)
    neut = CrossSectionNeutralizer(panel["industry"], panel["log_mktcap"],
                                   mask, device=cpu)

    out_dir.mkdir(parents=True, exist_ok=True)
    # 特征存全网格（时序算子依赖全历史：DELAY1/DECAY/MAX3 需前两期，
    # JUMP 的时序 zscore 用整条网格轴——切片上重执行无法复现）；
    # mask/行业/市值只需参考日截面（中性化逐日独立，无历史依赖）。
    np.savez_compressed(
        out_dir / "reference_panel.npz",
        grid_dates=panel["grid_dates"],
        ref_date_indices=np.array(ref_idx, dtype=np.int64),
        symbols=panel["symbols"],
        features=features.numpy(),
        mask=mask[:, ref_idx],
        industry=panel["industry"][:, ref_idx],
        log_mktcap=panel["log_mktcap"][:, ref_idx],
        feature_names=np.array(FEATURE_NAMES, dtype=object),
    )

    vm = StackVM()
    exported = []
    for cand in candidates:
        ids = [int(i) for i in cand["formula"]]
        scores = vm.execute(ids, features)                  # [N, Tg]
        if scores is None:
            print(f"⚠️ 跳过（栈机执行失败）: {ids}")
            continue
        raw = scores[:, ref_idx].numpy().astype(np.float32)
        neutralized = np.full(raw.shape, np.nan, dtype=np.float32)
        for j, t in enumerate(ref_idx):
            idx = torch.tensor(np.nonzero(mask[:, t])[0])
            neutralized[idx.numpy(), j] = neut.apply(
                scores[idx, t].unsqueeze(0), t)[0].numpy()

        name = f"ashare_formula_{_formula_hash(ids)}"
        pkg = out_dir / name
        pkg.mkdir(exist_ok=True)
        (pkg / "definition.json").write_text(
            json.dumps(_definition(cand, name), ensure_ascii=False, indent=2),
            encoding="utf-8")
        np.savez_compressed(pkg / "expected.npz",
                            raw_scores=raw, neutralized=neutralized)
        _self_check(pkg)
        exported.append((name, ids))

    print(f"✓ 导出 {len(exported)} 个定义包 → {out_dir}")
    # 剪除历史导出残留（只清本次未产出的公式包，参考面板在下方统一重写）
    exported_names = {name for name, _ in exported}
    pruned = 0
    for d in out_dir.glob("ashare_formula_*"):
        if d.name not in exported_names:
            shutil.rmtree(d)
            pruned += 1
    if pruned:
        print(f"  已清理 {pruned} 个过期残留包")
    print(f"  参考网格日: {[grid[t] for t in ref_idx]}（池口径中性化，自检通过）")
    return exported


def _self_check(pkg: Path):
    """包自洽校验：仅凭包内文件重建输入并执行，断言与 expected 一致。"""
    ref = np.load(pkg.parent / "reference_panel.npz", allow_pickle=True)
    exp = np.load(pkg / "expected.npz")
    ids = json.loads((pkg / "definition.json").read_text(encoding="utf-8"))["formula"]["ids"]
    feats = torch.tensor(ref["features"], dtype=torch.float32)   # 全网格特征
    res = StackVM().execute(ids, feats)
    assert res is not None
    got = res.numpy()[:, ref["ref_date_indices"]]
    assert np.allclose(got, exp["raw_scores"], atol=CONSISTENCY_ATOL), "raw 对账失败"
    # 中性化逐日独立：用参考日截面（包内原料）重建设计矩阵
    neut = CrossSectionNeutralizer(ref["industry"], ref["log_mktcap"],
                                   ref["mask"], device=torch.device("cpu"))
    for j in range(got.shape[1]):
        idx = torch.tensor(np.nonzero(ref["mask"][:, j])[0])
        neu = neut.apply(torch.tensor(got[idx, j]).unsqueeze(0), j)[0].numpy()
        assert np.allclose(neu, exp["neutralized"][idx.numpy(), j],
                           atol=CONSISTENCY_ATOL), f"neutralized 对账失败 @col{j}"
