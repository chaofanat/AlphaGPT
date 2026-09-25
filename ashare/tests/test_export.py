# -*- coding: utf-8 -*-
"""定义包导出试金石（纯本地合成数据）：

1. 未过出生门槛的候选不导出；过门槛的全部导出
2. 包自洽：definition.json + reference_panel.npz + expected.npz 齐全，
   导出自检（凭包内文件重执行并断言一致）内建于 export_packages
3. 语义抽查：单特征公式 [REV5] 的 raw_scores 恰为参考面板特征第 0 通道
"""
import json

import numpy as np

from ashare.export import export_packages
from ashare.features import FEATURE_NAMES

N, T, F = 90, 6, 6


def _synthetic():
    rng = np.random.default_rng(0)
    return {
        "features": rng.normal(size=(N, F, T)).astype(np.float32),
        "labels": rng.normal(size=(N, T)).astype(np.float32),
        "mask": np.ones((N, T), dtype=bool),
        "industry": np.tile(rng.choice(["银行", "白酒", "半导体", "钢铁"], size=N)[:, None],
                            (1, T)),
        "log_mktcap": rng.uniform(20.0, 24.0, size=(N, T)),
        "symbols": np.array([f"TEST.{i:04d}" for i in range(N)], dtype=object),
        "grid_dates": np.array([f"2024-0{m}-15" for m in range(1, 7)], dtype=object),
    }


def _candidates():
    return [
        {"formula": [0], "gate": {"passed": True, "reasons": []},
         "train": {"ic": 0.05}, "test": {"ic": 0.05}},
        {"formula": [1, 2, 6], "gate": {"passed": True, "reasons": []},
         "train": {"ic": 0.04}, "test": {"ic": 0.04}},
        {"formula": [3], "gate": {"passed": False, "reasons": ["t 1.5<2.0"]},
         "train": {"ic": 0.01}, "test": {"ic": 0.01}},
    ]


def test_export_filters_and_package_shape(tmp_path):
    exported = export_packages(candidates=_candidates(), panel=_synthetic(),
                               out_dir=tmp_path)
    assert len(exported) == 2, "未过门槛的候选必须被排除"
    assert [ids for _, ids in exported] == [[0], [1, 2, 6]]

    ref = np.load(tmp_path / "reference_panel.npz", allow_pickle=True)
    assert ref["features"].shape == (N, F, T)          # 特征为全网格（时序算子需全历史）
    assert ref["mask"].shape == (N, 3)                 # 参考日截面：首中末 3 日
    assert len(ref["ref_date_indices"]) == 3
    assert list(ref["feature_names"]) == list(FEATURE_NAMES)

    for name, ids in exported:
        d = json.loads((tmp_path / name / "definition.json").read_text(encoding="utf-8"))
        assert d["formula"]["ids"] == ids
        assert {"atoms", "vm_semantics", "neutralization",
                "provenance", "consistency"} <= set(d)
        exp = np.load(tmp_path / name / "expected.npz")
        assert exp["raw_scores"].shape == (N, 3)
        assert np.isfinite(exp["raw_scores"]).all()


def test_single_feature_formula_semantics(tmp_path):
    exported = export_packages(candidates=_candidates()[:1], panel=_synthetic(),
                               out_dir=tmp_path)
    name = exported[0][0]
    ref = np.load(tmp_path / "reference_panel.npz", allow_pickle=True)
    exp = np.load(tmp_path / name / "expected.npz")
    # 单特征公式：raw 分数 = 全网格特征第 0 通道在参考日上的切片（无转置/错位）
    ref_idx = np.load(tmp_path / "reference_panel.npz", allow_pickle=True)["ref_date_indices"]
    assert np.allclose(exp["raw_scores"],
                       ref["features"][:, 0, :][:, ref_idx], atol=1e-6)
    # 全池 mask → 中性化分数处处有限
    assert np.isfinite(exp["neutralized"]).all()
