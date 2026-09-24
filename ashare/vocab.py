# -*- coding: utf-8 -*-
"""公式词表 = 6 个 A 股特征原子 + 12 个算子（token id：前 6 特征，其后算子）。"""
from dataclasses import dataclass

from .features import FEATURE_NAMES
from .ops import OPS_CONFIG


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple
    operator_names: tuple

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def token_names(self) -> tuple:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)

    def formula_to_names(self, tokens) -> list:
        names = self.token_names
        return [names[int(t)] if 0 <= int(t) < len(names) else f"<{t}>" for t in tokens]


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)
