# -*- coding: utf-8 -*-
"""公式栈机（自 model_core/vm.py 移植，评估器与（未来的）推理端同源）。

特征 token 压入 [N, T] 张量；算子按元数弹栈计算；结束时栈中恰好剩一个
张量才合法，否则返回 None（训练侧以此给非法公式 -5 分，模型自然学会
只生成合法后缀表达式）。"""
import torch

from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB


class StackVM:
    def __init__(self):
        self.feat_offset = FORMULA_VOCAB.operator_offset
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

    def execute(self, formula_tokens, feat_tensor):
        """feat_tensor: [N, F, T]（或 [F, T]）；返回 [N, T]（或 [T]），非法返回 None。"""
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    if token >= feat_tensor.shape[-2]:
                        return None
                    stack.append(feat_tensor[..., token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity:
                        return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    res = self.op_map[token](*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                return stack[0]
            return None
        except Exception:
            return None
