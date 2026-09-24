# -*- coding: utf-8 -*-
"""StackVM 手工期望单测（纯本地，不依赖 GMtest）。"""
import numpy as np
import torch

from ashare.vocab import FORMULA_VOCAB as VOCAB
from ashare.vm import StackVM

N, F, T = 2, len(VOCAB.feature_names), 3
torch.manual_seed(0)
FEAT = torch.randn(N, F, T)


def tok(name):
    return VOCAB.token_names.index(name)


def test_single_feature():
    vm = StackVM()
    out = vm.execute([tok("REV5")], FEAT)
    assert out is not None
    assert torch.allclose(out, FEAT[:, VOCAB.feature_names.index("REV5"), :])


def test_binary_op_add():
    vm = StackVM()
    a, b = tok("REV5"), tok("BIAS60")
    out = vm.execute([a, b, tok("ADD")], FEAT)
    assert torch.allclose(out, FEAT[:, 0, :] + FEAT[:, 1, :])


def test_arity_violation_returns_none():
    vm = StackVM()
    assert vm.execute([tok("ADD")], FEAT) is None            # 栈空弹 2
    assert vm.execute([tok("REV5"), tok("GATE")], FEAT) is None  # GATE 需 3 元


def test_stack_leftover_returns_none():
    vm = StackVM()
    assert vm.execute([tok("REV5"), tok("BIAS60")], FEAT) is None  # 结束时栈剩 2


def test_invalid_token_returns_none():
    vm = StackVM()
    assert vm.execute([VOCAB.size + 99], FEAT) is None


def test_delay1_zero_pads_grid_axis():
    vm = StackVM()
    out = vm.execute([tok("REV5"), tok("DELAY1")], FEAT)
    expect = torch.zeros_like(FEAT[:, 0, :])
    expect[:, 1:] = FEAT[:, 0, :-1]
    assert torch.allclose(out, expect)


def test_gate_selects_by_condition_sign():
    vm = StackVM()
    cond = torch.tensor([[1.0, -2.0, 0.5]])
    x = torch.tensor([[10.0, 10.0, 10.0]])
    y = torch.tensor([[20.0, 20.0, 20.0]])
    # [1, 4, T]：位置 1=cond, 2=x, 3=y（词表 6 特征，id 0~3 合法；GATE 需恰 3 操作数）
    feat = torch.stack([torch.zeros_like(cond), cond, x, y], dim=1)
    out = vm.execute([1, 2, 3, tok("GATE")], feat)
    assert out is not None
    assert torch.allclose(out, torch.where(cond > 0, x, y))


def test_2d_layout_for_single_stock():
    vm = StackVM()
    out = vm.execute([tok("REV5")], FEAT[0])                 # [F, T] → [T]
    assert torch.allclose(out, FEAT[0, VOCAB.feature_names.index("REV5"), :])
