# -*- coding: utf-8 -*-
"""评估器试金石（纯本地合成数据，不依赖 GMtest）：

1. avg_rank 与 scipy.stats.rankdata（并列平均）一致
2. ICEvaluator 的 IC 与逐日 scipy.stats.spearmanr 直算一致
3. 植入公式试金石：VM 执行单特征公式的 reward == 该特征逐日 Spearman 均值
"""
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from ashare.config import MIN_POOL_IC
from ashare.evaluator import ICEvaluator, avg_rank
from ashare.vocab import FORMULA_VOCAB as VOCAB
from ashare.vm import StackVM

N, T, F = 60, 5, len(VOCAB.feature_names)


def _make_evaluator(seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.normal(size=(N, T)).astype(np.float32)
    labels[rng.random((N, T)) < 0.1] = np.nan          # 模拟丢弃（停牌/窗口不足）
    mask = rng.random((N, T)) < 0.9
    return ICEvaluator(np.zeros((N, F, T), dtype=np.float32), labels, mask,
                       device=torch.device("cpu"))


def test_avg_rank_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 37))
    x[:, :5] = 7.0                                      # 强制并列
    x[2, 10:14] = -3.0
    got = avg_rank(torch.tensor(x, dtype=torch.float32)).numpy()
    for b in range(x.shape[0]):
        expect = rankdata(x[b])
        assert np.allclose(got[b], expect, atol=1e-5)


def test_ic_matches_spearmanr():
    ev = _make_evaluator(2)
    rng = np.random.default_rng(3)
    scores = rng.normal(size=(N, T)).astype(np.float32)
    ics = ev.ic_series(torch.tensor(scores)).numpy()
    valid = ev.valid.numpy()
    for k, t in enumerate(ev.usable_dates):
        v = valid[:, t]
        expect = spearmanr(scores[v, t], ev.excess[:, t][v].numpy()).statistic
        assert np.isclose(ics[k], expect, atol=1e-5), f"date {t} mismatch"


def test_planted_single_feature_formula():
    """植入公式（单特征独立成式）的 reward == 该特征逐日 Spearman 均值（试金石）。"""
    rng = np.random.default_rng(4)
    labels = rng.normal(size=(N, T)).astype(np.float32)
    mask = np.ones((N, T), dtype=bool)
    feats = rng.normal(size=(N, F, T)).astype(np.float32)
    ev = ICEvaluator(feats, labels, mask, device=torch.device("cpu"))

    vm = StackVM()
    for f_idx, name in enumerate(VOCAB.feature_names):
        scores = vm.execute([f_idx], torch.tensor(feats))
        reward = float(ev.evaluate(scores.unsqueeze(0)))
        direct = []
        valid = ev.valid.numpy()
        for t in ev.usable_dates:
            v = valid[:, t]
            direct.append(spearmanr(feats[:, f_idx, t][v],
                                    ev.excess[:, t][v].numpy()).statistic)
        assert np.isclose(reward, float(np.mean(direct)), atol=1e-5), \
            f"{name}: reward {reward} vs direct {np.mean(direct)}"


def test_random_scores_ic_near_zero():
    """随机分数的 IC 均值应接近 0（评估器无偏性粗检）。"""
    ev = _make_evaluator(5)
    rng = np.random.default_rng(6)
    bs = 64
    scores = rng.normal(size=(bs, N, T)).astype(np.float32)
    rewards = ev.evaluate_columns(torch.tensor(scores), ev.usable_dates).numpy()
    assert abs(rewards.mean()) < 0.15                   # N=60 小样本，阈值放宽
    assert rewards.shape == (bs,)
