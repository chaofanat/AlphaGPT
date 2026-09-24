# -*- coding: utf-8 -*-
"""REINFORCE 训练引擎（自 model_core/engine.py 移植并适配截面评估）。

与原版的差异：
- 数据入口：Postgres → 物化 npz 面板（ashare/materialize.py）
- reward：单币 PnL 回测 → 网格日截面 RankIC 均值（池内超额，ashare/evaluator.py）
- 成本控制：截面 RankIC 含排序，每步随机抽 DATES_PER_STEP 个网格日估 reward
  （REINFORCE 本就是随机估计，日期子样可接受）；终评用全日期。

REINFORCE 主循环不变：批量采样公式 → 栈机执行 → 回测/IC 评分 →
advantage 归一化加权 log-prob → AdamW 更新 + LoRD 低秩衰减。
"""
import json
import random

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay
from .config import (BATCH_SIZE, CONSTANT_REWARD, DATES_PER_STEP, DEVICE,
                     ILLEGAL_REWARD, MAX_FORMULA_LEN, OUTPUT_DIR, SEED,
                     TRAIN_STEPS)
from .evaluator import ICEvaluator
from .materialize import load_panel
from .vocab import FORMULA_VOCAB
from .vm import StackVM

TOP_K = 16          # 终评候选数（按逐步 reward 保留的 top-K 公式再全日期评估）


class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3,
                 lord_num_iterations=5):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        panel = load_panel()
        self.features = torch.as_tensor(panel["features"], dtype=torch.float32,
                                        device=DEVICE)     # [N, F, Tg]
        self.evaluator = ICEvaluator(self.features, panel["labels"], panel["mask"])
        self.grid_dates = list(panel["grid_dates"])
        self.symbols = list(panel["symbols"])

        self.model = AlphaGPT().to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(), decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"])
        else:
            self.lord_opt = None

        self.vm = StackVM()
        self.best_score = -float("inf")
        self.best_formula = None
        self.top_formulas = []            # [(reward, formula)] 维护 top-K
        self.training_history = {"step": [], "avg_reward": [], "best_score": []}

    # ---- 采样与评估 ----

    def _sample_batch(self):
        bs = BATCH_SIZE
        inp = torch.zeros((bs, 1), dtype=torch.long, device=DEVICE)
        log_probs, tokens_list = [], []
        for _ in range(MAX_FORMULA_LEN):
            logits, _, _ = self.model(inp)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            tokens_list.append(action)
            inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
        return torch.stack(tokens_list, dim=1), log_probs

    @torch.no_grad()
    def _execute_batch(self, seqs, date_subset):
        """逐公式栈机执行，返回 [B, N, D] 分数张量与合法性/常量标记。"""
        n = self.features.shape[0]
        scores = torch.zeros((seqs.shape[0], n, len(date_subset)),
                             dtype=torch.float32, device=DEVICE)
        status = torch.zeros(seqs.shape[0], dtype=torch.int8)   # 0 合法 / 1 非法 / 2 常量
        cols = torch.as_tensor(date_subset, dtype=torch.long, device=DEVICE)
        for i in range(seqs.shape[0]):
            res = self.vm.execute(seqs[i].tolist(), self.features)
            if res is None:
                status[i] = 1
                continue
            if float(res.std()) < 1e-4:
                status[i] = 2
                continue
            scores[i] = res[:, cols]
        return scores, status

    def _step_reward(self, scores, status, date_subset):
        rewards = self.evaluator.evaluate_columns(scores, date_subset)
        rewards = torch.where(status == 1, torch.full_like(rewards, ILLEGAL_REWARD), rewards)
        rewards = torch.where(status == 2, torch.full_like(rewards, CONSTANT_REWARD), rewards)
        return rewards

    # ---- 主循环 ----

    def train(self, steps=TRAIN_STEPS, verbose=True):
        print(f"🚀 A股截面因子挖掘启动（device={DEVICE}, batch={BATCH_SIZE}, "
              f"dates/step={DATES_PER_STEP}, 词表={FORMULA_VOCAB.size} tokens）")
        usable = self.evaluator.usable_dates
        pbar = tqdm(range(steps), disable=not verbose)
        for step in pbar:
            seqs, log_probs = self._sample_batch()
            date_subset = random.sample(usable, min(DATES_PER_STEP, len(usable)))

            scores, status = self._execute_batch(seqs, date_subset)
            rewards = self._step_reward(scores, status, date_subset)

            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            loss = sum(-log_probs[t] * adv for t in range(len(log_probs))).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if self.use_lord:
                self.lord_opt.step()

            # top-K 追踪（按抽样 reward；终评用全日期纠正抽样噪声）
            order = torch.argsort(rewards, descending=True)
            for i in order[:TOP_K].tolist():
                if status[i] != 0:
                    continue
                formula = seqs[i].tolist()
                self.top_formulas.append((float(rewards[i]), formula))
            self.top_formulas = sorted(self.top_formulas, key=lambda x: -x[0])[:TOP_K]

            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(float(rewards.mean()))
            self.training_history["best_score"].append(
                self.top_formulas[0][0] if self.top_formulas else float("nan"))
            pbar.set_postfix({"AvgRew": f"{rewards.mean():.4f}",
                              "TopIC": f"{self.training_history['best_score'][-1]:.4f}"})

        self._finalize()
        return self.best_formula, self.best_score

    @torch.no_grad()
    def _finalize(self):
        """全日期终评 top-K，落盘最优公式与训练历史。"""
        if not self.top_formulas:
            print("⚠️ 未产生任何合法公式")
            return
        self.top_formulas = sorted(self.top_formulas, key=lambda x: -x[0])
        seen, candidates = set(), []
        for rew, formula in self.top_formulas:
            key = tuple(formula)
            if key not in seen:
                seen.add(key)
                candidates.append(formula)

        best = None
        for formula in candidates:
            res = self.vm.execute(formula, self.features)
            if res is None:
                continue
            ic = float(torch.nanmean(self.evaluator.ic_series(res.unsqueeze(0))))
            if best is None or ic > best[1]:
                best = (formula, ic)

        if best is None:
            print("⚠️ top-K 公式终评全部失败")
            return
        self.best_formula, self.best_score = best

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "formula_ids": self.best_formula,
            "formula": FORMULA_VOCAB.formula_to_names(self.best_formula),
            "in_sample_ic": self.best_score,
            "dates_evaluated": len(self.evaluator.usable_dates),
            "feature_names": list(FORMULA_VOCAB.feature_names),
            "operator_names": list(FORMULA_VOCAB.operator_names),
        }
        (OUTPUT_DIR / "best_ashare_formula.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "training_history.json").write_text(
            json.dumps(self.training_history, ensure_ascii=False), encoding="utf-8")
        print(f"✓ 训练完成 | 最优公式: {' '.join(payload['formula'])} "
              f"| 全日期样本内 IC: {self.best_score:.4f}")


if __name__ == "__main__":
    AlphaEngine(use_lord_regularization=True).train()
