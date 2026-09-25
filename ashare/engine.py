# -*- coding: utf-8 -*-
"""REINFORCE 训练引擎（自 model_core/engine.py 移植并适配截面评估，Phase 2）。

与原版的差异：
- 数据入口：Postgres → 物化 npz 面板（ashare/materialize.py）
- reward：单币 PnL 回测 → 网格日截面 RankIC（池内超额；Phase 2 起分数先过
  中性化链——MAD→z→行业+log市值→残差z，见 ashare/preprocess.py）
- 成本控制：截面 RankIC 含排序，每步随机抽 DATES_PER_STEP 个网格日估 reward
  （REINFORCE 本就随机，日期子样可接受）
- 时间外纪律（Phase 2 核心）：网格日 < TRAIN_END 才进入训练 reward；
  终选 top-K 按样本外段（>= TRAIN_END）的 ICIR 裁决，样本内指标仅作参考。

REINFORCE 主循环不变：批量采样公式 → 栈机执行 → IC 评分 →
advantage 归一化加权 log-prob → AdamW 更新 + LoRD 低秩衰减。
"""
import json
import random

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay
from .config import (BATCH_SIZE, CONSTANT_REWARD, COST_ROUND_TRIP_PCT,
                     DATES_PER_STEP, DEVICE, ILLEGAL_REWARD, MAX_FORMULA_LEN,
                     OUTPUT_DIR, REWARD_MODE, SEED, TRAIN_END, TRAIN_STEPS)
from .evaluator import ICEvaluator, icir
from .materialize import load_panel
from .vocab import FORMULA_VOCAB
from .vm import StackVM

TOP_K = 16          # 终评候选数（按逐步 reward 保留的 top-K 公式再分段评估）


class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3,
                 lord_num_iterations=5):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        panel = load_panel()
        self.features = torch.as_tensor(panel["features"], dtype=torch.float32,
                                        device=DEVICE)     # [N, F, Tg]
        self.evaluator = ICEvaluator(self.features, panel["labels"], panel["mask"],
                                     industry=panel["industry"],
                                     log_mktcap=panel["log_mktcap"])
        self.grid_dates = list(panel["grid_dates"])
        self.symbols = list(panel["symbols"])

        # 时间外切分：训练段供 REINFORCE 采样，样本外段只参与终选裁决
        self.train_dates = [t for t in self.evaluator.usable_dates
                            if self.grid_dates[t] < TRAIN_END]
        self.test_dates = [t for t in self.evaluator.usable_dates
                           if self.grid_dates[t] >= TRAIN_END]

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
        self._top_by_formula = {}         # {formula_tuple: 最高抽样 reward}
        self.top_formulas = []            # [(formula, reward)] 去重后 top-K
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
        status = torch.zeros(seqs.shape[0], dtype=torch.int8, device=DEVICE)   # 0 合法 / 1 非法 / 2 常量
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
        rewards = self.evaluator.reward_on_columns(scores, date_subset)
        rewards = torch.where(status == 1, torch.full_like(rewards, ILLEGAL_REWARD), rewards)
        rewards = torch.where(status == 2, torch.full_like(rewards, CONSTANT_REWARD), rewards)
        return rewards

    # ---- 主循环 ----

    def train(self, steps=TRAIN_STEPS, verbose=True):
        print(f"🚀 A股截面因子挖掘启动（device={DEVICE}, batch={BATCH_SIZE}, "
              f"dates/step={DATES_PER_STEP}, 词表={FORMULA_VOCAB.size} tokens, "
              f"reward={REWARD_MODE}, 中性化={'on' if self.evaluator.neut else 'off'}, "
              f"切分={len(self.train_dates)}train/{len(self.test_dates)}test @ {TRAIN_END}）")
        pbar = tqdm(range(steps), disable=not verbose)
        for step in pbar:
            seqs, log_probs = self._sample_batch()
            date_subset = random.sample(self.train_dates,
                                        min(DATES_PER_STEP, len(self.train_dates)))

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
            # 按公式去重保留最高分：同一公式的多次幸运抽样不得挤占多个榜位
            order = torch.argsort(rewards, descending=True)
            for i in order[:TOP_K].tolist():
                if status[i] != 0:
                    continue
                formula = tuple(seqs[i].tolist())
                rew = float(rewards[i])
                if rew > self._top_by_formula.get(formula, -float("inf")):
                    self._top_by_formula[formula] = rew
            self.top_formulas = sorted(self._top_by_formula.items(),
                                       key=lambda kv: -kv[1])[:TOP_K]

            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(float(rewards.mean()))
            self.training_history["best_score"].append(
                self.top_formulas[0][1] if self.top_formulas else float("nan"))
            pbar.set_postfix({"AvgRew": f"{rewards.mean():.4f}",
                              "TopIC": f"{self.training_history['best_score'][-1]:.4f}"})

        self._finalize()
        return self.best_formula, self.best_score

    @torch.no_grad()
    def _finalize(self):
        """top-K 分段终评：样本外（>= TRAIN_END）ICIR 裁决，落盘最优公式与指标。"""
        if not self.top_formulas:
            print("⚠️ 未产生任何合法公式")
            return
        candidates = [formula for formula, _ in self.top_formulas]  # 榜内已按公式去重

        reports = []
        for formula in candidates:
            res = self.vm.execute(formula, self.features)
            if res is None:
                continue
            rep = self._segment_report(res)
            if rep is not None:
                reports.append((formula, rep))
        if not reports:
            print("⚠️ top-K 公式终评全部失败")
            return

        # 裁决序：样本外 ICIR > 样本外 IC > 样本内 ICIR（无样本外段时的降级链）
        def verdict(rep):
            te, tr = rep["test"], rep["train"]
            if te["n"] > 0:
                return (te["icir"], te["ic"])
            return (tr["icir"], tr["ic"])

        reports.sort(key=lambda fr: verdict(fr[1]), reverse=True)
        self.best_formula, best_rep = reports[0]
        self.best_score = best_rep["test"]["ic"] if best_rep["test"]["n"] > 0 \
            else best_rep["train"]["ic"]

        print("\n==== 终评分段报告（按样本外 ICIR 排序）====")
        print(f"{'公式':<58}{'train IC':>9}{'ICIR':>7}{'test IC':>9}{'ICIR':>7}"
              f"{'净分层%':>8}")
        for formula, rep in reports:
            names = " ".join(FORMULA_VOCAB.formula_to_names(formula))
            te, tr = rep["test"], rep["train"]
            net = (te["layer_spread"] - COST_ROUND_TRIP_PCT) if te["n"] > 0 else float("nan")
            print(f"{names[:57]:<58}{tr['ic']:>9.4f}{tr['icir']:>7.2f}"
                  f"{te['ic']:>9.4f}{te['icir']:>7.2f}{net:>8.2f}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "formula_ids": self.best_formula,
            "formula": FORMULA_VOCAB.formula_to_names(self.best_formula),
            "selection": "test_icir" if best_rep["test"]["n"] > 0 else "train_icir",
            "reward_mode": REWARD_MODE,
            "neutralized": self.evaluator.neut is not None,
            "train_end": TRAIN_END,
            "cost_round_trip_pct": COST_ROUND_TRIP_PCT,
            "metrics": best_rep,
            "feature_names": list(FORMULA_VOCAB.feature_names),
            "operator_names": list(FORMULA_VOCAB.operator_names),
        }
        (OUTPUT_DIR / "best_ashare_formula.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "training_history.json").write_text(
            json.dumps(self.training_history, ensure_ascii=False), encoding="utf-8")
        te, tr = best_rep["test"], best_rep["train"]
        print(f"✓ 训练完成 | 最优公式: {' '.join(payload['formula'])}")
        print(f"  样本内({tr['n']}日): IC {tr['ic']:.4f} / ICIR {tr['icir']:.2f} | "
              f"样本外({te['n']}日): IC {te['ic']:.4f} / ICIR {te['icir']:.2f}")

    @torch.no_grad()
    def _segment_report(self, scores_full) -> dict | None:
        """单公式分段指标：train/test 各段 IC 均值、ICIR、五分位分层。"""
        def seg(dates):
            if not dates:
                return {"n": 0, "ic": float("nan"), "icir": float("nan"),
                        "layer_spread": float("nan")}
            ics = self.evaluator.ic_series(scores_full.unsqueeze(0), dates=dates)[0]
            ok = torch.isfinite(ics)
            n = int(ok.sum())
            ic = float(ics[ok].mean()) if n else float("nan")
            ir = float(icir(ics)) if n else float("nan")
            spread = self.evaluator.layer_spread(scores_full, dates)
            return {"n": n, "ic": ic, "icir": ir, "layer_spread": spread}

        return {"train": seg(self.train_dates), "test": seg(self.test_dates)}


if __name__ == "__main__":
    AlphaEngine(use_lord_regularization=True).train()
