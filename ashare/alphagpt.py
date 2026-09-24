# -*- coding: utf-8 -*-
"""AlphaGPT 生成器（自 model_core/alphagpt.py 移植，结构不变）：

小型自回归 Transformer（d_model=64, 4 头, 2 层 LoopedTransformer ×3 循环），
RMSNorm / QK-Norm / SwiGLU / MTPHead（多任务预留）/ critic 头（预留）。
词表换为 A 股 6 原子 + 12 算子。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import D_MODEL, MAX_FORMULA_LEN
from .vocab import FORMULA_VOCAB


class NewtonSchulzLowRankDecay:
    """LoRD 正则：Newton-Schulz 迭代近似正交极分解，对注意力权重做低秩衰减。"""

    def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5,
                 target_keywords=None):
        self.decay_rate = decay_rate
        self.num_iterations = num_iterations
        self.target_keywords = target_keywords or ["qk_norm", "attention"]
        self.params_to_decay = []
        for name, param in named_parameters:
            if not param.requires_grad or param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            self.params_to_decay.append((name, param))

    @torch.no_grad()
    def step(self):
        for _, W in self.params_to_decay:
            orig_dtype = W.dtype
            X = W.float()
            r, c = X.shape
            transposed = False
            if r > c:
                X = X.T
                transposed = True
            norm = X.norm() + 1e-8
            X = X / norm
            Y = X
            I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)
            for _ in range(self.num_iterations):
                A = Y.T @ Y
                Y = 0.5 * Y @ (3.0 * I - A)
            if transposed:
                Y = Y.T
            W.sub_(self.decay_rate * Y.to(orig_dtype))


class StableRankMonitor:
    """监控目标参数的有效秩（stable rank）。"""

    def __init__(self, model, target_keywords=None):
        self.model = model
        self.target_keywords = target_keywords or ["q_proj", "k_proj", "attention"]
        self.history = []

    @torch.no_grad()
    def compute(self):
        ranks = []
        for name, param in self.model.named_parameters():
            if param.ndim != 2:
                continue
            if not any(k in name for k in self.target_keywords):
                continue
            S = torch.linalg.svdvals(param.detach().float())
            ranks.append(((S.norm() ** 2) / (S[0] ** 2 + 1e-9)).item())
        avg = sum(ranks) / len(ranks) if ranks else 0.0
        self.history.append(avg)
        return avg


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model ** -0.5))

    def forward(self, q, k):
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_ff):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)

    def forward(self, x):
        x_glu = self.w(x)
        x, gate = x_glu.chunk(2, dim=-1)
        return self.fc(x * F.silu(gate))


class MTPHead(nn.Module):
    """多任务池化头（当前训练未启用，actor-critic / 多目标预留）。"""

    def __init__(self, d_model, vocab_size, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size) for _ in range(num_tasks)])
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        self.task_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, num_tasks))

    def forward(self, x):
        task_logits = self.task_router(x)
        task_probs = F.softmax(task_logits, dim=-1)
        outputs = torch.stack([head(x) for head in self.task_heads], dim=1)
        weighted = (task_probs.unsqueeze(-1) * outputs).sum(dim=1)
        return weighted, task_probs


class LoopedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_loops=3, dropout=0.1):
        super().__init__()
        self.num_loops = num_loops
        self.qk_norm = QKNorm(d_model // nhead)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True,
                                               dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, is_causal=False):
        for _ in range(self.num_loops):
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(x_norm, x_norm, x_norm,
                                         attn_mask=mask, is_causal=is_causal)
            x = x + self.dropout(attn_out)
            x_norm = self.norm2(x)
            x = x + self.dropout(self.ffn(x_norm))
        return x


class LoopedTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward,
                 num_loops=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
            for _ in range(num_layers)])

    def forward(self, x, mask=None, is_causal=False):
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        return x


class AlphaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = D_MODEL
        self.vocab = list(FORMULA_VOCAB.token_names)
        self.vocab_size = FORMULA_VOCAB.size

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, MAX_FORMULA_LEN + 1, self.d_model))
        self.blocks = LoopedTransformer(
            d_model=self.d_model, nhead=4, num_layers=2,
            dim_feedforward=128, num_loops=3, dropout=0.1)
        self.ln_f = RMSNorm(self.d_model)
        self.mtp_head = MTPHead(self.d_model, self.vocab_size, num_tasks=3)
        self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last_emb = x[:, -1, :]
        logits, task_probs = self.mtp_head(last_emb)
        value = self.head_critic(last_emb)
        return logits, value, task_probs
