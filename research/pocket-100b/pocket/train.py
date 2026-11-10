"""Trainer for the experimental subject.

Plain Adam with cosine decay, no frameworks, no batches larger than one document.
It exists so the model's *routing, sparsity and quantization* statistics are
measured on weights that learned something, instead of on noise.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Tuple

from . import nn
from .data import Corpus, corpus_fingerprint
from .nn import ModelConfig, Params, init_params, loss_and_grads


class Adam:
    def __init__(self, p: Params, lr: float = 3e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.p = p
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = {k: [0.0] * len(v) for k, v in p.data.items()}
        self.v = {k: [0.0] * len(v) for k, v in p.data.items()}
        self.t = 0

    def step(self, grads: Params, lr: float) -> None:
        self.t += 1
        b1, b2 = self.b1, self.b2
        c1 = 1.0 - b1 ** self.t
        c2 = 1.0 - b2 ** self.t
        for name, flat in self.p.data.items():
            g = grads.data[name]
            m = self.m[name]
            v = self.v[name]
            for i in range(len(flat)):
                gi = g[i]
                mi = m[i] = b1 * m[i] + (1.0 - b1) * gi
                vi = v[i] = b2 * v[i] + (1.0 - b2) * gi * gi
                flat[i] -= lr * (mi / c1) / (math.sqrt(vi / c2) + self.eps)


def clip_grads(grads: Params, max_norm: float) -> float:
    total = 0.0
    for flat in grads.data.values():
        for g in flat:
            total += g * g
    norm = math.sqrt(total)
    if norm > max_norm and norm > 0:
        scale = max_norm / norm
        for flat in grads.data.values():
            for i in range(len(flat)):
                flat[i] *= scale
    return norm


def evaluate(cfg: ModelConfig, p: Params, docs: List[Tuple[List[int], List[int]]]) -> Dict[str, float]:
    "Mean cross-entropy and top-1 accuracy over documents."

    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in docs:
        logits, _ = nn.forward(cfg, p, x)
        for t in range(len(x)):
            row = logits[t]
            m = max(row)
            ex = [math.exp(v - m) for v in row]
            s = sum(ex)
            pr = ex[y[t]] / s
            total_loss += -math.log(max(pr, 1e-12))
            best = max(range(len(row)), key=lambda i: row[i])
            total_correct += int(best == y[t])
            total += 1
    return {"loss": total_loss / total, "top1": total_correct / total, "positions": total}


def train(
    cfg: ModelConfig,
    corpus: Corpus,
    steps: int = 350,
    lr: float = 4e-3,
    seed: int = 0,
    log: bool = True,
    time_budget_s: Optional[float] = None,
) -> Tuple[Params, Dict[str, object]]:
    rng = random.Random(seed)
    p = init_params(cfg, rng)
    opt = Adam(p, lr=lr)
    train_docs = corpus.batched(48, seed=1234)
    heldout = corpus.batched(8, seed=9999)
    history: List[Dict[str, float]] = []
    started = time.time()
    step = 0
    while step < steps:
        x, y = train_docs[rng.randrange(len(train_docs))]
        loss, grads = loss_and_grads(cfg, p, x, y)
        clip_grads(grads, 1.0)
        cur_lr = lr * (0.5 * (1.0 + math.cos(math.pi * step / steps)))
        opt.step(grads, cur_lr)
        if log and (step % max(1, steps // 7) == 0 or step == steps - 1):
            ev = evaluate(cfg, p, heldout)
            history.append({"step": step, "train_loss": loss, "heldout_loss": ev["loss"], "top1": ev["top1"]})
            print(f"  step {step:4d}  loss {loss:.3f}  heldout {ev['loss']:.3f}  top1 {ev['top1']:.3f}", flush=True)
        step += 1
        if time_budget_s and time.time() - started > time_budget_s:
            if log:
                print(f"  stopping at step {step}: time budget reached", flush=True)
            break
    final = evaluate(cfg, p, heldout)
    meta = {
        "corpus": corpus_fingerprint(corpus),
        "steps": step,
        "heldout": final,
        "history": history,
        "seconds": time.time() - started,
        "seed": seed,
    }
    return p, meta


def generate(
    cfg: ModelConfig,
    p: Params,
    prompt: List[int],
    n_tokens: int = 8,
    temperature: float = 0.0,
    seed: int = 0,
) -> List[int]:
    "Greedy or sampled decoding.  No KV cache: every token re-runs the context."

    rng = random.Random(seed)
    seq = list(prompt)
    out = []
    for _ in range(n_tokens):
        ctx = seq[-cfg.context :]
        logits, _ = nn.forward(cfg, p, ctx)
        row = logits[-1]
        if temperature <= 0:
            tok = max(range(len(row)), key=lambda i: row[i])
        else:
            m = max(row)
            ex = [math.exp((v - m) / temperature) for v in row]
            s = sum(ex)
            r = rng.random()
            acc = 0.0
            tok = len(ex) - 1
            for i, e in enumerate(ex):
                acc += e / s
                if r <= acc:
                    tok = i
                    break
        out.append(tok)
        seq.append(tok)
    return out
