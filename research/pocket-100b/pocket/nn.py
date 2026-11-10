"""A small, dependency-free transformer used as an experimental subject.

Why this exists
---------------
Every claim in this lab is supposed to be measurable on hardware you already
own.  That rules out PyTorch: a 2-core / 3 GiB machine (the reference target
for this study) cannot build or run it, and we do not want the measurement
apparatus to hide the thing being measured.

So the subject is a deliberately small transformer -- dense or mixture-of-experts
-- written against the standard library only.  It is small enough to train on a
CPU in seconds and large enough to exhibit the three properties this study needs
to observe rather than assume:

  * *weight entropy*      -- what quantization error actually costs on trained
                             weights, not on Gaussian noise;
  * *activation sparsity* -- how many FFN channels a trained model really uses;
  * *router locality*     -- whether expert selection concentrates on a small,
                             stable subset of experts (the assumption under every
                             cached-expert inference engine).

Everything here is plain Python floats in flat ``list`` objects.  A matvec is
``sum(map(mul, row, x))``, which is the fastest thing the interpreter can do
without a native extension.  Do not read these kernels as a performance
proposal: they are the *reference* implementation (``OPS`` in this module counts
multiply-accumulates so experiments can report algorithmic cost independently of
how slow the interpreter is on the day).
"""

from __future__ import annotations

import math
from operator import mul
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

OPS = 0  # multiply-accumulate counter, incremented by the reference kernels


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Reference dot product.  The workhorse of every kernel below."""

    global OPS
    OPS += len(a)
    return sum(map(mul, a, b))


def _zeros(n: int) -> List[float]:
    return [0.0] * n


# --------------------------------------------------------------------------- #
# Parameter container
# --------------------------------------------------------------------------- #


class Params:
    """Named flat float lists.  Shapes are carried alongside, never inferred."""

    __slots__ = ("data", "shape")

    def __init__(self) -> None:
        self.data: Dict[str, List[float]] = {}
        self.shape: Dict[str, Tuple[int, ...]] = {}

    # -- construction ------------------------------------------------------ #
    def add(self, name: str, shape: Sequence[int], init: Optional[Iterable[float]] = None) -> None:
        n = 1
        for d in shape:
            n *= d
        self.shape[name] = tuple(int(d) for d in shape)
        if init is None:
            self.data[name] = _zeros(n)
        else:
            vals = list(init)
            if len(vals) != n:
                raise ValueError(f"{name}: expected {n} values, got {len(vals)}")
            self.data[name] = vals

    def rows(self, name: str) -> List[List[float]]:
        """View a 2-D parameter as a list of rows (no copy of the numbers)."""

        out_dim, in_dim = self.shape[name]
        flat = self.data[name]
        return [flat[i * in_dim : (i + 1) * in_dim] for i in range(out_dim)]

    # -- serialization ----------------------------------------------------- #
    def tensors(self) -> Dict[str, Tuple[Tuple[int, ...], List[float]]]:
        return {k: (self.shape[k], self.data[k]) for k in self.data}

    def copy(self) -> "Params":
        out = Params()
        out.shape = dict(self.shape)
        out.data = {k: list(v) for k, v in self.data.items()}
        return out

    def count(self) -> int:
        return sum(len(v) for v in self.data.values())


def init_params(cfg: "ModelConfig", rng) -> Params:
    """Xavier/Gaussian init.  Small and boring on purpose -- the point of the
    subject is what it learns, not how cleverly it started."""

    p = Params()
    d = cfg.d_model
    ff = cfg.ffn_dim
    e = cfg.n_experts

    def gauss(n: int, std: float) -> List[float]:
        # Box-Muller; stdlib random has no gauss without importing statistics
        out = []
        for _ in range(n // 2 + 1):
            u1 = max(rng.random(), 1e-12)
            u2 = rng.random()
            r = math.sqrt(-2.0 * math.log(u1))
            out.append(r * math.cos(2.0 * math.pi * u2) * std)
            out.append(r * math.sin(2.0 * math.pi * u2) * std)
        return out[:n]

    p.add("token_embd", (cfg.vocab, d), gauss(cfg.vocab * d, 0.02))
    p.add("pos_embd", (cfg.context, d), gauss(cfg.context * d, 0.01))
    p.add("out_norm", (d,), [1.0] * d)
    for layer in range(cfg.n_layers):
        pre = f"blk.{layer}."
        p.add(pre + "attn_norm", (d,), [1.0] * d)
        p.add(pre + "attn_q", (d, d), gauss(d * d, 1.0 / math.sqrt(d)))
        p.add(pre + "attn_k", (cfg.d_kv, d), gauss(cfg.d_kv * d, 1.0 / math.sqrt(d)))
        p.add(pre + "attn_v", (cfg.d_kv, d), gauss(cfg.d_kv * d, 1.0 / math.sqrt(d)))
        p.add(pre + "attn_out", (d, d), gauss(d * d, 1.0 / math.sqrt(d)))
        p.add(pre + "ffn_norm", (d,), [1.0] * d)
        if e > 1:
            p.add(pre + "router", (e, d), gauss(e * d, 1.0 / math.sqrt(d)))
        p.add(pre + "ffn_gate", (e * ff, d), gauss(e * ff * d, 1.0 / math.sqrt(d)))
        p.add(pre + "ffn_up", (e * ff, d), gauss(e * ff * d, 1.0 / math.sqrt(d)))
        p.add(pre + "ffn_down", (e * d, ff), gauss(e * d * ff, 1.0 / math.sqrt(ff)))
    return p


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class ModelConfig:
    def __init__(
        self,
        vocab: int = 48,
        d_model: int = 48,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_dim: int = 128,
        context: int = 32,
        n_experts: int = 1,
        top_k: int = 1,
        eps: float = 1e-5,
    ) -> None:
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.context = context
        self.n_experts = n_experts
        self.top_k = top_k if n_experts > 1 else 1
        self.eps = eps
        self.d_head = d_model // n_heads
        # Grouped-query attention: fewer KV channels than Q channels.
        self.d_kv = max(self.d_head, (d_model // n_heads) * max(1, n_heads // 2))

    @property
    def kv_heads(self) -> int:
        return max(1, self.d_kv // self.d_head)

    @property
    def kv_repeat(self) -> int:
        """Number of query heads that share one KV head."""

        return max(1, self.n_heads // self.kv_heads)

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1

    def as_dict(self) -> Dict[str, int]:
        return {
            "vocab": self.vocab,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ffn_dim": self.ffn_dim,
            "context": self.context,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "ModelConfig":
        return cls(
            vocab=int(d["vocab"]),
            d_model=int(d["d_model"]),
            n_layers=int(d["n_layers"]),
            n_heads=int(d["n_heads"]),
            ffn_dim=int(d["ffn_dim"]),
            context=int(d["context"]),
            n_experts=int(d.get("n_experts", 1)),
            top_k=int(d.get("top_k", 1)),
        )


# --------------------------------------------------------------------------- #
# Kernels (forward)
# --------------------------------------------------------------------------- #


def _rms_norm(x: Sequence[float], g: Sequence[float], eps: float) -> Tuple[List[float], float]:
    n = len(x)
    ms = dot(x, x) / n
    inv = 1.0 / math.sqrt(ms + eps)
    return [g[i] * x[i] * inv for i in range(n)], inv


def _softmax(v: Sequence[float]) -> List[float]:
    m = max(v)
    ex = [math.exp(t - m) for t in v]
    s = sum(ex)
    return [t / s for t in ex]


def _silu(a: float) -> float:
    return a / (1.0 + math.exp(-a))


def _matvec(rows: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    return [dot(r, x) for r in rows]


class ForwardTrace:
    """Optional instrumentation: what the model actually did, per token."""

    def __init__(self) -> None:
        self.router_choices: List[List[List[int]]] = []  # [layer][position] -> expert ids
        self.router_probs: List[List[List[float]]] = []
        self.ffn_activations: List[List[float]] = []  # flattened hidden activations
        self.attention_entropy: List[List[float]] = []


def _rows(flat: Sequence[float], in_dim: int) -> List[List[float]]:
    """Zero-copy-ish row view of a flat matrix (slices share no memory, but the
    reference implementation is not trying to be clever)."""

    return [flat[i * in_dim : (i + 1) * in_dim] for i in range(len(flat) // in_dim)]


def layer_forward(
    cfg: ModelConfig,
    layer: int,
    get,
    x: List[List[float]],
    trace: Optional[ForwardTrace] = None,
    keep: bool = False,
):
    """One transformer block.

    ``get(name)`` returns the flat values of a block parameter (``attn_q``,
    ``ffn_down``, ...) without the ``blk.N.`` prefix.  The callable is the only
    difference between an in-memory model and a streamed one: the whole point of
    this lab is that the arithmetic must not change when the weights arrive from
    disk instead of a list.
    """

    d = cfg.d_model
    T = len(x)
    H, dh = cfg.n_heads, cfg.d_head
    kvr, dkv = cfg.kv_repeat, cfg.d_kv
    eps = cfg.eps

    x_in = [list(v) for v in x]

    attn_norm = get("attn_norm")
    xnorms: List[List[float]] = []
    invs: List[float] = []
    for t in range(T):
        xn_t, inv = _rms_norm(x[t], attn_norm, eps)
        xnorms.append(xn_t)
        invs.append(inv)

    Wq = _rows(get("attn_q"), d)
    Wk = _rows(get("attn_k"), d)
    Wv = _rows(get("attn_v"), d)
    Wo = _rows(get("attn_out"), d)
    q = [_matvec(Wq, xnorms[t]) for t in range(T)]
    k = [_matvec(Wk, xnorms[t]) for t in range(T)]
    v = [_matvec(Wv, xnorms[t]) for t in range(T)]

    attn_out: List[List[float]] = []
    attn_probs: List[List[List[float]]] = []
    entropies: List[float] = []
    for t in range(T):
        row: List[float] = []
        probs_t: List[List[float]] = []
        for h in range(H):
            lo, hi = h * dh, (h + 1) * dh
            kvh = (h // kvr) * dh
            scores = [dot(q[t][lo:hi], k[s][kvh : kvh + dh]) / math.sqrt(dh) for s in range(t + 1)]
            pr = _softmax(scores)
            probs_t.append(pr)
            entropies.append(-sum(pi * math.log(pi + 1e-12) for pi in pr))
            acc = _zeros(dh)
            for s in range(t + 1):
                vh = v[s][kvh : kvh + dh]
                ps = pr[s]
                for i in range(dh):
                    acc[i] += ps * vh[i]
            row.extend(acc)
        attn_probs.append(probs_t)
        attn_out.append(_matvec(Wo, row))
    if trace is not None:
        trace.attention_entropy.append(entropies)

    x_mid = [[x[t][i] + attn_out[t][i] for i in range(d)] for t in range(T)]

    ffn_norm = get("ffn_norm")
    fnorms: List[List[float]] = []
    finvs: List[float] = []
    for t in range(T):
        fn_t, inv = _rms_norm(x_mid[t], ffn_norm, eps)
        fnorms.append(fn_t)
        finvs.append(inv)

    E, ff = cfg.n_experts, cfg.ffn_dim
    Wg = _rows(get("ffn_gate"), d)
    Wu = _rows(get("ffn_up"), d)
    Wd_all = get("ffn_down")
    router = _rows(get("router"), d) if E > 1 else None

    chosen: List[List[int]] = []
    weights: List[List[float]] = []
    gates_hidden: List[List[List[float]]] = []
    ups_hidden: List[List[List[float]]] = []
    hidden_states: List[List[List[float]]] = []
    ffn_out: List[List[float]] = []
    for t in range(T):
        if E > 1:
            logits_r = _matvec(router, fnorms[t])
            order = sorted(range(E), key=lambda i: (-logits_r[i], i))[: cfg.top_k]
            sel_p = _softmax([logits_r[i] for i in order])
        else:
            order = [0]
            sel_p = [1.0]
        gs, us, hs = [], [], []
        acc = _zeros(d)
        for slot, e in enumerate(order):
            g_pre = _matvec(Wg[e * ff : (e + 1) * ff], fnorms[t])
            u_val = _matvec(Wu[e * ff : (e + 1) * ff], fnorms[t])
            h = [_silu(g_pre[i]) * u_val[i] for i in range(ff)]
            Wd = [Wd_all[(e * d + i) * ff : (e * d + i + 1) * ff] for i in range(d)]
            y = _matvec(Wd, h)
            w = sel_p[slot]
            for i in range(d):
                acc[i] += w * y[i]
            gs.append(g_pre)
            us.append(u_val)
            hs.append(h)
        ffn_out.append(acc)
        chosen.append(order)
        weights.append(sel_p)
        gates_hidden.append(gs)
        ups_hidden.append(us)
        hidden_states.append(hs)

    if trace is not None:
        trace.router_choices.append([list(c) for c in chosen])
        trace.router_probs.append([list(w) for w in weights])
        flat: List[float] = []
        for t in range(T):
            for h in hidden_states[t]:
                flat.extend(h)
        trace.ffn_activations.append(flat)

    x_out = [[x_mid[t][i] + ffn_out[t][i] for i in range(d)] for t in range(T)]

    if not keep:
        return x_out, None

    lc = {
        "x_in": x_in,
        "xnorms": xnorms,
        "invs": invs,
        "q": q,
        "k": k,
        "v": v,
        "attn_probs": attn_probs,
        "x_mid": x_mid,
        "fnorms": fnorms,
        "finvs": finvs,
        "chosen": chosen,
        "weights": weights,
        "gate_hidden": gates_hidden,
        "up_hidden": ups_hidden,
        "hidden": hidden_states,
        "ffn_out": ffn_out,
        "attn_norm": attn_norm,
        "ffn_norm": ffn_norm,
        "Wq": Wq,
        "Wk": Wk,
        "Wv": Wv,
        "Wo": Wo,
        "router": router,
    }
    return x_out, lc


def embed_from(
    cfg: ModelConfig, tok_flat: Sequence[float], pos_flat: Sequence[float], tokens: Sequence[int]
) -> List[List[float]]:
    """Token + position embedding from flat tensors (resident or freshly read)."""

    d = cfg.d_model
    tok = _rows(tok_flat, d)
    pos = _rows(pos_flat, d)
    out = []
    for t, tok_id in enumerate(tokens):
        out.append([tok[tok_id][i] + pos[t][i] for i in range(d)])
    return out


def embed(cfg: ModelConfig, p: Params, tokens: Sequence[int]) -> List[List[float]]:
    return embed_from(cfg, p.data["token_embd"], p.data["pos_embd"], tokens)


def head_from(
    cfg: ModelConfig,
    x: List[List[float]],
    out_norm: Sequence[float],
    tok_flat: Sequence[float],
    keep: bool = False,
):
    """Final RMS norm + logits against tied token embeddings."""

    xnorms = []
    invs = []
    for t in range(len(x)):
        xn_t, inv = _rms_norm(x[t], out_norm, cfg.eps)
        xnorms.append(xn_t)
        invs.append(inv)
    emb = _rows(tok_flat, cfg.d_model)
    logits = [_matvec(emb, xnorms[t]) for t in range(len(x))]
    if keep:
        return logits, {"final_norms": xnorms, "final_invs": invs, "x_out": x}
    return logits, None


def head(cfg: ModelConfig, p: Params, x: List[List[float]], keep: bool = False):
    return head_from(cfg, x, p.data["out_norm"], p.data["token_embd"], keep=keep)


def forward(
    cfg: ModelConfig,
    p: Params,
    tokens: Sequence[int],
    trace: Optional[ForwardTrace] = None,
) -> Tuple[List[List[float]], dict]:
    """Causal forward pass.  Returns (logits per position, cache for backward)."""

    T = len(tokens)
    if T > cfg.context:
        raise ValueError(f"sequence of {T} exceeds context {cfg.context}")

    x = embed(cfg, p, tokens)
    cache: dict = {"tokens": list(tokens), "layers": [], "x_in": x}
    data = p.data
    for layer in range(cfg.n_layers):
        prefix = f"blk.{layer}."

        def get(name, _p=prefix):
            return data[_p + name]

        x, lc = layer_forward(cfg, layer, get, x, trace=trace, keep=True)
        cache["layers"].append(lc)

    logits, head_cache = head(cfg, p, x, keep=True)
    cache.update(head_cache)
    return logits, cache



# --------------------------------------------------------------------------- #
# Backward
# --------------------------------------------------------------------------- #


def _rms_norm_backward(
    dz: Sequence[float], x: Sequence[float], g: Sequence[float], inv: float, eps: float
) -> Tuple[List[float], List[float]]:
    n = len(x)
    # y = g * x * inv ; dx_j = dz_j g_j inv - (inv^3 / n) x_j * sum_i(dz_i g_i x_i)
    s = 0.0
    for i in range(n):
        s += dz[i] * g[i] * x[i]
    coef = (inv ** 3) / n
    dx = [dz[i] * g[i] * inv - coef * x[i] * s for i in range(n)]
    return dx, [dz[i] * x[i] * inv for i in range(n)]


def _softmax_backward(dp: Sequence[float], p: Sequence[float]) -> List[float]:
    s = 0.0
    for i in range(len(p)):
        s += dp[i] * p[i]
    return [p[i] * (dp[i] - s) for i in range(len(p))]


def _silu_grad(a: float) -> float:
    s = 1.0 / (1.0 + math.exp(-a))
    return s * (1.0 + a * (1.0 - s))


def backward(
    cfg: ModelConfig,
    p: Params,
    cache: dict,
    dlogits: Sequence[Sequence[float]],
) -> Params:
    """Gradients of sum(dlogits * logits) with respect to every parameter."""

    grads = Params()
    for name in p.data:
        grads.add(name, p.shape[name])

    d = cfg.d_model
    H, dh = cfg.n_heads, cfg.d_head
    E, ff = cfg.n_experts, cfg.ffn_dim
    T = len(dlogits)

    # ---- head ------------------------------------------------------------- #
    # logits[t] = E . norm(x_out[t]) with tied embeddings.
    emb = p.rows("token_embd")
    d_head_norm = [[0.0] * d for _ in range(T)]
    d_emb_flat = grads.data["token_embd"]
    for t in range(T):
        xn = cache["final_norms"][t]
        for v in range(cfg.vocab):
            dv = dlogits[t][v]
            if dv == 0.0:
                continue
            row = emb[v]
            for i in range(d):
                d_head_norm[t][i] += dv * row[i]
            base = v * d
            for i in range(d):
                d_emb_flat[base + i] += dv * xn[i]
    dx = [[0.0] * d for _ in range(T)]
    for t in range(T):
        dxi, dg = _rms_norm_backward(
            d_head_norm[t], cache["x_out"][t], p.data["out_norm"], cache["final_invs"][t], cfg.eps
        )
        for i in range(d):
            dx[t][i] = dxi[i]
            grads.data["out_norm"][i] += dg[i]

    # ---- walk layers backwards -------------------------------------------- #
    for layer in range(cfg.n_layers - 1, -1, -1):
        pre = f"blk.{layer}."
        lc = cache["layers"][layer]

        # residual split: x_out = x_mid + ffn_out
        d_ffn_out = dx
        d_x_mid = [[dx[t][i] for i in range(d)] for t in range(T)]

        # ---- FFN / MoE ---------------------------------------------------- #
        d_fnorms = [[0.0] * d for _ in range(T)]
        dWg = grads.data[pre + "ffn_gate"]
        dWu = grads.data[pre + "ffn_up"]
        dWd = grads.data[pre + "ffn_down"]
        d_router = grads.data[pre + "router"] if E > 1 else None

        for t in range(T):
            choose = lc["chosen"][t]
            sel_p = lc["weights"][t]
            dy = d_ffn_out[t]
            # expert outputs: y_e = Wd_e h_e
            dps = []
            for slot, e in enumerate(choose):
                h = lc["hidden"][t][slot]
                Wd = [p.data[pre + "ffn_down"][(e * d + i) * ff : (e * d + i + 1) * ff] for i in range(d)]
                dp = dot(dy, _matvec(Wd, h))
                dps.append(dp)
                w = sel_p[slot]
                # dWd[i][j] += w * dy[i] * h[j]; d_h[j] = sum_i w*dy[i]*Wd[i][j]
                d_h = [0.0] * ff
                for i in range(d):
                    g_i = w * dy[i]
                    if g_i == 0.0:
                        continue
                    base = (e * d + i) * ff
                    row = Wd[i]
                    for j in range(ff):
                        dWd[base + j] += g_i * h[j]
                        d_h[j] += g_i * row[j]
                # h = silu(g_pre) * u : dg_pre, du
                g_pre = lc["gate_hidden"][t][slot]
                u_val = lc["up_hidden"][t][slot]
                dg = [d_h[j] * u_val[j] * _silu_grad(g_pre[j]) for j in range(ff)]
                du = [d_h[j] * _silu(g_pre[j]) for j in range(ff)]
                fn = lc["fnorms"][t]
                for j in range(ff):
                    if dg[j]:
                        base = (e * ff + j) * d
                        for i in range(d):
                            dWg[base + i] += dg[j] * fn[i]
                    if du[j]:
                        base = (e * ff + j) * d
                        for i in range(d):
                            dWu[base + i] += du[j] * fn[i]
                # dx through the expert input
                g_rows = [p.data[pre + "ffn_gate"][(e * ff + j) * d : (e * ff + j + 1) * d] for j in range(ff)]
                u_rows = [p.data[pre + "ffn_up"][(e * ff + j) * d : (e * ff + j + 1) * d] for j in range(ff)]
                for j in range(ff):
                    gj, uj = dg[j], du[j]
                    if gj == 0.0 and uj == 0.0:
                        continue
                    grow, urow = g_rows[j], u_rows[j]
                    for i in range(d):
                        d_fnorms[t][i] += gj * grow[i] + uj * urow[i]

            # router: gradients only for the selected logits.  Note the second
            # term: the router reads the normed activation, so its gradient has
            # to reach d_fnorms as well -- with top_k == 1 the softmax over one
            # logit is constant, which hides this path entirely.
            if E > 1:
                dsel = _softmax_backward(dps, sel_p)
                fn = lc["fnorms"][t]
                for slot, e in enumerate(choose):
                    base = e * d
                    ds = dsel[slot]
                    if ds == 0.0:
                        continue
                    router_row = p.data[pre + "router"][base : base + d]
                    for i in range(d):
                        d_router[base + i] += ds * fn[i]
                        d_fnorms[t][i] += ds * router_row[i]

        # ffn_norm backward into x_mid
        for t in range(T):
            dxi, dg = _rms_norm_backward(
                d_fnorms[t], lc["x_mid"][t], p.data[pre + "ffn_norm"], lc["finvs"][t], cfg.eps
            )
            for i in range(d):
                d_x_mid[t][i] += dxi[i]
                grads.data[pre + "ffn_norm"][i] += dg[i]

        # ---- attention ---------------------------------------------------- #
        d_attn_out = d_x_mid  # x_mid = x_in + attn_out
        d_x_in = [[d_x_mid[t][i] for i in range(d)] for t in range(T)]
        dWq = grads.data[pre + "attn_q"]
        dWk = grads.data[pre + "attn_k"]
        dWv = grads.data[pre + "attn_v"]
        dWo = grads.data[pre + "attn_out"]
        Wq = p.rows(pre + "attn_q")
        Wk = p.rows(pre + "attn_k")
        Wv = p.rows(pre + "attn_v")
        Wo = p.rows(pre + "attn_out")
        kv_repeat = cfg.kv_repeat

        d_q = [[0.0] * d for _ in range(T)]
        d_k = [[0.0] * d for _ in range(T)]
        d_v = [[0.0] * d for _ in range(T)]

        for t in range(T):
            dao = d_attn_out[t]
            # row = concat of head outputs; attn_out = Wo . row
            row_full = []
            for h in range(H):
                lo, hi = h * dh, (h + 1) * dh
                kvh = (h // kv_repeat) * dh
                acc = [0.0] * dh
                for s in range(t + 1):
                    ps = lc["attn_probs"][t][h][s]
                    vh = lc["v"][s][kvh : kvh + dh]
                    for i in range(dh):
                        acc[i] += ps * vh[i]
                row_full.extend(acc)
            d_row = [0.0] * len(row_full)
            for i in range(len(row_full)):
                s = 0.0
                for o in range(d):
                    s += dao[o] * Wo[o][i]
                d_row[i] = s
                for o in range(d):
                    dWo[o * len(row_full) + i] += dao[o] * row_full[i]

            for h in range(H):
                lo, hi = h * dh, (h + 1) * dh
                kvh = (h // kv_repeat) * dh
                drows = d_row[lo:hi]
                pr = lc["attn_probs"][t][h]
                dp = []
                for s in range(t + 1):
                    vh = lc["v"][s][kvh : kvh + dh]
                    dp.append(dot(drows, vh))
                    ps = pr[s]
                    if ps != 0.0:
                        for i in range(dh):
                            d_v[s][kvh + i] += ps * drows[i]
                dscores = _softmax_backward(dp, pr)
                for s in range(t + 1):
                    g = dscores[s] / math.sqrt(dh)
                    if g == 0.0:
                        continue
                    qh = lc["q"][t][lo:hi]
                    kh = lc["k"][s][kvh : kvh + dh]
                    for i in range(dh):
                        d_q[t][lo + i] += g * kh[i]
                        d_k[s][kvh + i] += g * qh[i]

        # project attention grads back through the input projections
        d_attn_norm = [[0.0] * d for _ in range(T)]
        for t in range(T):
            xn = lc["xnorms"][t]
            for o in range(d):
                gq, gk, gv = d_q[t][o], None, None
                if gq:
                    base = o * d
                    for i in range(d):
                        dWq[base + i] += gq * xn[i]
                if o < cfg.d_kv:
                    gk, gv = d_k[t][o], d_v[t][o]
                    base = o * d
                    if gk:
                        for i in range(d):
                            dWk[base + i] += gk * xn[i]
                    if gv:
                        for i in range(d):
                            dWv[base + i] += gv * xn[i]
            for i in range(d):
                s = 0.0
                for o in range(d):
                    s += d_q[t][o] * Wq[o][i]
                for o in range(cfg.d_kv):
                    s += d_k[t][o] * Wk[o][i] + d_v[t][o] * Wv[o][i]
                d_attn_norm[t][i] = s
            dxi, dg = _rms_norm_backward(
                d_attn_norm[t],
                lc["x_in"][t],
                p.data[pre + "attn_norm"],
                lc["invs"][t],
                cfg.eps,
            )
            for i in range(d):
                d_x_in[t][i] += dxi[i]
                grads.data[pre + "attn_norm"][i] += dg[i]

        dx = d_x_in

    # ---- embeddings -------------------------------------------------------- #
    d_pos_flat = grads.data["pos_embd"]
    d_tok_flat = grads.data["token_embd"]
    for t in range(T):
        for i in range(d):
            d_pos_flat[t * d + i] += dx[t][i]
            d_tok_flat[cache["tokens"][t] * d + i] += dx[t][i]
    return grads



def loss_and_grads(
    cfg: ModelConfig,
    p: Params,
    tokens: Sequence[int],
    targets: Sequence[int],
    trace: Optional[ForwardTrace] = None,
) -> Tuple[float, Params]:
    """Cross-entropy loss over positions plus exact gradients."""

    logits, cache = forward(cfg, p, tokens, trace=trace)
    T = len(tokens)
    loss = 0.0
    dlogits = []
    for t in range(T):
        pr = _softmax(logits[t])
        loss += -math.log(max(pr[targets[t]], 1e-12))
        dl = list(pr)
        dl[targets[t]] -= 1.0
        # the reported loss is the mean over positions, so gradients carry 1/T
        dlogits.append([v / T for v in dl])
    grads = backward(cfg, p, cache, dlogits)
    return loss / T, grads
