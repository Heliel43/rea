"""Probes: read the model's internals, then decide what to move.

The planner in ``roofline.py`` is only as good as two numbers it cannot measure by
itself -- the *hit rate* of an expert cache and the *channel concentration* of the
FFN -- so this module extracts both from a real router trace, plus the quality cost
of quantization measured end-to-end instead of in weight space.

What is measured, exactly:

``router_stats``          how uneven expert selection is (per layer, and overall)
``cache_curve``           LRU hit rate over the real request stream vs cache size
``channel_concentration`` how much of an FFN block's output a fraction of its
                          channels carries -- the relevance-driven sparsity that
                          PowerInfer-style engines exploit
``quant_quality``         held-out loss / top-1 agreement / logit cosine for each
                          scheme, with the same weights the trainer produced
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple

from . import nn, quant


def collect_traces(
    cfg: nn.ModelConfig,
    p: nn.Params,
    docs: Sequence[Tuple[List[int], List[int]]],
    limit: int = 8,
) -> nn.ForwardTrace:
    """Run the model over documents and keep per-token, per-layer routing decisions.

    The merge is *per layer*: ``router_choices[i]`` is the concatenation of layer
    ``i``'s decisions across documents.  (Getting this wrong silently turns the
    layer axis into a document axis, and every cache number downstream becomes
    meaningless -- which is exactly what happened the first time.)
    """

    merged = nn.ForwardTrace()
    for x, _ in docs[:limit]:
        trace = nn.ForwardTrace()
        nn.forward(cfg, p, x, trace=trace)
        if not merged.router_choices:
            merged.router_choices = [list(layer) for layer in trace.router_choices]
            merged.router_probs = [list(layer) for layer in trace.router_probs]
            merged.ffn_activations = [list(a) for a in trace.ffn_activations]
            merged.attention_entropy = [list(a) for a in trace.attention_entropy]
        else:
            for i, layer in enumerate(trace.router_choices):
                merged.router_choices[i].extend(layer)
            for i, layer in enumerate(trace.router_probs):
                merged.router_probs[i].extend(layer)
            for i, acts in enumerate(trace.ffn_activations):
                merged.ffn_activations[i].extend(acts)
            for i, ent in enumerate(trace.attention_entropy):
                merged.attention_entropy[i].extend(ent)
    return merged


def skewed_documents(corpus, n_docs: int, seed: int = 4242, topic_weights: Sequence[float] = (0.5, 0.3, 0.15, 0.05)):
    """Sample documents with a skewed topic distribution.

    Real traffic is not topic-uniform: a handful of subjects dominate.  Routing
    locality in a *trace* is therefore stronger than in a uniformly sampled
    evaluation set, and using the uniform set would understate what a cache can do.
    """

    import random as _random

    rng = _random.Random(seed)
    docs = []
    for _ in range(n_docs):
        r = rng.random()
        acc = 0.0
        topic = 0
        for i, w in enumerate(topic_weights):
            acc += w
            if r <= acc:
                topic = i
                break
        doc = corpus.document(rng, topic)
        docs.append((doc[:-1], doc[1:]))
    return docs


def router_stats(cfg: nn.ModelConfig, trace: nn.ForwardTrace) -> Dict[str, object]:
    """Per-layer load, plus a Gini coefficient over the whole expert pool."""

    if not trace.router_choices:
        return {"layers": [], "gini": 0.0, "experts": cfg.n_experts}
    per_layer = []
    counts_all: Dict[int, int] = {}
    for layer_idx, layer in enumerate(trace.router_choices):
        counts = [0] * cfg.n_experts
        for chosen in layer:
            for e in chosen:
                counts[e] += 1
        total = sum(counts) or 1
        shares = [c / total for c in counts]
        per_layer.append(
            {
                "layer": layer_idx,
                "counts": counts,
                "shares": shares,
                "max_share": max(shares),
                "entropy_bits": -sum(s * math.log2(s) for s in shares if s > 0),
                "max_entropy_bits": math.log2(cfg.n_experts) if cfg.n_experts > 1 else 0.0,
                "used": sum(1 for c in counts if c),
            }
        )
        for e, c in enumerate(counts):
            counts_all[e] = counts_all.get(e, 0) + c
    vals = sorted(counts_all.values())
    n = len(vals)
    gini = 0.0
    if n and sum(vals):
        cum = 0.0
        for i, v in enumerate(vals, 1):
            cum += i * v
        gini = (2 * cum) / (n * sum(vals)) - (n + 1) / n
    return {"layers": per_layer, "gini": gini, "experts": cfg.n_experts}


def request_stream(cfg: nn.ModelConfig, trace: nn.ForwardTrace) -> List[Tuple[int, int]]:
    "Flatten the trace into the (layer, expert) request order a cache would see."

    stream: List[Tuple[int, int]] = []
    for token_idx in range(len(trace.router_choices[0])):
        for layer_idx, layer in enumerate(trace.router_choices):
            for e in layer[token_idx]:
                stream.append((layer_idx, e))
    return stream


def lru_hits(stream: Sequence[Tuple[int, int]], capacity: int) -> Dict[str, float]:
    cache: "OrderedDict[Tuple[int, int], None]" = OrderedDict()
    hits = 0
    for item in stream:
        if item in cache:
            hits += 1
            cache.move_to_end(item)
        else:
            cache[item] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    total = len(stream) or 1
    return {"capacity": capacity, "hits": hits, "requests": len(stream), "hit_rate": hits / total}


def cache_curve(cfg: nn.ModelConfig, trace: nn.ForwardTrace, points: int = 12) -> List[Dict[str, float]]:
    """Measured hit rate vs cache capacity, expressed as a fraction of the pool."""

    stream = request_stream(cfg, trace)
    pool = cfg.n_layers * cfg.n_experts
    if pool == 0 or not stream:
        return []
    caps = sorted({max(1, round(pool * f)) for f in [i / (points - 1) for i in range(points)]})
    out = []
    for cap in caps:
        r = lru_hits(stream, cap)
        r["pool_fraction"] = cap / pool
        out.append(r)
    return out


def channel_concentration(cfg: nn.ModelConfig, trace: nn.ForwardTrace) -> Dict[str, object]:
    """How much of each FFN block's output a fraction of its channels explains.

    ``ffn_activations`` holds ``silu(gate) * up`` per selected expert per token.
    For each token we sort channels by |h| and ask how many are needed for 50/90/99%
    of the sum of |h|.  A flat distribution means "no sparsity to exploit"; a
    skewed one is the licence for relevance-driven loading.
    """

    if not trace.ffn_activations:
        return {"tokens": 0}
    needed: Dict[int, List[float]] = {50: [], 90: [], 99: []}
    zeroish: List[float] = []
    total_channels = 0
    for layer_acts in trace.ffn_activations:
        per_token = cfg.top_k * cfg.ffn_dim
        for start in range(0, len(layer_acts), per_token):
            h = [abs(v) for v in layer_acts[start : start + per_token]]
            s = sum(h)
            if s <= 0 or not h:
                continue
            h.sort(reverse=True)
            cum = 0.0
            marks = {50: None, 90: None, 99: None}
            for i, v in enumerate(h, 1):
                cum += v
                frac = cum / s
                for key in (50, 90, 99):
                    if marks[key] is None and frac >= key / 100.0:
                        marks[key] = i / len(h)
                if marks[99] is not None:
                    break
            for key in (50, 90, 99):
                needed[key].append(marks[key] if marks[key] is not None else 1.0)
            top = h[0]
            zeroish.append(sum(1 for v in h if v < 1e-3 * top) / len(h))
            total_channels += len(h)
    out: Dict[str, object] = {"tokens": len(needed[90]), "channels": total_channels}
    for k in (50, 90, 99):
        vals = needed.get(k, [])
        out[f"fraction_for_{k}pct"] = sum(vals) / len(vals) if vals else 0.0
    out["near_zero_fraction"] = sum(zeroish) / len(zeroish) if zeroish else 0.0
    return out


# --------------------------------------------------------------------------- #
# Quantization quality, end to end
# --------------------------------------------------------------------------- #


def _quantize_params(cfg: nn.ModelConfig, p: nn.Params, scheme: str, group_size: int = 32) -> nn.Params:
    q = nn.Params()
    for name, shape in p.shape.items():
        vals = p.data[name]
        if scheme == "fp16":
            decoded = quant.decode(quant.encode(vals, "fp16", group_size), len(vals), "fp16", group_size)
        else:
            decoded = quant.decode(quant.encode(vals, scheme, group_size), len(vals), scheme, group_size)
        q.add(name, shape, decoded)
    return q


def _logits_for(cfg: nn.ModelConfig, p: nn.Params, docs: Sequence[Tuple[List[int], List[int]]]) -> List[List[List[float]]]:
    out = []
    for x, _ in docs:
        logits, _ = nn.forward(cfg, p, x)
        out.append(logits)
    return out


def quant_quality(
    cfg: nn.ModelConfig,
    p: nn.Params,
    docs: Sequence[Tuple[List[int], List[int]]],
    schemes: Iterable[str] = quant.SCHEMES,
    group_size: int = 32,
) -> List[Dict[str, object]]:
    """Per-scheme: bytes, weight error, held-out loss, top-1 agreement, logit cosine."""

    reference = _logits_for(cfg, p, docs)
    base_loss = 0.0
    base_correct = 0
    total = 0
    for (x, y), logits in zip(docs, reference):
        for t in range(len(x)):
            row = logits[t]
            m = max(row)
            ex = [math.exp(v - m) for v in row]
            s = sum(ex)
            base_loss += -math.log(max(ex[y[t]] / s, 1e-12))
            base_correct += int(max(range(len(row)), key=lambda i: row[i]) == y[t])
            total += 1
    base_loss /= total
    base_top1 = base_correct / total

    rows = []
    for scheme in schemes:
        qp = _quantize_params(cfg, p, scheme, group_size)
        # weight-space error over the whole model
        sq = 0.0
        var = 0.0
        n = 0
        for name, vals in p.data.items():
            for a, b in zip(vals, qp.data[name]):
                sq += (a - b) ** 2
                var += a * a
                n += 1
        weight_rel_rmse = math.sqrt(sq / n) / (math.sqrt(var / n) or 1.0)

        loss = 0.0
        correct = 0
        agree = 0
        cos_total = 0.0
        positions = 0
        for (x, y), ref_logits in zip(docs, reference):
            logits, _ = nn.forward(cfg, qp, x)
            for t in range(len(x)):
                row = logits[t]
                m = max(row)
                ex = [math.exp(v - m) for v in row]
                s = sum(ex)
                loss += -math.log(max(ex[y[t]] / s, 1e-12))
                qbest = max(range(len(row)), key=lambda i: row[i])
                rbest = max(range(len(ref_logits[t])), key=lambda i: ref_logits[t][i])
                correct += int(qbest == y[t])
                agree += int(qbest == rbest)
                dot = sum(a * b for a, b in zip(row, ref_logits[t]))
                na = math.sqrt(sum(a * a for a in row))
                nb = math.sqrt(sum(b * b for b in ref_logits[t]))
                cos_total += dot / (na * nb) if na and nb else 1.0
                positions += 1
        storage = sum(quant.nbytes_for(len(v), scheme, group_size) for v in p.data.values())
        raw = sum(len(v) for v in p.data.values()) * 2
        rows.append(
            {
                "scheme": scheme,
                "group_size": group_size,
                "bytes": storage,
                "bits_per_weight": 8.0 * storage / sum(len(v) for v in p.data.values()),
                "compression_vs_fp16": raw / storage if storage else 0.0,
                "weight_relative_rmse": weight_rel_rmse,
                "heldout_loss": loss / positions,
                "loss_delta": loss / positions - base_loss,
                "heldout_top1": correct / positions,
                "top1_delta": correct / positions - base_top1,
                "agreement_with_fp16": agree / positions,
                "logit_cosine": cos_total / positions,
            }
        )
    return rows
