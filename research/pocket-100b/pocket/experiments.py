"""The experiments, as importable functions.

Each one returns plain dictionaries so that ``run_experiments.py`` can dump them
to JSON, render figures and write RESULTS.md without owning any of the logic.  The
unit tests call the same functions with tiny budgets, which is the only way a
results file stays true to the code that produced it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from . import analysis, catalog, nn, quant, roofline, store, streaming
from .data import Corpus, corpus_fingerprint
from .nn import ModelConfig, Params
from .train import generate, train

ARTIFACTS = "artifacts"

SUBJECTS: Dict[str, Dict[str, object]] = {
    "dense": dict(vocab=40, d_model=32, n_layers=3, n_heads=4, ffn_dim=64, context=24),
    "moe": dict(vocab=40, d_model=32, n_layers=3, n_heads=4, ffn_dim=32, context=24, n_experts=6, top_k=2),
}


def _ensure(workdir: str) -> str:
    path = os.path.join(workdir, ARTIFACTS)
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Subjects
# --------------------------------------------------------------------------- #


def exp_subjects(workdir: str = ".", steps: int = 450, train_docs: int = 160) -> Dict[str, object]:
    """Train the two subjects (dense and MoE) once, cache them as checkpoints."""

    art = _ensure(workdir)
    corpus = Corpus(vocab=40, doc_len=24, seed=0)
    out: Dict[str, object] = {"corpus_fingerprint": corpus_fingerprint(corpus), "subjects": {}}
    for name, kwargs in SUBJECTS.items():
        cfg = ModelConfig(**kwargs)
        ckpt = os.path.join(art, f"subject-{name}.pkck")
        meta_path = os.path.join(art, f"subject-{name}.json")
        meta = None
        if os.path.exists(ckpt) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("corpus") != corpus_fingerprint(corpus) or meta.get("steps", 0) < steps:
                meta = None
        if meta is None:
            print(f"[subjects] training {name} ({cfg.n_layers} layers, {cfg.d_model} wide, "
                  f"{cfg.n_experts} experts, {steps} steps)", flush=True)
            p, meta = train(cfg, corpus, steps=steps, lr=6e-3, seed=0, log=False)
            store.save_checkpoint(ckpt, cfg, p, meta)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, sort_keys=True)
        else:
            _, p, _ = store.load_checkpoint(ckpt)
        print(f"[subjects] {name}: heldout loss {meta['heldout']['loss']:.3f} "
              f"top1 {meta['heldout']['top1']:.3f} params {p.count()}", flush=True)
        out["subjects"][name] = {
            "config": cfg.as_dict(),
            "params": p.count(),
            "heldout": meta["heldout"],
            "train_seconds": round(meta.get("seconds", 0.0), 2),
            "history": meta.get("history", []),
            "checkpoint": ckpt,
        }
    return out


def _load_subject(workdir: str, name: str) -> Tuple[ModelConfig, Params]:
    cfg, p, _ = store.load_checkpoint(os.path.join(workdir, ARTIFACTS, f"subject-{name}.pkck"))
    return cfg, p


# --------------------------------------------------------------------------- #
# Quantization: what low bits actually cost in quality
# --------------------------------------------------------------------------- #


def exp_quant(workdir: str = ".", group_sizes: Sequence[int] = (16, 32, 64, 128)) -> Dict[str, object]:
    cfg, p = _load_subject(workdir, "dense")
    corpus = Corpus(vocab=40, doc_len=24, seed=0)
    docs = corpus.batched(6, seed=9999)

    print("[quant] weight-space and end-to-end quality per scheme", flush=True)
    rows = analysis.quant_quality(cfg, p, docs, quant.SCHEMES, group_size=32)
    by_group = []
    for gs in group_sizes:
        for scheme in ("int4", "int2"):
            by_group.extend(analysis.quant_quality(cfg, p, docs, [scheme], group_size=gs))

    # decode throughput of the reference kernels on real stored bytes
    art = _ensure(workdir)
    throughput = []
    for scheme in ("int8", "int4", "int2", "ternary"):
        probe = os.path.join(art, f"decode-probe-{scheme}.pksw")
        store.write_store(probe, cfg, p, scheme=scheme, group_size=32)
        stat = streaming.benchmark_decode(probe, scheme, seconds=1.5)
        stat["bits_per_weight"] = quant.bits_per_weight(p.count(), scheme, 32)
        throughput.append(stat)
        print(f"[quant] {scheme:8s} {stat['mb_per_s']:6.2f} MB/s of stored weights decoded "
              f"({stat['values_per_s']/1e6:5.2f} M weights/s)", flush=True)
    return {"end_to_end": rows, "group_size_sweep": by_group, "decode_throughput": throughput}


# --------------------------------------------------------------------------- #
# MoE: is the routing actually local?
# --------------------------------------------------------------------------- #


def _synthetic_zipf_curve(cfg: nn.ModelConfig, requests: int, exponent: float = 1.1) -> List[Dict[str, float]]:
    """Hit-rate curve if expert demand followed a Zipf law (production-like skew)."""

    import random as _random

    rng = _random.Random(11)
    pool = cfg.n_layers * cfg.n_experts
    weights = [1.0 / (rank + 1) ** exponent for rank in range(cfg.n_experts)]
    total = sum(weights)
    probs = [w / total for w in weights]
    stream = []
    for _ in range(requests):
        layer = rng.randrange(cfg.n_layers)
        r = rng.random()
        acc = 0.0
        choice = 0
        for e, p in enumerate(probs):
            acc += p
            if r <= acc:
                choice = e
                break
        stream.append((layer, choice))
    out = []
    for frac in (i / 10 for i in range(11)):
        cap = max(1, round(pool * frac))
        r = analysis.lru_hits(stream, cap)
        r["pool_fraction"] = cap / pool
        r["synthetic"] = True
        out.append(r)
    return out


def exp_moe(workdir: str = ".") -> Dict[str, object]:
    cfg, p = _load_subject(workdir, "moe")
    corpus = Corpus(vocab=40, doc_len=24, seed=0)
    docs = analysis.skewed_documents(corpus, 24)
    trace = analysis.collect_traces(cfg, p, docs, limit=24)
    stats = analysis.router_stats(cfg, trace)
    curve = analysis.cache_curve(cfg, trace, points=11)
    conc = analysis.channel_concentration(cfg, trace)

    print("[moe] expert load per layer (share of routed tokens):", flush=True)
    for layer in stats["layers"]:
        shares = " ".join(f"{s*100:4.1f}%" for s in layer["shares"])
        print(f"  layer {layer['layer']}: {shares}  entropy {layer['entropy_bits']:.2f}/"
              f"{layer['max_entropy_bits']:.2f} bits", flush=True)
    print(f"[moe] Gini {stats['gini']:.3f}; cache hit rate: " +
          " ".join(f"{c['pool_fraction']*100:.0f}%->{c['hit_rate']*100:.0f}%" for c in curve), flush=True)
    print(f"[moe] channel concentration: 50% of |h| in {conc.get('fraction_for_50pct',0)*100:.0f}% "
          f"of channels, 90% in {conc.get('fraction_for_90pct',0)*100:.0f}%", flush=True)

    # A synthetic, maximally-skewed reference: Zipf-weighted expert choice with the
    # same request count.  This is *not* a measurement of this model; it is the
    # shape production MoEs are reported to show, and it exists so a reader can tell
    # "routing is uniform here" apart from "caching never works".
    synthetic = _synthetic_zipf_curve(cfg, len(analysis.request_stream(cfg, trace)))

    # measured curve vs the planner's assumption
    comparison = []
    for point in curve:
        f = point["pool_fraction"]
        comparison.append(
            {
                "pool_fraction": f,
                "measured_hit_rate": point["hit_rate"],
                "assumed_hit_rate": roofline.default_hit_rate(f),
            }
        )

    return {
        "router": stats,
        "cache_curve": curve,
        "synthetic_reference": synthetic,
        "cache_comparison": comparison,
        "concentration": conc,
        "requests": len(analysis.request_stream(cfg, trace)),
    }


# --------------------------------------------------------------------------- #
# Streaming: does the scheduler keep RSS flat and stay correct?
# --------------------------------------------------------------------------- #


LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stream_probe(store_path: str, prompt: List[int], n_tokens: int, depth: int, use_mmap: bool = False) -> Dict[str, object]:
    """Run one generation in a *fresh process* so peak RSS means something."""

    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "pocket.streamprobe",
        "--store", store_path,
        "--prompt", ",".join(str(t) for t in prompt),
        "--tokens", str(n_tokens),
        "--depth", str(depth),
    ]
    if use_mmap:
        cmd.append("--mmap")
    proc = subprocess.run(cmd, cwd=LAB_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"streamprobe failed: {proc.stderr[-2000:]}")
    line = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
    if not line:
        raise RuntimeError(f"streamprobe produced no JSON: {proc.stdout[-500:]}")
    return json.loads(line[-1])


def exp_stream(workdir: str = ".", n_tokens: int = 6) -> Dict[str, object]:
    cfg, p = _load_subject(workdir, "dense")
    art = _ensure(workdir)
    corpus = Corpus(vocab=40, doc_len=24, seed=0)
    prompt = corpus.sequences(1, seed=7)[0][:8]

    reference = generate(cfg, p, prompt, n_tokens=n_tokens, temperature=0.0)
    t0 = time.perf_counter()
    generate(cfg, p, prompt, n_tokens=n_tokens, temperature=0.0)
    resident_ms = (time.perf_counter() - t0) / n_tokens * 1000

    schemes = []
    runs = []
    for scheme in ("int8", "int4"):
        path = os.path.join(art, f"subject-dense.{scheme}.pksw")
        records = store.write_store(path, cfg, p, scheme=scheme, group_size=32, meta={"subject": "dense"})
        size = os.path.getsize(path)
        schemes.append(
            {
                "scheme": scheme,
                "file_bytes": size,
                "bits_per_weight": 8.0 * size / p.count(),
                "tensors": len(records),
            }
        )
        for depth in (0, 1, 2):
            run = _stream_probe(path, prompt, n_tokens, depth)
            run["scheme"] = scheme
            run["agreement_with_fp32"] = sum(a == b for a, b in zip(run["tokens"], reference)) / n_tokens
            runs.append(run)
            print(f"[stream] {scheme} depth={depth}: {run['tok_s']:6.2f} tok/s  "
                  f"{run['bytes_read_per_token']/1e3:6.1f} KB/token  "
                  f"stall {run['stall_s']/run['seconds']*100:5.1f}%  peak RSS {run['peak_rss_mb']:5.1f} MB  "
                  f"agreement {run['agreement_with_fp32']*100:3.0f}%", flush=True)

    # the same file executed through mmap instead of explicit pread buffers
    mmap_run = _stream_probe(os.path.join(art, "subject-dense.int8.pksw"), prompt, n_tokens, 1, use_mmap=True)
    mmap_run["scheme"] = "int8"
    mmap_run["agreement_with_fp32"] = sum(a == b for a, b in zip(mmap_run["tokens"], reference)) / n_tokens
    print(f"[stream] mmap depth=1: {mmap_run['tok_s']:6.2f} tok/s  peak RSS {mmap_run['peak_rss_mb']:5.1f} MB",
          flush=True)

    return {
        "prompt": prompt,
        "reference_tokens": reference,
        "resident_ms_per_token": resident_ms,
        "stores": schemes,
        "runs": runs,
        "mmap_runs": [mmap_run],
        "model_params": p.count(),
        "model_fp32_mb": p.count() * 4 / 1e6,
    }


# --------------------------------------------------------------------------- #
# Transport: bytes per second, and what RSS does while they move
# --------------------------------------------------------------------------- #


def exp_transport(workdir: str = ".", size_mb: int = 1536, chunk_kb: int = 1024) -> Dict[str, object]:
    """Move a blob bigger than RAM, with and without overlap, and watch RSS.

    ``compute_ms`` is a *stand-in* for a native dequantize kernel: it sleeps, which
    releases the interpreter, exactly like a compiled kernel would.  Two settings
    are used -- 0 ms (pure transport) and 0.2 ms per MiB (a ~5 GB/s kernel, the
    regime a hand-written AVX2/NEON ``Q4_K`` kernel actually lives in).
    """

    art = _ensure(workdir)
    path = os.path.join(art, "transport-blob.bin")
    want = size_mb * 1024 * 1024
    if not os.path.exists(path) or os.path.getsize(path) != want:
        print(f"[transport] creating {size_mb} MB blob", flush=True)
        block = bytes(chunk_kb * 1024)
        with open(path, "wb") as fh:
            written = 0
            while written < want:
                fh.write(block)
                written += len(block)

    runs = []
    for compute_ms in (0.0, 0.2):
        for mode, depth in (("sync", 1), ("prefetch", 2), ("prefetch", 4)):
            # Two passes, best kept: this box has 2 vCPUs, a shared page cache and a
            # virtualised disk, so single measurements are noisy enough to invert.
            passes = []
            for _ in range(2):
                passes.append(
                    streaming.benchmark_transport(
                        path, want, chunk_bytes=chunk_kb * 1024, depth=depth, mode=mode,
                        compute_ms=compute_ms, drop_cache=True,
                    )
                )
            r = max(passes, key=lambda x: x["mb_per_s"])
            r["compute_ms"] = compute_ms
            r["mb_per_s_min"] = min(p["mb_per_s"] for p in passes)
            r["mb_per_s_max"] = max(p["mb_per_s"] for p in passes)
            runs.append(r)
            print(f"[transport] kernel {compute_ms:3.1f} ms/MiB  {mode:8s} depth={depth}: "
                  f"{r['mb_per_s']:7.0f} MB/s  peak RSS {r['peak_rss_kb']/1024:6.1f} MB", flush=True)

    # mmap: same bytes, but the pages count against RSS until they are dropped
    for madvise in (False, True):
        r = streaming.benchmark_transport(
            path, want, chunk_bytes=chunk_kb * 1024, depth=1, mode="mmap", fadvise=madvise,
            drop_cache=True,
        )
        r["label"] = "mmap+madvise" if madvise else "mmap (no madvise)"
        r["compute_ms"] = 0.0
        runs.append(r)
        print(f"[transport] {r['label']:18s}: {r['mb_per_s']:7.0f} MB/s  "
              f"peak RSS {r['peak_rss_kb']/1024:6.1f} MB", flush=True)

    # overlap accounting
    analysis_rows = []
    for compute_ms in (0.0, 0.2):
        subset = [r for r in runs if r.get("compute_ms") == compute_ms and r["mode"] != "mmap"]
        if not subset:
            continue
        base = next(r for r in subset if r["mode"] == "sync")
        for r in subset:
            if r["mode"] != "prefetch":
                continue
            busy = r.get("reader_busy_s") or 0.0
            transfer_share = busy / base["seconds"] if base["seconds"] else 0.0
            analysis_rows.append(
                {
                    "compute_ms": compute_ms,
                    "depth": r["depth"],
                    "sync_mb_per_s": base["mb_per_s"],
                    "prefetch_mb_per_s": r["mb_per_s"],
                    "speedup": r["mb_per_s"] / base["mb_per_s"] if base["mb_per_s"] else 0.0,
                    "reader_busy_s": busy,
                    "transfer_share_of_sync": transfer_share,
                    "max_possible_speedup": 1.0 / (1.0 - min(transfer_share, 0.95)),
                }
            )

    art_probe = os.path.join(art, "decode-probe-int4.pksw")
    decode_rate = None
    if os.path.exists(art_probe):
        decode_rate = streaming.benchmark_decode(art_probe, "int4", seconds=1.0)["mb_per_s"]
    return {
        "blob_mb": size_mb,
        "chunk_kb": chunk_kb,
        "runs": runs,
        "overlap": analysis_rows,
        "python_decode_mb_per_s": decode_rate,
    }


# --------------------------------------------------------------------------- #
# Roofline: the plan for models nobody can fit
# --------------------------------------------------------------------------- #


def exp_roofline(workdir: str = ".", hw: Optional[dict] = None) -> Dict[str, object]:
    devices = [
        catalog.device("phone-8gb"),
        catalog.device("laptop-16gb"),
        catalog.device("desktop-32gb-4090"),
        catalog.device("mac-studio-96gb"),
    ]
    if hw:
        devices.append(catalog.device_from_probe(hw, "this sandbox (pure-Python kernels)"))
    model_keys = [
        "llama-3.1-8b",
        "llama-3.1-70b",
        "mixtral-8x7b",
        "qwen3-235b-a22b",
        "gpt-oss-120b",
        "deepseek-v3",
        "dense-100b",
    ]
    rows = []
    for mk in model_keys:
        m = catalog.model(mk)
        for d in devices:
            for bits in (16.0, 8.0, 4.0, 2.0):
                p = roofline.plan(m, d, bits=bits, ctx=4096)
                rows.append(
                    {
                        "model": m.name,
                        "model_key": mk,
                        "device": d.name,
                        "bits": bits,
                        "strategy": p["strategy"],
                        "tok_s": p["tok_s"],
                        "bytes_per_token": p["bytes_per_token"],
                        "bottleneck": p["bottleneck"],
                        "resident_gb": p["resident_gb"],
                        "weight_gb": p["weight_gb"],
                        "kv_gb": p["kv_gb_at_ctx"],
                        "fits_in_ram": p["fits_in_ram"],
                        "alternatives": p.get("alternatives", []),
                    }
                )

    levers = {}
    for mk in ("gpt-oss-120b", "dense-100b", "qwen3-235b-a22b"):
        levers[mk] = {}
        for d in devices:
            levers[mk][d.name] = roofline.lever_stack(catalog.model(mk), d, ctx=4096)
    return {"rows": rows, "levers": levers, "device_names": [d.name for d in devices]}


def exp_speculation(sweep: bool = True) -> Dict[str, object]:
    """Verify the speculation formula against a Monte-Carlo of the process it models."""

    import random

    rng = random.Random(0)
    rows = []
    for alpha in (0.3, 0.5, 0.7, 0.85):
        for k in (1, 2, 4, 8):
            trials = 4000
            total = 0
            for _ in range(trials):
                accepted = 0
                for _ in range(k):
                    if rng.random() <= alpha:
                        accepted += 1
                    else:
                        break
                total += accepted + 1  # the bonus token
            rows.append(
                {
                    "alpha": alpha,
                    "draft_tokens": k,
                    "simulated_tokens_per_pass": total / trials,
                    "formula_tokens_per_pass": roofline.speculation_tokens(alpha, k),
                }
            )
            rows[-1]["abs_error"] = abs(
                rows[-1]["simulated_tokens_per_pass"] - rows[-1]["formula_tokens_per_pass"]
            )
    return {"rows": rows, "sweep": sweep}
