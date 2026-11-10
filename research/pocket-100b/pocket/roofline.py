"""The cost model: what it takes to emit one token, and where it comes from.

One equation
------------
For single-stream (batch 1) decoding, arithmetic is not the constraint; movement
is.  Every plan in this file reduces to:

    seconds_per_token  =  sum over tiers( bytes_touched[tier] / bandwidth[tier] )
    tokens_per_second  =  1 / max(seconds_per_token, flops_per_token / device_flops)

and the only interesting question is **which bytes are touched**.  That gives the
three levers this study is about, in increasing order of how much they are worth:

    bits per weight     -> scale every byte term
    sparsity / caching  -> divide the byte term (MoE routes, hot experts stay put)
    speculation         -> divide again, by accepting several tokens per weight pass

The model deliberately keeps the numbers separable, so an experiment can replace
any assumption with a measurement.  ``hit_rate`` in particular should come from a
real router trace (``moe_probe.py``) rather than the shape assumption used when no
trace is supplied.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .catalog import Device, ModelSpec


# --------------------------------------------------------------------------- #
# Cache-hit curve
# --------------------------------------------------------------------------- #


def default_hit_rate(cache_fraction: float, skew: float = 1.55) -> float:
    """Fraction of expert requests served by holding ``cache_fraction`` of them.

    Router load is heavy-tailed: a small hot set absorbs most tokens (PowerInfer
    reports >80% of activations on the hot set; production MoE offloaders report
    ~77% hit rate with ~10% of the experts cached).  ``1-(1-f)^skew`` is a
    one-parameter stand-in for that shape -- concave, saturating, exactly 0 at
    f=0 and 1 at f=1.  It is an assumption, and RESULTS.md prints the measured
    curve from this repo's own MoE next to it so the difference is visible.
    """

    f = min(max(cache_fraction, 0.0), 1.0)
    return 1.0 - (1.0 - f) ** skew


def speculation_tokens(alpha: float, k: int) -> float:
    """Expected tokens accepted per target-model weight pass (incl. the bonus).

    Geometric acceptance: a draft of ``k`` tokens is verified in one pass; each
    token survives with probability ``alpha``.  E[tokens] = sum_{i<=k} alpha^i.
    """

    if alpha <= 0 or k <= 0:
        return 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha) if alpha < 1 else float(k + 1)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def plan(
    model: ModelSpec,
    device: Device,
    bits: float = 4.0,
    ctx: int = 4096,
    kv_bits: float = 8.0,
    strategy: str = "auto",
    alpha: float = 0.0,
    draft_tokens: int = 0,
    cache_bytes: Optional[float] = None,
    skew: float = 1.55,
    batch: int = 1,
) -> Dict[str, object]:
    """Cost of generating one token under an explicit placement strategy.

    Strategies, in the order real engines try them:

    ``vram``       weights + KV entirely in VRAM (the good case).
    ``ram``        entirely in system RAM, computed on the CPU (unified memory
                   devices like Apple silicon or a plain CPU box).
    ``moecache``   hot experts cached in VRAM, cold experts *computed on the CPU*
                   from system RAM, overflow read from storage.  This is what
                   llama.cpp ``--n-cpu-moe`` / PowerInfer / ktransformers do, and
                   for MoE it is usually the fastest placement that exists on a
                   consumer box.
    ``pcie``       hot experts in VRAM, cold experts *copied to the GPU* over PCIe.
                   Costs bandwidth instead of CPU time; wins only when the CPU is
                   very weak.
    ``stream``     every weight read from storage once per token (worst case, and
                   the only option for a dense model bigger than RAM).

    ``auto`` evaluates every feasible strategy and returns the fastest.
    """

    if strategy == "auto":
        feasible = []
        for cand in ("vram", "ram", "moecache", "pcie", "stream"):
            try:
                feasible.append(
                    plan(model, device, bits, ctx, kv_bits, cand, alpha, draft_tokens,
                         cache_bytes, skew, batch)
                )
            except ValueError:
                continue
        if not feasible:
            raise ValueError("no feasible strategy")
        best = dict(max(feasible, key=lambda c: c["tok_s"]))
        best["alternatives"] = sorted(
            [
                {"strategy": c["strategy"], "tok_s": c["tok_s"], "bytes_per_token": c["bytes_per_token"]}
                for c in feasible
            ],
            key=lambda c: -c["tok_s"],
        )
        return best

    W = model.weight_bytes(bits)
    kv = model.kv_bytes(ctx, batch=batch, kv_bits=kv_bits)
    vram_cap = device.vram_gb * 1e9
    ram_cap = device.ram_gb * 1e9
    per_token_active = model.active_weight_bytes(bits)
    has_vram = device.vram_gb > 0
    fast_bw = device.vram_bw_gbs if has_vram else device.ram_bw_gbs
    fast_tier = "vram" if has_vram else "ram"
    notes: List[str] = []

    # flows: (bytes moved per token, bandwidth GB/s, tier name)
    flows: List[Tuple[float, float, str]] = []
    gpu_params = 0.0
    cpu_params = 0.0
    resident = kv
    resident_tier = fast_tier

    if strategy == "vram":
        if not has_vram or vram_cap <= W + kv:
            raise ValueError("model does not fit in VRAM")
        # Capacity needs the whole model; bandwidth only needs what a token touches.
        # For a resident MoE that is the active experts, not the file.
        flows.append((per_token_active, device.vram_bw_gbs, "vram"))
        resident = W + kv
        gpu_params = model.active_params_b
        notes.append(
            "weights and KV cache resident in VRAM: bandwidth-bound, the good case"
            + (" (a token touches only its routed experts)" if model.is_moe else "")
        )

    elif strategy == "ram":
        if ram_cap <= W + kv:
            raise ValueError("model does not fit in RAM")
        flows.append((per_token_active, device.ram_bw_gbs, "ram"))
        resident = W + kv
        resident_tier = "ram"
        cpu_params = model.active_params_b
        notes.append(
            "weights and KV cache in system memory, arithmetic on the CPU"
            + (" (a token touches only its routed experts)" if model.is_moe else "")
        )

    elif strategy in ("moecache", "pcie"):
        if not has_vram:
            raise ValueError("no VRAM to cache the hot set")
        room = max(0.0, vram_cap - kv)
        if room >= W:
            raise ValueError("everything fits in VRAM; use strategy=vram")
        hot_bytes = min(room, W)
        if model.is_moe:
            expert_bytes = model.expert_params_b * 1e9 * bits / 8.0
            non_expert = W - expert_bytes
            # VRAM holds the dense part (attention, norms, shared expert) first,
            # then as much of the expert pool as fits.
            room_for_experts = max(0.0, hot_bytes - min(non_expert, hot_bytes))
            cached_frac = min(1.0, room_for_experts / expert_bytes) if expert_bytes else 0.0
            if cache_bytes is not None:
                cached_frac = min(1.0, cache_bytes / expert_bytes) if expert_bytes else 0.0
                hot_bytes = min(W, non_expert + cached_frac * expert_bytes)
            hit = default_hit_rate(cached_frac, skew)
            cold_expert_bytes = per_token_active * model.expert_share * (1.0 - hit)
            hot_expert_bytes = per_token_active * model.expert_share * hit
            dense_bytes = per_token_active * (1.0 - model.expert_share)
            flows.append((hot_expert_bytes + dense_bytes, device.vram_bw_gbs, "vram"))
            resident = hot_bytes + kv
            if strategy == "moecache":
                # cold experts are read (from RAM) and computed by the CPU
                in_ram = min(cold_expert_bytes, max(0.0, ram_cap - resident))
                overflow = max(0.0, cold_expert_bytes - in_ram)
                flows.append((in_ram, device.ram_bw_gbs, "ram"))
                if overflow:
                    flows.append((overflow, device.storage_bw_gbs, "storage"))
                cpu_params = model.active_params_b * model.expert_share * (1.0 - hit)
                gpu_params = model.active_params_b - cpu_params
                notes.append(
                    f"hot experts in VRAM ({cached_frac*100:.0f}% of the expert pool, "
                    f"{hit*100:.0f}% of requests), cold experts computed on the CPU"
                )
                if overflow:
                    notes.append(
                        f"{overflow/1e9:.2f} GB/token of cold experts does not fit in RAM and "
                        f"comes from storage at {device.storage_bw_gbs:.1f} GB/s"
                    )
            else:
                flows.append((cold_expert_bytes, device.pcie_gbs, "pcie"))
                gpu_params = model.active_params_b
                notes.append(
                    f"hot experts in VRAM ({cached_frac*100:.0f}% of the pool), cold experts "
                    f"copied over PCIe at {device.pcie_gbs:.0f} GB/s"
                )
        else:
            resident_bytes = hot_bytes
            moved = W - resident_bytes
            flows.append((resident_bytes, device.vram_bw_gbs, "vram"))
            if strategy == "moecache":
                flows.append((moved, device.ram_bw_gbs, "ram"))
                cpu_params = model.active_params_b * (moved / W)
                gpu_params = model.active_params_b - cpu_params
            else:
                flows.append((moved, device.pcie_gbs, "pcie"))
                gpu_params = model.active_params_b
            resident = resident_bytes + kv
            notes.append(
                f"{resident_bytes/1e9:.1f} GB resident in VRAM, {moved/1e9:.1f} GB/token streamed"
            )

    elif strategy == "stream":
        if model.is_moe:
            # If the fast pool can hold (almost) the whole expert pool, this is not
            # streaming any more -- say so instead of reporting a 99%-hit "stream".
            expert_pool = model.expert_params_b * 1e9 * bits / 8.0
            room = max(0.0, (vram_cap if has_vram else ram_cap) - kv)
            if expert_pool and room >= 0.95 * expert_pool:
                try:
                    return plan(
                        model, device, bits=bits, ctx=ctx, kv_bits=kv_bits,
                        strategy=("moecache" if has_vram else "ram"),
                        alpha=alpha, draft_tokens=draft_tokens, cache_bytes=cache_bytes,
                        skew=skew, batch=batch,
                    )
                except ValueError:
                    # the experts fit but the dense part does not: fall through and
                    # stream the remainder, which is the honest answer
                    pass
            expert_bytes = model.expert_params_b * 1e9 * bits / 8.0
            cached = 0.0
            if cache_bytes is not None:
                cached = min(cache_bytes, expert_bytes)
            else:
                cached = min(max(0.0, (vram_cap if has_vram else ram_cap) - kv), expert_bytes)
            hit = default_hit_rate(cached / expert_bytes, skew) if expert_bytes else 1.0
            cold = per_token_active * model.expert_share * (1.0 - hit)
            hot = per_token_active * model.expert_share * hit
            dense = per_token_active * (1.0 - model.expert_share)
            flows.append((hot + dense, fast_bw, fast_tier))
            flows.append((cold, device.storage_bw_gbs, "storage"))
            resident = cached + kv
            cpu_params = model.active_params_b * model.expert_share * (1.0 - hit)
            gpu_params = model.active_params_b - cpu_params
            notes.append(
                f"{cached/1e9:.1f} GB of hot experts cached in {fast_tier} "
                f"({hit*100:.0f}% hit rate); {cold/1e9:.2f} GB/token read from storage"
            )
        else:
            flows.append((W, device.storage_bw_gbs, "storage"))
            resident = kv
            cpu_params = model.active_params_b
            notes.append(
                "every weight crosses storage once per token: throughput is "
                "storage bandwidth divided by model size, and nothing else"
            )
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    return _finish(
        model, device, bits, ctx, kv_bits, batch, alpha, draft_tokens, strategy,
        flows, gpu_params, cpu_params, resident, resident_tier, notes,
    )


def _finish(model, device, bits, ctx, kv_bits, batch, alpha, draft_tokens, strategy,
            flows, gpu_params, cpu_params, resident, resident_tier, notes):
    bytes_per_pass = sum(nb for nb, _, _ in flows)
    io_s = sum(nb / (bw * 1e9) for nb, bw, _ in flows if bw > 0)
    compute_s = max(
        (2.0 * gpu_params * 1e9) / max(device.effective_flops_tflops * 1e12, 1e-9),
        (2.0 * cpu_params * 1e9) / max(device.cpu_flops_tflops * 1e12, 1e-9),
    )
    tokens_per_pass = speculation_tokens(alpha, draft_tokens)
    io_per_token = io_s / tokens_per_pass
    tok_s = 1.0 / max(io_per_token, compute_s, 1e-12)

    dominant = max(flows, key=lambda f: f[0] / (f[1] * 1e9) if f[1] else 0.0) if flows else (0, 0, "none")
    if compute_s > io_per_token:
        bottleneck = "compute"
    elif dominant[0] < 1e-9:
        bottleneck = "none"
    else:
        bottleneck = dominant[2]

    weight_bytes = model.weight_bytes(bits)
    kv = model.kv_bytes(ctx, batch=batch, kv_bits=kv_bits)
    return {
        "model": model.name,
        "device": device.name,
        "strategy": strategy,
        "bits": bits,
        "context": ctx,
        "bytes_per_token": bytes_per_pass,
        "io_s_per_weight_pass": io_s,
        "tokens_per_weight_pass": tokens_per_pass,
        "tok_s": tok_s,
        "latency_ms": 1000.0 / tok_s if tok_s else float("inf"),
        "bottleneck": bottleneck,
        "compute_s_per_token": compute_s,
        "io_s_per_token": io_per_token,
        "flows": [{"gb_per_token": nb / 1e9, "from": tier, "bandwidth_gbs": bw} for nb, bw, tier in flows],
        "resident_gb": resident / 1e9,
        "resident_tier": resident_tier,
        "weight_gb": weight_bytes / 1e9,
        "kv_gb_at_ctx": kv / 1e9,
        "active_weight_gb": model.active_weight_bytes(bits) / 1e9,
        "fits_in_ram": weight_bytes + kv <= device.ram_gb * 1e9,
        "fits_in_vram": device.vram_gb > 0 and weight_bytes + kv <= device.vram_gb * 1e9,
        "needs_streaming": weight_bytes + kv > device.ram_gb * 1e9,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Lever stack and tables
# --------------------------------------------------------------------------- #


def lever_stack(model: ModelSpec, device: Device, ctx: int = 4096, alpha: float = 0.7,
                draft_tokens: int = 4) -> List[Dict[str, object]]:
    """The same model on the same hardware, one technique at a time.

    Each row is a plan a real runtime could execute today; the last row is the
    combination this study argues for.
    """

    rows: List[Dict[str, object]] = []

    def add(label: str, **kw) -> None:
        try:
            p = plan(model, device, ctx=ctx, **kw)
        except ValueError as exc:
            rows.append({"technique": label, "tok_s": 0.0, "bytes_per_token": 0.0, "error": str(exc)})
            return
        rows.append(
            {
                "technique": label,
                "tok_s": p["tok_s"],
                "bytes_per_token": p["bytes_per_token"],
                "bottleneck": p["bottleneck"],
                "resident_gb": p["resident_gb"],
                "strategy": p["strategy"],
            }
        )

    add("fp16 resident", bits=16.0, strategy="ram")
    add("8-bit resident", bits=8.0, strategy="ram")
    add("4-bit resident", bits=4.0, strategy="ram")
    add("+ expert cache (VRAM)", bits=4.0, strategy="moecache")
    add("+ stream cold from SSD", bits=4.0, strategy="stream")
    add("+ speculation", bits=4.0, strategy="stream", alpha=alpha, draft_tokens=draft_tokens)
    base = rows[0]["tok_s"] or 1e-9
    for row in rows:
        row["speedup_vs_fp16_resident"] = (row["tok_s"] or 0.0) / base
    return rows


def bandwidth_wall(devices: Sequence[Device], model: ModelSpec, bits: float = 4.0) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """tok/s against bytes-per-token, one line per device, log-log.

    The lines are the straight line ``tok/s = bandwidth / bytes``: this is what
    "the model does not fit" actually costs, and it is why no amount of scheduling
    cleverness rescues a dense model on a slow bus.
    """

    out = []
    for dev in devices:
        bw = dev.storage_bw_gbs if model.weight_bytes(bits) > dev.ram_gb * 1e9 else dev.ram_bw_gbs
        pts = []
        bpt = 1e7
        while bpt <= 1e12:
            pts.append((bpt, bw * 1e9 / bpt))
            bpt *= 2
        out.append((f"{dev.name} ({bw:.0f} GB/s)", pts))
    return out


def summarise(rows: Sequence[Dict[str, object]]) -> str:
    head = f"{'model':26s} {'device':26s} {'strategy':8s} {'GB/token':>9s} {'tok/s':>8s}  bottleneck"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{str(r['model']):26s} {str(r['device']):26s} {str(r['strategy']):8s} "
            f"{float(r['bytes_per_token'])/1e9:9.2f} {float(r['tok_s']):8.2f}  {r['bottleneck']}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from . import catalog

    ap = argparse.ArgumentParser(description="Plan streaming inference for a model/device pair")
    ap.add_argument("models", nargs="*", default=["dense-100b"])
    ap.add_argument("--device", action="append", default=None, help="device key (repeatable)")
    ap.add_argument("--bits", type=float, default=4.0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    device_keys = args.device or ["phone-8gb", "laptop-16gb", "desktop-32gb-4090"]
    rows = []
    for mk in args.models:
        m = catalog.model(mk)
        for dk in device_keys:
            d = catalog.device(dk)
            p = plan(m, d, bits=args.bits, ctx=args.ctx)
            rows.append(p)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(summarise(rows))
        print()
        for r in rows:
            for note in r["notes"]:
                print(f"  {r['model']} / {r['device']}: {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
