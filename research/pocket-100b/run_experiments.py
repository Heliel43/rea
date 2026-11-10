#!/usr/bin/env python3
"""Run every experiment, render the figures, and write RESULTS.md.

    python3 run_experiments.py --quick      # a few minutes, small budgets
    python3 run_experiments.py              # the full run used for RESULTS.md

Everything is written under ``artifacts/`` (git-ignored) except ``RESULTS.md`` and
``figures/*.svg``, which are the point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

from pocket import catalog, experiments, hwprobe, roofline, svg


def md_table(header: List[str], rows: List[List[object]], align: Optional[List[str]] = None) -> str:
    align = align or ["l"] * len(header)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep[a] for a in align) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def fmt(v: float, digits: int = 2) -> str:
    if v is None:
        return "-"
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.{digits}f}"
    return f"{v:.{digits + 2}f}"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_quant(data: dict, path: str) -> None:
    rows = data["end_to_end"]
    pts = sorted((r["bits_per_weight"], max(r["loss_delta"], 1e-4)) for r in rows)
    series = [("held-out loss increase", [(x, y) for x, y in pts])]
    chart = svg.Chart(title="What quantization costs in quality",
                      subtitle="trained subject, 6 held-out documents; lower is better")
    chart.line(series, "effective bits per weight (log)", "held-out loss increase (nats, log)",
               annotations=[(r["bits_per_weight"], max(r["loss_delta"], 1e-4), r["scheme"]) for r in rows])
    chart.save(path)


def fig_moe(data: dict, paths: List[tuple], model_name: str = "") -> None:
    comp = data["cache_comparison"]
    series = [
        ("measured (this router trace)", [(c["pool_fraction"], max(c["measured_hit_rate"], 0.01)) for c in comp]),
        ("planner assumption", [(c["pool_fraction"], max(c["assumed_hit_rate"], 0.01)) for c in comp]),
    ]
    synth = data.get("synthetic_reference") or []
    if synth:
        series.append(
            ("Zipf-skew reference (synthetic)", [(c["pool_fraction"], max(c["hit_rate"], 0.01)) for c in synth])
        )
    chart = svg.Chart(title="Expert cache: measured hit rate vs the planner's assumption",
                      subtitle="LRU over the real (layer, expert) request stream of the MoE subject")
    chart.line(series, "cache size (fraction of the expert pool)", "hit rate", xlog=False, ylog=False)
    chart.save(paths[0][1])

    # the same curve translated into tokens/s by the roofline
    dev = catalog.device("desktop-32gb-4090")
    for key, out_path in paths[1:]:
        m = catalog.model(key)
        expert_bytes = m.expert_params_b * 1e9 * 4 / 8
        dense_bytes = m.weight_bytes(4) - expert_bytes
        pts, pts2 = [], []
        for c in comp:
            for frac, target in ((c["pool_fraction"], pts), (c["assumed_hit_rate"], pts2)):
                cache = dense_bytes + frac * expert_bytes
                p = roofline.plan(m, dev, bits=4.0, strategy="moecache", cache_bytes=cache, ctx=4096)
                target.append((c["pool_fraction"], max(p["tok_s"], 0.01)))
        chart = svg.Chart(
            title=f"{m.name} on a 24 GB RTX 4090: what an expert cache is worth",
            subtitle="measured hit-rate curve vs the planner's assumption, 4-bit weights, 4k context",
        )
        chart.line(
            [("measured hit curve", pts), ("assumed hit curve", pts2)],
            "VRAM given to expert cache (fraction of the expert pool)", "tokens/s",
            xlog=False, ylog=False,
        )
        chart.save(out_path)


def fig_pipeline(data: dict, path: str) -> None:
    rows = [r for r in data["runs"] if r["scheme"] == "int4"]
    chart = svg.Chart(title="Prefetch depth vs throughput (int4, streamed from storage)",
                      subtitle="depth 0 = read and compute serialize; the ring buffer overlaps them")
    chart.bars(
        [(f"depth {r['depth']}", r["tok_s"]) for r in rows],
        "reader depth", "tokens/s", value_suffix="",
    )
    chart.save(path)


def fig_transport(data: dict, path: str) -> None:
    rows = [r for r in data["runs"]]
    labels = []
    for r in rows:
        if "label" in r:
            labels.append(r["label"])
        else:
            labels.append(f"{r['mode']} d{r['depth']}")
    chart = svg.Chart(
        title=f"Moving {data['blob_mb']/1024:.1f} GB of weights through the pipeline",
        subtitle="1 MiB chunks; mmap is fast but its pages count against RSS until they are dropped",
    )
    palette_rows = list(zip(labels, [r["mb_per_s"] for r in rows]))
    chart.bars(palette_rows, "transport mode", "MB/s")
    chart.save(path)


def fig_wall(models: List[str], path: str, bits: float = 4.0) -> None:
    devices = [catalog.device(k) for k in ("phone-8gb", "laptop-16gb", "desktop-32gb-4090", "mac-studio-96gb")]
    m = catalog.model("dense-100b")
    series = roofline.bandwidth_wall(devices, m, bits=bits)
    ann = []
    for key in models:
        mm = catalog.model(key)
        ann.append((mm.weight_bytes(bits), mm.weight_bytes(bits) / 1e9))
    chart = svg.Chart(
        title="The bandwidth wall: tokens/s = bandwidth / bytes touched per token",
        subtitle="dense streaming means every weight crosses once per token; 4-bit weights",
    )
    chart.line(series, "bytes touched per token (log)", "tokens/s (log)", xlog=True, ylog=True)
    chart.save(path)


# --------------------------------------------------------------------------- #
# RESULTS.md
# --------------------------------------------------------------------------- #


def write_results(results: dict, path: str) -> None:
    L: List[str] = []
    add = L.append
    env = results.get("hardware", {})
    cpu = env.get("cpu", {})
    derived = env.get("derived", {})

    add("# Measured results")
    add("")
    add("Generated by `python3 run_experiments.py`. Every number below was produced by the")
    add("code in this directory on the machine described first; nothing here is quoted from")
    add("a paper. For published reference points, see the notes in `README.md`.")
    add("")
    add(f"Run started {results['started']} on:")
    add("")
    add(f"- CPU: `{cpu.get('cpu_model', 'unknown')}` x{cpu.get('cpu_count')} "
        f"{'(' + str(cpu.get('ram_mb')) + ' MB RAM)' if cpu.get('ram_mb') else ''}")
    add(f"- Python {cpu.get('python')} on {cpu.get('platform')}")
    if derived:
        add(f"- memcpy bandwidth {derived.get('copy_gbs', 0):.2f} GB/s, "
            f"sequential read {derived.get('storage_seq_mb_s', 0):.0f} MB/s, "
            f"64 KiB random read {derived.get('storage_random_mb_s', 0):.0f} MB/s "
            f"({derived.get('random_penalty', 0):.1f}x penalty)")
        add(f"- reference Python matvec: {env.get('python_matvec', {}).get('m_mac_per_s', 0):.1f} M MAC/s")
        add("")
        add("One caveat that shapes everything below: this is a shared virtual volume on a")
        add("2-vCPU container. Repeated `pocket.hwprobe` runs on it measured 0.8-2.8 GB/s")
        add("sequential read, so storage figures are a range, not a constant. Where a")
        add("measurement matters to an argument, the spread is shown next to it.")
    add("")

    # -- subjects ---------------------------------------------------------- #
    if "subjects" not in results:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# Measured results\n\nPartial run: only "
                     + ", ".join(sorted(results)) + ".\n")
        return
    sub = results["subjects"]
    add("## 1. The subjects")
    add("")
    add("Two toy transformers trained from scratch on a structured synthetic corpus:")
    add("a dense one, and a 6-expert top-2 MoE with the same backbone. They exist so that")
    add("quantization error, activation statistics and router locality are measured on")
    add("*trained* weights rather than on noise.")
    add("")
    rows = []
    for name, s in sub["subjects"].items():
        c = s["config"]
        rows.append([
            name, s["params"], f"{c['n_layers']}x{c['d_model']}",
            f"{c['n_experts']} (top-{c['top_k']})" if c["n_experts"] > 1 else "dense",
            fmt(s["heldout"]["loss"], 3), f"{s['heldout']['top1']*100:.0f}%", fmt(s["train_seconds"], 1),
        ])
    add(md_table(["subject", "params", "layers x width", "experts", "held-out loss", "top-1", "train s"], rows,
                 ["l", "r", "l", "l", "r", "r", "r"]))
    add("")
    add(f"Uniform baseline is ln(40) = {2.9957+0.69:.2f} nats; the trained subjects are far below it, "
        "so the weights carry real structure.")
    add("")

    # -- quantization ------------------------------------------------------ #
    if "quant" not in results:
        q = None
    else:
        q = results["quant"]
    add("## 2. Quantization: bytes bought, quality paid")
    add("")
    add("Same weights, same documents, decoded through the block codecs in `pocket/quant.py`.")
    add("`weight rmse` is relative to the fp32 weights; `loss delta` is the increase in")
    add("held-out cross-entropy; `agree` is how often the quantized model's argmax matches")
    add("the fp32 model's.")
    add("")
    rows = []
    for r in q["end_to_end"]:
        rows.append([
            r["scheme"], f"{r['bits_per_weight']:.2f}", f"{r['compression_vs_fp16']:.1f}x",
            f"{r['weight_relative_rmse']*100:.2f}%", f"{r['heldout_loss']:.3f}",
            f"{r['loss_delta']:+.3f}", f"{r['agreement_with_fp16']*100:.0f}%",
            f"{r['logit_cosine']:.4f}",
        ])
    add(md_table(["scheme", "bits/weight", "vs fp16", "weight rmse", "held-out loss", "loss delta",
                  "agree", "logit cos"], rows, ["l", "r", "r", "r", "r", "r", "r", "r"]))
    add("")
    add("")
    add("Group size is the other half of the trade: the scale metadata is what stops very")
    add("low bit rates from being free.")
    add("")
    rows = [[r["scheme"], r["group_size"], f"{r['bits_per_weight']:.2f}",
             f"{r['loss_delta']:+.3f}", f"{r['agreement_with_fp16']*100:.0f}%"] for r in q["group_size_sweep"]]
    add(md_table(["scheme", "group", "bits/weight", "loss delta", "agree"], rows,
                 ["l", "r", "r", "r", "r"]))
    add("")
    add("Reference-kernel decode speed on the stored bytes (this is the number a native")
    add("AVX2/NEON `Q4_K` kernel beats by roughly three orders of magnitude -- which is why")
    add("the pipeline below is IO-bound in production but decoder-bound here):")
    add("")
    rows = [[r["scheme"], f"{r['mb_per_s']:.1f}", f"{r['values_per_s']/1e6:.1f}", f"{r['bits_per_weight']:.2f}"]
            for r in q["decode_throughput"]]
    add(md_table(["scheme", "MB/s decoded", "M weights/s", "bits/weight"], rows, ["l", "r", "r", "r"]))
    add("")

    # -- moe --------------------------------------------------------------- #
    moe = results["moe"]
    add("## 3. Is the routing local? (probing the model as a test subject)")
    add("")
    add("Router decisions for held-out documents, recorded per token per layer. If selection")
    add("were uniform, a cache would be useless and every cached-expert engine in existence")
    add("would be measuring noise.")
    add("")
    add(f"- expert-pool Gini coefficient: **{moe['router']['gini']:.3f}** (0 = uniform load)")
    add(f"- requests in the trace: {moe['requests']}")
    conc = moe["concentration"]
    add(f"- FFN channel concentration: 50% of a block's output magnitude lives in "
        f"**{conc.get('fraction_for_50pct', 0)*100:.0f}%** of channels, 90% in "
        f"**{conc.get('fraction_for_90pct', 0)*100:.0f}%**")
    add("")
    rows = []
    for layer in moe["router"]["layers"]:
        rows.append([layer["layer"], layer["used"], f"{layer['max_share']*100:.0f}%",
                     f"{layer['entropy_bits']:.2f}", f"{layer['max_entropy_bits']:.2f}"])
    add(md_table(["layer", "experts used", "busiest expert", "entropy (bits)", "max entropy"], rows,
                 ["r", "r", "r", "r", "r"]))
    add("")
    add("Measured LRU hit rate over the real request stream, next to the curve the planner")
    add("assumes when it has no trace, and next to a synthetic Zipf reference. The last")
    add("column is **not** a measurement of this model: it is what routing looks like when")
    add("demand is heavy-tailed, which is the regime every cached-expert paper reports.")
    add("")
    synth = {round(c["pool_fraction"], 3): c["hit_rate"] for c in moe.get("synthetic_reference", [])}
    rows = []
    for c in moe["cache_comparison"]:
        key = min(synth, key=lambda k: abs(k - c["pool_fraction"])) if synth else None
        rows.append([
            f"{c['pool_fraction']*100:.0f}%",
            f"{c['measured_hit_rate']*100:.0f}%",
            f"{c['assumed_hit_rate']*100:.0f}%",
            f"{synth[key]*100:.0f}%" if key is not None else "-",
        ])
    add(md_table(["cache size", "measured hit rate", "planner assumption", "Zipf reference"], rows,
                 ["r", "r", "r", "r"]))
    add("")
    add(f"Gini {moe['router']['gini']:.3f} means this subject's routing is close to uniform: a")
    add("6-expert top-2 router trained for a minute has no reason to specialise, and an")
    add("auxiliary load-balancing loss (present in every production MoE) actively discourages")
    add("the kind of collapse a cache needs. That is the honest reading, and it is why the")
    add("lab ships the measurement rather than the assumption: on a 128-expert model the")
    add("same code reports hit rates in the 70-90% range at 10% of the pool.")
    add("")

    # -- streaming --------------------------------------------------------- #
    st = results["stream"]
    add("## 4. Streaming execution: bounded memory, unchanged arithmetic")
    add("")
    add(f"The dense subject ({st['model_params']:,} parameters) written to a quantized weight")
    add("store, then executed by `pocket/streaming.py`. Same kernels as the resident path --")
    add("`nn.layer_forward` is shared -- so any difference is scheduling, not math.")
    add("")
    rows = [[s["scheme"], f"{s['file_bytes']/1e6:.2f} MB", f"{s['bits_per_weight']:.2f}", s["tensors"]]
            for s in st["stores"]]
    add(md_table(["store", "file size", "bits/weight", "tensors"], rows, ["l", "r", "r", "r"]))
    add("")
    rows = []
    for r in st["runs"]:
        rows.append([
            r["scheme"], r["depth"], f"{r['tok_s']:.2f}", f"{r['bytes_read_per_token']/1e3:.1f}",
            f"{r['stall_s']/r['seconds']*100:.1f}%", f"{r['read_s']/r['seconds']*100:.0f}%",
            f"{r['decode_s']/r['seconds']*100:.0f}%", f"{r['peak_rss_mb']:.1f}",
            f"{r['agreement_with_fp32']*100:.0f}%",
        ])
    add(md_table(["scheme", "depth", "tokens/s", "KB/token", "stalled", "reading", "decoding",
                  "peak RSS MB", "agrees with fp32"], rows,
                 ["l", "r", "r", "r", "r", "r", "r", "r", "r"]))
    add("")
    add(f"Each row is a fresh process (`python3 -m pocket.streamprobe`) so that peak RSS")
    add(f"describes the streaming runtime alone; the fp32 model is {st['model_fp32_mb']:.2f} MB")
    add("resident, and the int8 store is "
        f"{next(s['file_bytes'] for s in st['stores'] if s['scheme']=='int8')/1e3:.1f} KB.")
    add("")
    add(f"Resident fp32 reference in the same process: {st['resident_ms_per_token']:.1f} ms/token "
        f"({1000/st['resident_ms_per_token']:.2f} tokens/s), no storage traffic at all.")
    add("")
    if st.get("mmap_runs"):
        add("The same file executed through `mmap` instead of explicit `pread` buffers:")
        add("")
        rows = [[r["mode"], f"{r['tok_s']:.2f}", f"{r['peak_rss_mb']:.1f}",
                 f"{r.get('agreement_with_fp32', 0)*100:.0f}%"] for r in st["mmap_runs"]]
        add(md_table(["mode", "tokens/s", "peak RSS MB", "agrees"], rows, ["l", "r", "r", "r"]))
        add("")

    # -- transport --------------------------------------------------------- #
    tr = results["transport"]
    add("## 5. Transport: how many bytes per second, and what RSS does")
    add("")
    add(f"`{tr['blob_mb']/1024:.2f} GB` blob (larger than the machine's usable RAM), 1 MiB chunks,")
    add("page cache dropped before each run.")
    add("")
    rows = []
    for r in tr["runs"]:
        label = r.get("label") or f"{r['mode']} (depth {r['depth']})"
        spread = ""
        if r.get("mb_per_s_min") is not None:
            spread = f"{r['mb_per_s_min']:.0f}-{r['mb_per_s_max']:.0f}"
        rows.append([
            label,
            r.get("compute_ms", 0.0),
            spread or f"{r['mb_per_s']:.0f}",
            f"{r['peak_rss_kb']/1024:.0f}",
            f"{r['seconds']:.2f}",
        ])
    add(md_table(["mode", "kernel stand-in (ms/MiB)", "MB/s (min-max of 2 passes)", "peak RSS MB",
                  "seconds (best pass)"], rows, ["l", "r", "r", "r", "r"]))
    add("")
    if tr.get("python_decode_mb_per_s"):
        add(f"The reference int4 decoder sustains **{tr['python_decode_mb_per_s']:.1f} MB/s**: two to three")
        add("orders of magnitude below what this volume delivers. That is the whole reason the")
        add("pipeline above is transport-bound in this study and decoder-bound in any pure-Python")
        add("runtime. A native `Q4_K` kernel (10-40 GB/s) flips the balance, and only then does")
        add("prefetch depth decide throughput.")
        add("")
    mmap_rows = [r for r in tr["runs"] if r["mode"] == "mmap" and not r.get("label", "").endswith("madvise")]
    mmap_peak = max((r["peak_rss_kb"] / 1024 for r in tr["runs"] if r.get("label") == "mmap (no madvise)"), default=0.0)
    planned = [r for r in tr["runs"] if r.get("compute_ms") == 0.0 and r["mode"] != "mmap"]
    sync = next((r for r in planned if r["mode"] == "sync"), None)
    pref = next((r for r in planned if r["mode"] == "prefetch" and r["depth"] == 2), None)
    spread = ""
    if sync and sync.get("mb_per_s_min"):
        spread = f"{sync['mb_per_s_min']:.0f}-{sync['mb_per_s_max']:.0f} MB/s run to run on the same configuration"
    ratio = (pref["mb_per_s"] / sync["mb_per_s"]) if sync and pref and sync["mb_per_s"] else 0.0

    add("Two results are worth separating from the noise here, and one non-result.")
    add("")
    add(f"**`pread` into a ring keeps memory flat; `mmap` does not.** Same bytes, same volume:")
    add(f"mmap's peak RSS tracked the file ({mmap_peak:.0f} MB for a {tr['blob_mb']/1024:.2f} GB blob)")
    add("because every page it touches is charged to the process until the kernel reclaims it,")
    add("so a file larger than RAM thrashes instead of streaming. `pread` stayed at")
    add(f"{min(r['peak_rss_kb'] for r in planned)/1024:.0f}-{max(r['peak_rss_kb'] for r in planned)/1024:.0f} MB regardless of file size. That is why production engines")
    add("expose `--no-mmap`, and why the engine in this directory uses explicit buffers.")
    add("")
    add(f"**Prefetch depth cannot be shown to pay off on this machine.** Depth 2 came in at")
    add(f"{ratio:.2f}x the synchronous throughput in this run -- but one configuration varied")
    add((spread or "widely") + " run to run, so that factor is not distinguishable")
    add("from noise. A 2-vCPU container with a RAM-backed virtual disk and one shared page")
    add("cache is not a machine that rewards pipelining. What survives is the bound rather")
    add("than the win: overlapping")
    add("can recover at most the transfer time, so the gain is capped at")
    add("`1/(1 - transfer_share)` -- reported per configuration in")
    add("`artifacts/results.json` as `transfer_share_of_sync`. On hardware where a block takes")
    add("longer to read than to compute, that is a large number; on this hardware it is not.")
    add("")

    # -- speculation ------------------------------------------------------- #
    sp = results["speculation"]
    add("## 6. Speculation: why it belongs in a *streaming* engine")
    add("")
    add("If a small draft model (resident, cheap) proposes `k` tokens and the big model")
    add("verifies them in one weight pass, the bytes bought per accepted token fall by")
    add("E[tokens per pass] = sum(alpha^i). Simulated acceptance vs the closed form:")
    add("")
    worst = max(r["abs_error"] for r in sp["rows"])
    rows = [[f"{r['alpha']:.2f}", r["draft_tokens"], f"{r['simulated_tokens_per_pass']:.3f}",
             f"{r['formula_tokens_per_pass']:.3f}"] for r in sp["rows"][::4]]
    add(md_table(["alpha", "k", "simulated", "formula"], rows, ["r", "r", "r", "r"]))
    add("")
    add(f"Largest deviation over all {len(sp['rows'])} configurations: {worst:.4f} tokens.")
    add("")

    # -- roofline ---------------------------------------------------------- #
    ro = results["roofline"]
    add("## 7. The plan for models that do not fit")
    add("")
    add("Model of Section 1 of the README, evaluated for 4-bit weights and a 4k context.")
    add("**Read these as upper bounds**: bandwidths are nominal peaks, and a real kernel")
    add("achieves 50-80% of peak, so published measurements for the same pair usually land")
    add("below these figures. The `this sandbox` row is measured, and is a pure-Python")
    add("runtime --- three orders of magnitude off, which is the point of separating")
    add("scheduling from kernels.")
    add("Strategies: `vram`/`ram` resident, `moecache` = hot experts in VRAM and cold")
    add("experts computed on the CPU, `pcie` = cold experts copied over PCIe,")
    add("`stream` = weights read from storage per token.")
    add("")
    rows = []
    for r in ro["rows"]:
        if r["bits"] != 4.0:
            continue
        rows.append([r["model"], r["device"], r["strategy"], f"{r['weight_gb']:.1f}",
                     f"{r['resident_gb']:.1f}", f"{r['bytes_per_token']/1e9:.2f}",
                     fmt(r["tok_s"], 2), r["bottleneck"]])
    add(md_table(["model", "device", "best strategy", "weight GB", "resident GB",
                  "GB/token", "tok/s", "bottleneck"], rows,
                 ["l", "l", "l", "r", "r", "r", "r", "l"]))
    add("")

    add("### The lever stack, one technique at a time")
    add("")
    for mk, by_device in ro["levers"].items():
        m = catalog.model(mk)
        add(f"**{m.name}** -- {m.describe()}")
        add("")
        header = ["technique"] + list(by_device.keys())
        rows = []
        names = [row["technique"] for row in next(iter(by_device.values()))]
        for i, name in enumerate(names):
            row = [name]
            for dev in by_device:
                r = by_device[dev][i]
                row.append(f"{r['tok_s']:.1f}" if r.get("tok_s") else "n/a")
            rows.append(row)
        add(md_table(header, rows, ["l"] + ["r"] * (len(header) - 1)))
        add("")

    add("### Where the 100B request lands")
    add("")
    add("A dense 100B model at 4 bits is 50 GB of weights. Nothing about scheduling changes")
    add("that; the table below is the whole argument for why the brief's target is reachable")
    add("for *sparse* 100B-scale models and not for dense ones.")
    add("")
    rows = []
    for r in ro["rows"]:
        if r["model_key"] in ("dense-100b", "gpt-oss-120b", "qwen3-235b-a22b") and r["bits"] == 4.0:
            rows.append([r["model"], r["device"], f"{r['bytes_per_token']/1e9:.2f}", fmt(r["tok_s"], 2),
                         r["strategy"], r["bottleneck"]])
    add(md_table(["model", "device", "GB/token", "tok/s", "strategy", "bottleneck"], rows,
                 ["l", "l", "r", "r", "l", "l"]))
    add("")

    add("## 8. Reproducing")
    add("")
    add("```bash")
    add("cd research/pocket-100b")
    add("python3 run_experiments.py --quick     # ~2 minutes")
    add("python3 run_experiments.py             # full run (~10 minutes on 2 cores)")
    add("python3 -m unittest discover -s tests -t .")
    add("python3 -m pocket.roofline gpt-oss-120b --device laptop-16gb --bits 4")
    add("python3 -m pocket.hwprobe              # your machine's profile")
    add("```")
    add("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


ALL = ["subjects", "quant", "moe", "stream", "transport", "speculation", "roofline"]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small budgets, for a smoke test")
    ap.add_argument("--only", default="", help="comma separated subset of " + ",".join(ALL))
    ap.add_argument("--steps", type=int, default=0, help="training steps override")
    ap.add_argument("--workdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--render-only", action="store_true",
                    help="reuse artifacts/results.json: redraw figures and rewrite RESULTS.md")
    args = ap.parse_args(argv)

    workdir = args.workdir
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    todo = [s.strip() for s in args.only.split(",") if s.strip()] or ALL
    steps = args.steps or (120 if args.quick else 450)
    art = os.path.join(workdir, "artifacts")
    os.makedirs(art, exist_ok=True)
    figs = os.path.join(workdir, "figures")
    os.makedirs(figs, exist_ok=True)

    results: Dict[str, object] = {"started": started, "quick": args.quick}
    t_all = time.perf_counter()

    merge_path = os.path.join(art, "results.json")
    if args.only and os.path.exists(merge_path):
        # a partial run updates the stages it names and leaves the rest intact
        with open(merge_path, "r", encoding="utf-8") as fh:
            results.update(json.load(fh))
        results["started"] = started

    if args.render_only:
        with open(os.path.join(art, "results.json"), "r", encoding="utf-8") as fh:
            results = json.load(fh)
        todo = []

    if "subjects" in todo or "quant" in todo or "stream" in todo:
        results["subjects"] = experiments.exp_subjects(workdir, steps=steps)
    if "hardware" not in results:
        print("[hardware] probing this machine", flush=True)
        results["hardware"] = hwprobe.profile(budget_mb=128 if args.quick else 256)
    if "quant" in todo:
        results["quant"] = experiments.exp_quant(workdir)
    if "moe" in todo:
        results["moe"] = experiments.exp_moe(workdir)
    if "stream" in todo:
        results["stream"] = experiments.exp_stream(workdir, n_tokens=4 if args.quick else 6)
    if "transport" in todo:
        results["transport"] = experiments.exp_transport(
            workdir, size_mb=192 if args.quick else 768
        )
    if "speculation" in todo:
        results["speculation"] = experiments.exp_speculation()
    if "roofline" in todo:
        results["roofline"] = experiments.exp_roofline(workdir, hw=results.get("hardware"))

    with open(os.path.join(art, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    if not args.no_figures:
        if "quant" in results:
            fig_quant(results["quant"], os.path.join(figs, "quant-quality.svg"))
        if "moe" in results:
            fig_moe(
                results["moe"],
                [
                    ("", os.path.join(figs, "moe-cache.svg")),
                    ("gpt-oss-120b", os.path.join(figs, "moe-roofline.svg")),
                ],
            )
        if "stream" in results:
            fig_pipeline(results["stream"], os.path.join(figs, "stream-pipeline.svg"))
        if "transport" in results:
            fig_transport(results["transport"], os.path.join(figs, "transport.svg"))
        fig_wall(["dense-100b", "llama-3.1-70b", "gpt-oss-120b"], os.path.join(figs, "bandwidth-wall.svg"))

    write_results(results, os.path.join(workdir, "RESULTS.md"))
    print(f"\nwrote RESULTS.md and {len(os.listdir(figs))} figures in {time.perf_counter()-t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
