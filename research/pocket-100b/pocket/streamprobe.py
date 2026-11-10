"""Run one streamed generation in a fresh process and report its footprint.

Why a subprocess: ``VmHWM`` is the process's high-water mark for its whole life,
so a streamed run inside a process that also holds a resident fp32 reference model
reports the wrong number.  The only honest way to answer "how much memory does the
streaming engine need" is to start from nothing.

    python3 -m pocket.streamprobe --store artifacts/x.int4.pksw --depth 1 --tokens 6
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional

from . import streaming
from .store import MappedStore


def run(store_path: str, prompt: List[int], n_tokens: int = 6, depth: int = 1, threads: int = 1,
        use_mmap: bool = False, release_mmap: bool = False) -> dict:
    streaming.reset_peak_rss()
    rss0 = streaming.rss_kb()
    t_import = time.perf_counter()
    with MappedStore(store_path, use_mmap=use_mmap) as ms:
        model = streaming.StreamedModel(ms, depth=depth, threads=threads)
        t_start = time.perf_counter()
        tokens = model.generate(prompt, n_tokens=n_tokens)
        elapsed = time.perf_counter() - t_start
        peak = streaming.peak_rss_kb()
        stats = {
            "tok_s": n_tokens / elapsed,
            "seconds": elapsed,
            "tokens": tokens,
            "bytes_read": model.pipe.bytes_read,
            "bytes_read_per_token": model.pipe.bytes_read / n_tokens,
            "stall_s": model.pipe.stall_s,
            "read_s": model.pipe.read_s,
            "decode_s": model.pipe.decode_s,
            "peak_rss_mb": peak / 1024,
            "rss_start_mb": rss0 / 1024,
            "store_file_mb": os.path.getsize(store_path) / 1e6,
            "params": ms.cfg.n_layers,
            "depth": depth,
            "threads": threads,
            "mode": "mmap" if use_mmap else "pread",
            "layer_compute_s": sum(s["compute_s"] for s in model.layer_stats),
            "layer_load_s": sum(s["load_s"] for s in model.layer_stats),
        }
        model.close()
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="One streamed generation in a clean process")
    ap.add_argument("--store", required=True)
    ap.add_argument("--prompt", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mmap", action="store_true")
    args = ap.parse_args(argv)
    prompt = [int(x) for x in args.prompt.split(",") if x.strip()]
    out = run(args.store, prompt, n_tokens=args.tokens, depth=args.depth, threads=args.threads,
              use_mmap=args.mmap)
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
