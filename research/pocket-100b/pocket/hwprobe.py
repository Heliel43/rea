"""Measure the machine you are actually on, because the plan depends on it.

The roofline model in ``roofline.py`` needs four numbers per device: usable RAM,
sustained storage read bandwidth, memory bandwidth, and available FLOPs.  Vendors
publish the first-order specs; this module measures what a container or laptop
actually delivers, which is routinely 2-5x worse than the datasheet -- especially
for random reads, which is what expert offloading turns into.

Every probe is bounded (``--budget-mb``) so it can be run on a phone.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from typing import Dict, List


def cpu_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for line in text.splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        cache = {}
        for line in text.splitlines():
            for key in ("cache size", "cache_alignment"):
                if line.startswith(key):
                    cache[key] = line.split(":", 1)[1].strip()
        info["cpu_cache"] = cache
    except OSError:  # pragma: no cover
        pass
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["ram_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    info["ram_available_mb"] = int(line.split()[1]) // 1024
    except OSError:  # pragma: no cover
        pass
    return info


def memcpy_bandwidth(mb: int = 128, reps: int = 5) -> Dict[str, float]:
    """Copy bandwidth: the cheapest honest proxy for streaming memory bandwidth."""

    src = bytes(mb * 1024 * 1024)
    dst = bytearray(len(src))
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        dst[:] = src
        dt = time.perf_counter() - t0
        # one read + one write per byte
        gbs = (2 * len(src) / 1e9) / dt
        best = max(best, gbs)
    read_only_s = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        n = src.count(0)
        dt = time.perf_counter() - t0
        read_only_s = max(read_only_s, (len(src) / 1e9) / dt)
    return {"copy_gbs": best, "read_gbs": read_only_s, "bytes_probed": len(src)}


def _make_file(path: str, mb: int) -> None:
    if os.path.exists(path) and os.path.getsize(path) >= mb * 1024 * 1024:
        return
    with open(path, "wb") as fh:
        block = bytes(1 << 20)
        for _ in range(mb):
            fh.write(block)


def _drop_cache(fd: int, size: int) -> None:
    try:
        os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):  # pragma: no cover
        pass


def sequential_read(path: str, chunk: int = 1 << 20, mb: int = 256) -> Dict[str, float]:
    size = min(os.path.getsize(path), mb * 1024 * 1024)
    fd = os.open(path, os.O_RDONLY)
    try:
        _drop_cache(fd, size)
        t0 = time.perf_counter()
        off = 0
        while off < size:
            buf = os.pread(fd, min(chunk, size - off), off)
            if not buf:
                break
            off += len(buf)
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
    return {"mb_per_s": (size / 1e6) / dt, "chunk_bytes": chunk, "bytes": size, "seconds": dt}


def random_read(path: str, reads: int = 256, block: int = 65536) -> Dict[str, float]:
    """Random 64 KiB reads -- the pattern a cache miss on an expert produces."""

    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    rng = random.Random(0)
    offsets = [rng.randrange(0, max(1, size - block) // 4096) * 4096 for _ in range(reads)]
    try:
        _drop_cache(fd, size)
        t0 = time.perf_counter()
        total = 0
        for off in offsets:
            total += len(os.pread(fd, block, off))
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
    return {
        "iops": reads / dt,
        "mb_per_s": (total / 1e6) / dt,
        "block_bytes": block,
        "mean_latency_ms": 1000 * dt / reads,
    }


def python_matvec_rate(dim: int = 64, reps: int = 3000) -> Dict[str, float]:
    """Reference-kernel throughput, so experiments can separate algorithm from
    interpreter overhead."""

    from operator import mul

    a = [0.5] * dim
    b = [0.25] * dim
    t0 = time.perf_counter()
    for _ in range(reps):
        sum(map(mul, a, b))
    dt = time.perf_counter() - t0
    macs = dim * reps
    return {"m_mac_per_s": macs / dt / 1e6, "dim": dim}


def storage_bw(path: str = "/tmp/pocket-hwprobe.bin", mb: int = 256) -> Dict[str, object]:
    _make_file(path, mb)
    out: Dict[str, object] = {}
    for chunk in (1 << 20, 1 << 22, 1 << 24):
        out[f"seq_{chunk >> 10}k"] = sequential_read(path, chunk=chunk, mb=mb)
    out["random_64k"] = random_read(path, reads=192, block=65536)
    return out


def profile(budget_mb: int = 256, workdir: str = "/tmp") -> Dict[str, object]:
    t0 = time.perf_counter()
    out: Dict[str, object] = {"cpu": cpu_info(), "memory": memcpy_bandwidth(mb=min(128, budget_mb))}
    out["storage"] = storage_bw(os.path.join(workdir, "pocket-hwprobe.bin"), mb=budget_mb)
    out["python_matvec"] = python_matvec_rate()
    out["probe_seconds"] = time.perf_counter() - t0
    seq = out["storage"]["seq_1024k"]["mb_per_s"]
    rnd = out["storage"]["random_64k"]["mb_per_s"]
    out["derived"] = {
        "storage_seq_mb_s": seq,
        "storage_random_mb_s": rnd,
        "random_penalty": seq / rnd if rnd else 0.0,
        "copy_gbs": out["memory"]["copy_gbs"],
    }
    return out


def main(argv: List[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Measure the inference-relevant hardware profile")
    ap.add_argument("--budget-mb", type=int, default=256, help="file size for storage probes")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    prof = profile(budget_mb=args.budget_mb)
    text = json.dumps(prof, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if args.json:
        print(text)
    else:
        c = prof["cpu"]
        print(f"cpu           {c.get('cpu_model', '?')} x{c.get('cpu_count')}  ram {c.get('ram_mb')} MB")
        d = prof["derived"]
        print(f"copy bw       {d['copy_gbs']:.2f} GB/s   read-only {prof['memory']['read_gbs']:.2f} GB/s")
        print(f"storage seq   {d['storage_seq_mb_s']:.0f} MB/s")
        print(f"storage rand  {d['storage_random_mb_s']:.0f} MB/s (64 KiB reads, {d['random_penalty']:.1f}x penalty)")
        print(f"python matvec {prof['python_matvec']['m_mac_per_s']:.1f} M MAC/s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
