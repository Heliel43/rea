"""Streaming execution: run a model whose weights do not fit in RAM.

The scheduling problem, stated plainly
--------------------------------------
Autoregressive decoding touches every weight once per token.  If the weights live
on flash, then for *any* device:

    tokens/s  <=  bytes/s(storage -> compute)  /  bytes touched per token

Everything in this module is about the right-hand side.  Three mechanisms:

1. **Bounded residency.**  Only one block's weights are decoded at a time.  Peak
   RSS must not grow with model size, or the whole exercise is pointless.
2. **Prefetch with a ring of buffers.**  A worker thread reads block ``i+1``
   while the main thread computes block ``i``.  With depth 1 the reader and the
   arithmetic serialize; the measured difference is the "overlap efficiency" in
   RESULTS.md.
3. **Explicit reads, not mmap.**  ``mmap`` looks free because the kernel hides the
   read, but the mapped pages count against RSS until the kernel decides to
   reclaim them, so a model larger than RAM thrashes instead of streaming.  This
   module therefore uses ``pread`` into reusable buffers and keeps ``mmap`` only
   as a comparison mode.

Nothing here decides *which* weights to touch -- that is the planner's job (see
``roofline.py``); this module executes a plan and reports what it cost.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import nn, quant
from .nn import ModelConfig
from .store import MappedStore

PAGE = 4096


# --------------------------------------------------------------------------- #
# Process accounting
# --------------------------------------------------------------------------- #


def rss_kb() -> int:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:  # pragma: no cover - non-Linux
        pass
    return 0


def peak_rss_kb() -> int:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:  # pragma: no cover
        pass
    return 0


def reset_peak_rss() -> None:
    """VmHWM is monotonic for the process; writing to it resets the high-water mark."""

    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as fh:
            fh.write("5")
    except OSError:  # pragma: no cover - needs no privilege on most kernels
        pass


# --------------------------------------------------------------------------- #
# Layer pipeline
# --------------------------------------------------------------------------- #


class LayerResult:
    __slots__ = ("params", "raw_bytes", "read_s", "decode_s", "stall_s")

    def __init__(self, params: Dict[str, List[float]], raw_bytes: int, read_s: float, decode_s: float, stall_s: float):
        self.params = params
        self.raw_bytes = raw_bytes
        self.read_s = read_s
        self.decode_s = decode_s
        self.stall_s = stall_s


class LayerPipeline:
    """Reads and decodes blocks ahead of the compute thread.

    ``layers`` is the ordered list of block indices the runtime will ask for.
    Depth 1 means "read the next block while computing this one"; depth 0 means
    fully synchronous (the baseline).
    """

    def __init__(
        self,
        store: MappedStore,
        order: Sequence[str],
        depth: int = 1,
        threads: int = 1,
        decode: bool = True,
    ):
        self.store = store
        self.depth = depth
        self.threads = max(1, threads)
        self.decode = decode
        self.order = list(order)
        self._slots: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._closed = False
        self.bytes_read = 0
        self.read_s = 0.0
        self.decode_s = 0.0
        self.stall_s = 0.0
        self.waits = 0
        # Slots are unbounded: the *window* (below) is what bounds memory, and
        # bounding both is how a reader deadlocks against its own consumer.
        for name in self.order:
            self._slots[name] = queue.Queue()
        self._pending = queue.Queue()
        self._workers = []
        if self.depth > 0:
            for _ in range(self.threads):
                t = threading.Thread(target=self._worker, daemon=True)
                t.start()
                self._workers.append(t)

    # -- worker ------------------------------------------------------------ #
    def _worker(self) -> None:
        while True:
            item = self._pending.get()
            if item is None:
                return
            name = item
            t0 = time.perf_counter()
            raw = self.store.read_raw(name)
            t1 = time.perf_counter()
            if self.decode:
                rec = self.store.records[name]
                vals = quant.decode(raw, rec.count, rec.scheme, rec.group_size)
            else:
                vals = []
            t2 = time.perf_counter()
            with self._lock:
                self.read_s += t1 - t0
                self.decode_s += t2 - t1
                self.bytes_read += len(raw)
            self._slots[name].put((vals, len(raw), t1 - t0, t2 - t1))

    # -- API --------------------------------------------------------------- #
    def submit(self, name: str) -> None:
        """Queue a tensor for background read.  No-op in synchronous mode."""

        if self.depth > 0:
            self._pending.put(name)

    def take(self, name: str) -> Tuple[List[float], int, float, float, float]:
        """Block until a tensor is available.  Returns (values, bytes, read, decode, stall)."""

        if self.depth == 0:
            t0 = time.perf_counter()
            raw = self.store.read_raw(name)
            t1 = time.perf_counter()
            rec = self.store.records[name]
            vals = quant.decode(raw, rec.count, rec.scheme, rec.group_size) if self.decode else []
            t2 = time.perf_counter()
            with self._lock:
                self.bytes_read += len(raw)
                self.read_s += t1 - t0
                self.decode_s += t2 - t1
            return vals, len(raw), t1 - t0, t2 - t1, 0.0
        t0 = time.perf_counter()
        vals, nbytes, read_s, decode_s = self._slots[name].get()
        stall = time.perf_counter() - t0
        with self._lock:
            self.stall_s += stall
            self.waits += 1
        return vals, nbytes, read_s, decode_s, stall

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._workers:
            self._pending.put(None)
        for t in self._workers:
            t.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# Streamed forward pass
# --------------------------------------------------------------------------- #


class StreamedModel:
    """Runs the reference kernels with weights pulled block by block."""

    def __init__(self, store: MappedStore, depth: int = 1, threads: int = 1, log: bool = False):
        self.store = store
        self.cfg = store.cfg
        self.log = log
        self.pipe = LayerPipeline(store, self._order(), depth=depth, threads=threads)
        self.layer_stats: List[dict] = []
        self._next_layer = 0

    def _order(self) -> List[str]:
        "Canonical request order: embeddings, blocks, head -- same as the store."

        names = ["token_embd", "pos_embd"]
        for layer in range(self.cfg.n_layers):
            for suffix in ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_out", "ffn_norm"):
                names.append(f"blk.{layer}.{suffix}")
            if self.cfg.is_moe:
                names.append(f"blk.{layer}.router")
            for suffix in ("ffn_gate", "ffn_up", "ffn_down"):
                names.append(f"blk.{layer}.{suffix}")
        names.append("out_norm")
        return names

    def _layer_names(self, layer: int) -> List[str]:
        pre = f"blk.{layer}."
        suffixes = ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_out", "ffn_norm"]
        if self.cfg.is_moe:
            suffixes.append("router")
        suffixes.extend(["ffn_gate", "ffn_up", "ffn_down"])
        return [pre + s for s in suffixes]

    def _prefetch_upto(self, upto: int) -> None:
        """Keep at most ``depth`` blocks queued ahead of the block being computed.

        ``upto`` is an exclusive block index; ``cfg.n_layers`` means "the head".
        """

        limit = min(upto, self.cfg.n_layers + 1)
        while self._next_layer < limit:
            if self._next_layer < self.cfg.n_layers:
                for name in self._layer_names(self._next_layer):
                    self.pipe.submit(name)
            else:
                self.pipe.submit("out_norm")
            self._next_layer += 1

    def forward(self, tokens: Sequence[int]) -> List[List[float]]:
        cfg = self.cfg
        nbytes = 0
        t_start = time.perf_counter()
        self._next_layer = 0  # the lookahead cursor is per forward pass

        # Embeddings are needed first and re-read every token, so queue this
        # token's copy before waiting on it.  (Submissions must always be a
        # superset of takes -- the async reader has no other source of work.)
        self.pipe.submit("token_embd")
        self.pipe.submit("pos_embd")
        tok, nb, *_ = self.pipe.take("token_embd")
        pos, nb2, *_ = self.pipe.take("pos_embd")
        nbytes += nb + nb2
        x = nn.embed_from(cfg, tok, pos, tokens)
        self._prefetch_upto(self.pipe.depth + 1)

        for layer in range(cfg.n_layers):
            t_layer = time.perf_counter()
            values = {}
            stalls = 0.0
            for name in self._layer_names(layer):
                vals, nb, read_s, decode_s, stall = self.pipe.take(name)
                values[name[len(f"blk.{layer}.") :]] = vals
                nbytes += nb
                stalls += stall
            self._prefetch_upto(layer + 1 + self.pipe.depth)
            t_loaded = time.perf_counter()
            x, _ = nn.layer_forward(cfg, layer, values.__getitem__, x)
            t_done = time.perf_counter()
            self.layer_stats.append(
                {"layer": layer, "load_s": t_loaded - t_layer, "compute_s": t_done - t_loaded, "stall_s": stalls}
            )

        self._prefetch_upto(cfg.n_layers + 1)
        out_norm, nb, *_ = self.pipe.take("out_norm")
        nbytes += nb
        self.pipe.submit("out_norm")
        logits, _ = nn.head_from(cfg, x, out_norm, tok)
        return logits

    def generate(self, prompt: Sequence[int], n_tokens: int = 8, greedy: bool = True) -> List[int]:
        seq = list(prompt)
        out: List[int] = []
        for _ in range(n_tokens):
            ctx = seq[-self.cfg.context :]
            logits = self.forward(ctx)
            row = logits[-1]
            tok = max(range(len(row)), key=lambda i: row[i]) if greedy else None
            out.append(tok)
            seq.append(tok)
        return out

    def close(self) -> None:
        self.pipe.close()
        self.store.close()


# --------------------------------------------------------------------------- #
# Transport benchmark: how fast can bytes actually reach the CPU?
# --------------------------------------------------------------------------- #


def _consume_translate(table: bytes) -> Callable[[bytes], int]:
    "A C-speed stand-in for a native dequantize kernel (keeps the bench IO-bound)."

    def consume(buf: bytes) -> int:
        return len(buf.translate(table))

    return consume


def benchmark_transport(
    path: str,
    total_bytes: int,
    chunk_bytes: int = 1 << 20,
    depth: int = 2,
    mode: str = "prefetch",
    drop_cache: bool = False,
    fadvise: bool = True,
    compute_ms: float = 0.0,
    consumer: str = "translate",
) -> dict:
    """Stream ``path`` with a fixed ring of buffers and report throughput + RSS.

    ``mode``: ``sync`` (read + consume in one thread), ``prefetch`` (reader thread,
    ``depth`` buffers), ``mmap`` (map the file, touch every page, optionally drop
    pages behind us).

    ``compute_ms`` inserts a per-chunk cost that *releases the GIL* (``sleep``), which
    is how a native dequantize/matvec kernel behaves: the interpreter is free while
    the kernel runs.  That is the regime where prefetching pays, and the point of
    running it here is to separate "IO is slow" from "our decoder holds the GIL".
    """

    table = bytes(range(256))
    consume = _consume_translate(table)
    def consume_chunk(buf: bytes) -> int:
        n = consume(buf)
        if compute_ms > 0:
            time.sleep(compute_ms / 1000.0)
        return n
    size = min(total_bytes, os.path.getsize(path))
    reset_peak_rss()
    rss0 = rss_kb()

    if drop_cache:
        try:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except (AttributeError, OSError):  # pragma: no cover
            pass

    t0 = time.perf_counter()
    processed = 0
    stalls = 0.0

    if mode == "mmap":
        import mmap as _mmap

        fd = os.open(path, os.O_RDONLY)
        mm = _mmap.mmap(fd, 0, prot=_mmap.PROT_READ)
        try:
            off = 0
            while off < size:
                end = min(off + chunk_bytes, size)
                consume_chunk(mm[off:end])
                processed += end - off
                if fadvise:
                    try:
                        mm.madvise(_mmap.MADV_DONTNEED, off, end - off)
                    except (AttributeError, OSError):  # pragma: no cover
                        pass
                off = end
        finally:
            mm.close()
            os.close(fd)
    elif mode == "sync":
        fd = os.open(path, os.O_RDONLY)
        try:
            off = 0
            while off < size:
                buf = os.pread(fd, min(chunk_bytes, size - off), off)
                if not buf:
                    break
                consume_chunk(buf)
                processed += len(buf)
                off += len(buf)
        finally:
            os.close(fd)
    elif mode == "prefetch":
        fd = os.open(path, os.O_RDONLY)
        ring: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        done = threading.Event()
        reader_busy = [0.0]

        def reader() -> None:
            off = 0
            try:
                while off < size:
                    t_read = time.perf_counter()
                    buf = os.pread(fd, min(chunk_bytes, size - off), off)
                    reader_busy[0] += time.perf_counter() - t_read
                    if not buf:
                        break
                    ring.put(buf)
                    off += len(buf)
            finally:
                done.set()

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        try:
            while True:
                try:
                    buf = ring.get(timeout=0.001)
                except queue.Empty:
                    if done.is_set() and ring.empty():
                        break
                    stalls += 0.001
                    continue
                consume_chunk(buf)
                processed += len(buf)
                if done.is_set() and ring.empty():
                    break
        finally:
            th.join(timeout=5.0)
            os.close(fd)
    else:  # pragma: no cover
        raise ValueError(f"unknown mode {mode!r}")

    elapsed = time.perf_counter() - t0
    return {
        "mode": mode,
        "depth": depth,
        "chunk_bytes": chunk_bytes,
        "bytes": processed,
        "seconds": elapsed,
        "mb_per_s": (processed / 1e6) / elapsed if elapsed else 0.0,
        "rss_start_kb": rss0,
        "peak_rss_kb": peak_rss_kb(),
        "idle_wait_s": stalls,
        "drop_cache": drop_cache,
        "compute_ms_per_chunk": compute_ms,
        "consumer": consumer,
        "reader_busy_s": reader_busy[0] if mode == "prefetch" else None,
    }


def benchmark_decode(
    path: str,
    scheme: str,
    group_size: int = 32,
    seconds: float = 2.0,
) -> dict:
    """Throughput of the reference dequantizer on real stored bytes."""

    with MappedStore(path) as store:
        raws = []
        for name, rec in store.records.items():
            if rec.scheme == scheme:
                raws.append((store.read_raw(name), rec.count))
        if not raws:
            return {"scheme": scheme, "mb_per_s": 0.0, "values_per_s": 0.0}
        total_vals = sum(c for _, c in raws)
        total_bytes = sum(len(r) for r, _ in raws)
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < seconds:
            for raw, count in raws:
                quant.decode(raw, count, scheme, group_size)
            n += 1
        elapsed = time.perf_counter() - t0
        vals = total_vals * n
        return {
            "scheme": scheme,
            "values_per_s": vals / elapsed,
            "mb_per_s": (total_bytes * n / 1e6) / elapsed,
            "group_size": group_size,
        }
