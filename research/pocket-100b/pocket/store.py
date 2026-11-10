"""Checkpoint and weight-store serialization.

Two formats, one purpose: get weights out of Python objects and onto a file so
that the experiments measure bytes, not interpreter bookkeeping.

``save_checkpoint``  float32, faithful, used to keep a trained subject between
                     runs (the reference the streamed engine must reproduce).

``WeightStore``      a quantized, page-friendly container: one contiguous record
                     per tensor, groups of ``group_size`` values sharing a scale,
                     with an aligned JSON directory up front.  Reading a tensor
                     costs one sequential read -- which is the whole premise of
                     streaming inference from flash.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

from . import quant
from .nn import ModelConfig, Params

CHECKPOINT_MAGIC = b"PKCK1\x00\x00\x00"
STORE_MAGIC = b"PKSW1\x00\x00\x00"


# --------------------------------------------------------------------------- #
# Checkpoints (exact)
# --------------------------------------------------------------------------- #


def save_checkpoint(path: str, cfg: ModelConfig, p: Params, meta: Optional[dict] = None) -> None:
    import array

    names = list(p.data)
    directory = {
        "config": cfg.as_dict(),
        "meta": meta or {},
        "tensors": [
            {"name": n, "shape": list(p.shape[n]), "count": len(p.data[n])} for n in names
        ],
    }
    blob = json.dumps(directory, sort_keys=True).encode()
    with open(path, "wb") as fh:
        fh.write(CHECKPOINT_MAGIC)
        fh.write(struct.pack("<I", len(blob)))
        fh.write(blob)
        pad = (-fh.tell()) % 64
        fh.write(b"\x00" * pad)
        for n in names:
            a = array.array("f", p.data[n])
            fh.write(a.tobytes())


def load_checkpoint(path: str) -> Tuple[ModelConfig, Params, dict]:
    import array

    with open(path, "rb") as fh:
        magic = fh.read(len(CHECKPOINT_MAGIC))
        if magic != CHECKPOINT_MAGIC:
            raise ValueError("not a checkpoint")
        (blen,) = struct.unpack("<I", fh.read(4))
        directory = json.loads(fh.read(blen).decode())
        pad = (-(len(CHECKPOINT_MAGIC) + 4 + blen)) % 64
        fh.read(pad)
        cfg = ModelConfig.from_dict(directory["config"])
        p = Params()
        for row in directory["tensors"]:
            n = row["count"]
            a = array.array("f")
            a.frombytes(fh.read(4 * n))
            p.add(row["name"], row["shape"], [float(x) for x in a])
    return cfg, p, directory.get("meta", {})


# --------------------------------------------------------------------------- #
# Quantized weight store
# --------------------------------------------------------------------------- #


def tensor_order(cfg: ModelConfig) -> List[str]:
    """Canonical order: the order the runtime will actually ask for tensors.

    Embeddings first, then block by block, then the head.  Sequential order on
    disk means a streaming reader gets the read pattern it wants for free.
    """

    names = ["token_embd", "pos_embd"]
    for layer in range(cfg.n_layers):
        pre = f"blk.{layer}."
        names.extend(
            [
                pre + "attn_norm",
                pre + "attn_q",
                pre + "attn_k",
                pre + "attn_v",
                pre + "attn_out",
                pre + "ffn_norm",
            ]
        )
        if cfg.is_moe:
            names.append(pre + "router")
        names.extend([pre + "ffn_gate", pre + "ffn_up", pre + "ffn_down"])
    names.append("out_norm")
    return names


class WeightRecord:
    __slots__ = ("name", "shape", "scheme", "group_size", "offset", "nbytes", "count")

    def __init__(self, name, shape, scheme, group_size, offset, nbytes, count):
        self.name = name
        self.shape = shape
        self.scheme = scheme
        self.group_size = group_size
        self.offset = offset
        self.nbytes = nbytes
        self.count = count

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "scheme": self.scheme,
            "group_size": self.group_size,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "count": self.count,
        }


def write_store(
    path: str,
    cfg: ModelConfig,
    p: Params,
    scheme: str = "int4",
    group_size: int = 32,
    meta: Optional[dict] = None,
) -> Dict[str, WeightRecord]:
    """Write every tensor quantized, in canonical order, 64-byte aligned.

    Offsets in the header are relative to the start of the data section, which is
    the first 64-byte boundary after ``magic | header_length | header``.  That
    keeps the directory independent of its own serialized length, so writing is a
    single pass with no offset fix-ups.
    """

    order = tensor_order(cfg)
    missing = set(p.data) - set(order)
    if missing:  # pragma: no cover - guards silent format drift
        raise ValueError(f"tensors outside canonical order: {sorted(missing)}")

    directory: List[dict] = []
    payload = bytearray()
    for name in order:
        vals = p.data[name]
        nbytes = quant.nbytes_for(len(vals), scheme, group_size)
        directory.append(
            {
                "name": name,
                "shape": list(p.shape[name]),
                "scheme": scheme,
                "group_size": group_size,
                "offset": len(payload),
                "nbytes": nbytes,
                "count": len(vals),
            }
        )
        blob = quant.encode(vals, scheme, group_size)
        if len(blob) != nbytes:  # pragma: no cover - codec/store contract
            raise RuntimeError(f"{name}: codec wrote {len(blob)} bytes, expected {nbytes}")
        payload.extend(blob)
        payload.extend(b"\x00" * ((-len(payload)) % 64))

    header = {
        "format": "pocket-weight-store",
        "version": 1,
        "config": cfg.as_dict(),
        "scheme": scheme,
        "group_size": group_size,
        "meta": meta or {},
        "tensors": directory,
    }
    raw = json.dumps(header, sort_keys=True).encode()
    prefix = STORE_MAGIC + struct.pack("<I", len(raw))
    with open(path, "wb") as fh:
        fh.write(prefix)
        fh.write(raw)
        fh.write(b"\x00" * ((-(len(prefix) + len(raw))) % 64))
        fh.write(bytes(payload))

    return {
        rec["name"]: WeightRecord(
            rec["name"], tuple(rec["shape"]), rec["scheme"], rec["group_size"],
            rec["offset"], rec["nbytes"], rec["count"],
        )
        for rec in directory
    }


def data_start(header_len: int) -> int:
    "Absolute offset of the first payload byte for a header of ``header_len``."

    return (len(STORE_MAGIC) + 4 + header_len + 63) // 64 * 64


class MappedStore:
    """Read-only view of a ``PKSW1`` file with explicit, bounded reads."""

    def __init__(self, path: str, use_mmap: bool = False):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.file_size = os.fstat(self.fd).st_size
        with open(path, "rb") as fh:
            magic = fh.read(len(STORE_MAGIC))
            if magic != STORE_MAGIC:
                raise ValueError("not a pocket weight store")
            (blen,) = struct.unpack("<I", fh.read(4))
            header = json.loads(fh.read(blen).decode())
        self.header = header
        self.cfg = ModelConfig.from_dict(header["config"])
        self.scheme = header["scheme"]
        self.group_size = int(header["group_size"])
        self.meta = header.get("meta", {})
        base = data_start(blen)
        self.data_start = base
        self.records: Dict[str, WeightRecord] = {}
        for rec in header["tensors"]:
            self.records[rec["name"]] = WeightRecord(
                rec["name"], tuple(rec["shape"]), rec["scheme"], int(rec["group_size"]),
                base + int(rec["offset"]), int(rec["nbytes"]), int(rec["count"]),
            )
        self.bytes_read = 0
        self._mm: Optional[mmap.mmap] = None
        if use_mmap:
            self._mm = mmap.mmap(self.fd, 0, prot=mmap.PROT_READ)

    # -- raw access -------------------------------------------------------- #
    def read_raw(self, name: str) -> bytes:
        rec = self.records[name]
        if self._mm is not None:
            self.bytes_read += rec.nbytes
            return self._mm[rec.offset : rec.offset + rec.nbytes]
        self.bytes_read += rec.nbytes
        return os.pread(self.fd, rec.nbytes, rec.offset)

    def decode(self, name: str) -> List[float]:
        rec = self.records[name]
        return quant.decode(self.read_raw(name), rec.count, rec.scheme, rec.group_size)

    def release(self) -> None:
        """Drop mapped pages back to the OS so RSS does not grow without bound."""

        if self._mm is not None:
            start = 0
            size = self.file_size
            try:
                self._mm.madvise(mmap.MADV_DONTNEED, start, size)
            except (AttributeError, OSError):  # pragma: no cover - platform dependent
                pass

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "MappedStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- convenience ------------------------------------------------------- #
    def layer_bytes(self, layer: int) -> int:
        pre = f"blk.{layer}."
        return sum(r.nbytes for k, r in self.records.items() if k.startswith(pre))

    def total_bytes(self) -> int:
        return sum(r.nbytes for r in self.records.values())
