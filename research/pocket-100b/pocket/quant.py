"""Block quantization codecs, written to be inspectable rather than fast.

Every scheme except ``fp16`` is *group-wise affine*: ``group_size`` consecutive
weights share one (scale, zero-point) pair, exactly like ggml's ``Q4_K``/``Q8_0``
blocks and every other deployed low-bit format.  Group size trades metadata
overhead against reconstruction error; both appear in RESULTS.md.

Schemes
-------
``fp16``     2 B/weight, no groups -- the "don't quantize" baseline.
``int8``     1 B/weight + 4 B per group (symmetric).
``int4``     0.5 B/weight + 4 B per group (symmetric) -- the consumer sweet spot.
``int2``     0.25 B/weight + 8 B per group (asymmetric, 4 linear levels).
``ternary``  BitNet-style: alpha = mean(|w|) per group, levels {-a, 0, +a} in
             2 bits with one spare code.  Weights only; activations stay float.

The point of this module in the study is arithmetic: *bytes per token* is the
denominator of every throughput number, so quantization is the cheapest lever on
how much of a model has to cross a bus at all.
"""

from __future__ import annotations

import math
import struct
from typing import List, Sequence, Tuple

SCHEMES = ("fp16", "int8", "int4", "int2", "ternary")

_BITS = {"fp16": 16, "int8": 8, "int4": 4, "int2": 2, "ternary": 2}
_PER_BYTE = {"fp16": 0, "int8": 1, "int4": 2, "int2": 4, "ternary": 4}
# bytes of per-group metadata
_META = {"fp16": 0, "int8": 4, "int4": 4, "int2": 8, "ternary": 4}


def nbytes_for(count: int, scheme: str, group_size: int = 32) -> int:
    if scheme not in _BITS:
        raise ValueError(f"unknown scheme {scheme!r}")
    if scheme == "fp16":
        return 2 * count
    groups = max(1, (count + group_size - 1) // group_size)
    packed = (max(count, 1) + _PER_BYTE[scheme] - 1) // _PER_BYTE[scheme]
    return groups * _META[scheme] + packed


def encoded_size(count: int, scheme: str, group_size: int = 32) -> Tuple[Tuple[int, ...], int]:
    """Compatibility shim used by the store writer."""

    return (count,), nbytes_for(count, scheme, group_size)


def bits_per_weight(count: int, scheme: str, group_size: int = 32) -> float:
    """Effective storage cost, group scales included.  The honest number."""

    if count <= 0:
        return float(_BITS[scheme])
    return 8.0 * nbytes_for(count, scheme, group_size) / count


# --------------------------------------------------------------------------- #
# Parametric quantizer
# --------------------------------------------------------------------------- #

#  nlevels, symmetric?, levels used by the codec
_CFG = {
    "int8": (256, True),
    "int4": (16, True),
    "int2": (4, False),
}


def _group_params(group: Sequence[float], scheme: str) -> Tuple[float, float]:
    """Return (scale, zero) such that value ~= (code - zero) * scale."""

    if scheme == "ternary":
        alpha = sum(abs(v) for v in group) / max(1, len(group))
        return (alpha if alpha > 0 else 1e-12), 0.0
    lo = min(group)
    hi = max(group)
    levels, symmetric = _CFG[scheme]
    if symmetric:
        # codes 1..levels-1 straddle a fixed zero point at levels/2, leaving code
        # 0 unused -- the same layout as ggml Q4_0 / Q8_0 blocks.
        amax = max(abs(lo), abs(hi))
        qmax = levels / 2.0 - 1  # 127 for int8, 7 for int4
        scale = amax / qmax if amax > 0 else 1e-12
        return scale, levels / 2.0
    if hi == lo:
        return 1e-12, 0.0
    scale = (hi - lo) / (levels - 1)
    # "zero" is the additive offset: code = round(v/scale + zero), v = (code - zero)*scale
    return scale, -lo / scale


def _code(v: float, scale: float, zero: float, levels: int) -> int:
    return min(levels - 1, max(0, int(round(v / scale + zero))))


def encode(values: Sequence[float], scheme: str, group_size: int = 32) -> bytes:
    if scheme == "fp16":
        out = bytearray(2 * len(values))
        if values:
            struct.pack_into("<%de" % len(values), out, 0, *values)
        return bytes(out)

    out = bytearray()
    n = len(values)
    per_byte = _PER_BYTE[scheme]
    step = 8 // per_byte
    for g0 in range(0, max(n, 1), group_size):
        group = values[g0 : g0 + group_size]
        if not group:
            group = [0.0]
        scale, zero = _group_params(group, scheme)
        if scheme == "ternary":
            alpha = scale
            codes = [1 if v > 0.5 * alpha else (2 if v < -0.5 * alpha else 0) for v in group]
            out.extend(struct.pack("<f", alpha))
        else:
            levels = _CFG[scheme][0]
            codes = [_code(v, scale, zero, levels) for v in group]
            out.extend(struct.pack("<f", scale))
            if _META[scheme] == 8:
                out.extend(struct.pack("<f", zero))
        if per_byte == 1:
            out.extend(bytes(codes))
        else:
            for i in range(0, len(codes), per_byte):
                chunk = codes[i : i + per_byte]
                while len(chunk) < per_byte:
                    chunk.append(0)
                byte = 0
                for j, code in enumerate(chunk):
                    byte |= (code & 0xFF) << (step * j)
                out.append(byte)
    return bytes(out)


def decode(buf: bytes, count: int, scheme: str, group_size: int = 32) -> List[float]:
    if scheme == "fp16":
        return list(struct.unpack_from("<%de" % count, buf, 0)) if count else []

    per_byte = _PER_BYTE[scheme]
    step = 8 // per_byte
    mask = (1 << step) - 1
    meta = _META[scheme]
    out: List[float] = []
    pos = 0
    remaining = count
    levels, symmetric = _CFG.get(scheme, (4, False))
    fixed_zero = levels / 2.0 if symmetric else 0.0
    while remaining > 0:
        (scale,) = struct.unpack_from("<f", buf, pos)
        pos += 4
        if meta == 8:
            (zero,) = struct.unpack_from("<f", buf, pos)
            pos += 4
        else:
            zero = fixed_zero
        take = min(group_size, remaining)
        if per_byte == 1:
            group = buf[pos : pos + take]
            if scheme == "int8":
                out.extend((b - zero) * scale for b in group)
            pos += take
        else:
            nbytes = (take + per_byte - 1) // per_byte
            group = buf[pos : pos + nbytes]
            produced = 0
            for byte in group:
                for j in range(per_byte):
                    if produced >= take:
                        break
                    code = (byte >> (step * j)) & mask
                    if scheme == "ternary":
                        out.append(1.0 * scale if code == 1 else (-1.0 * scale if code == 2 else 0.0))
                    else:
                        out.append((code - zero) * scale)
                    produced += 1
                if produced >= take:
                    break
            pos += nbytes
        remaining -= take
    return out


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #


def reconstruction_error(values: Sequence[float], decoded: Sequence[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"rmse": 0.0, "relative_rmse": 0.0, "max_abs": 0.0, "cosine": 1.0}
    sq = sum((a - b) ** 2 for a, b in zip(values, decoded))
    norm = math.sqrt(sum(a * a for a in values))
    dot = sum(a * b for a, b in zip(values, decoded))
    db = math.sqrt(sum(b * b for b in decoded))
    return {
        "rmse": math.sqrt(sq / n),
        "relative_rmse": math.sqrt(sq / n) / (norm / math.sqrt(n)) if norm else 0.0,
        "max_abs": max((abs(a - b) for a, b in zip(values, decoded)), default=0.0),
        "cosine": dot / (norm * db) if norm and db else 1.0,
    }
