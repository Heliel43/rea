"""Minimal GGUF v3 writer.

Purpose: let the streaming planner be exercised end to end without shipping a
40 GB file.  A fixture written here is a *real* GGUF file -- magic, version,
metadata, tensor directory, alignment, padding -- with shapes and names taken from
a published architecture but payloads of zeros.  REA's own
``inspect-model-artifact`` reads it, and ``pocket.reaadapter`` turns that reading
into a plan.

This is a test/fixture tool, not a model converter: it writes uncompressed tensor
data only, and it deliberately refuses to claim otherwise.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Sequence, Tuple

MAGIC = b"GGUF"

# ggml tensor type ids (subset)
GGML_TYPE = {"f32": 0, "f16": 1, "q4_0": 2, "q8_0": 8}

# gguf metadata value types
GGUF_UINT32, GGUF_FLOAT32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_UINT64 = 4, 6, 7, 8, 9, 10

ELEMENT_BYTES = {"f32": 4, "f16": 2}


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(vtype: int, value) -> bytes:
    if vtype == GGUF_STRING:
        return _string(value)
    if vtype == GGUF_UINT32:
        return struct.pack("<I", value)
    if vtype == GGUF_UINT64:
        return struct.pack("<Q", value)
    if vtype == GGUF_FLOAT32:
        return struct.pack("<f", value)
    if vtype == GGUF_BOOL:
        return struct.pack("<?", value)
    if vtype == GGUF_ARRAY:
        element_type, items = value
        if element_type == GGUF_STRING:
            return struct.pack("<I", element_type) + struct.pack("<Q", len(items)) + b"".join(
                _string(x) for x in items
            )
        if element_type == GGUF_UINT32:
            return struct.pack("<I", element_type) + struct.pack("<Q", len(items)) + b"".join(
                struct.pack("<I", x) for x in items
            )
        raise ValueError(f"unsupported array element type {element_type}")
    raise ValueError(f"unsupported metadata type {vtype}")


def write_gguf(
    path: str,
    metadata: Sequence[Tuple[str, int, object]],
    tensors: Sequence[Tuple[str, Sequence[int], str]],
    alignment: int = 32,
    payload: bool = True,
) -> int:
    """Write a GGUF v3 file. Returns its size in bytes.

    ``tensors`` is ``(name, dims, type_name)`` with dims in **GGUF order** (fastest
    varying dimension first, as llama.cpp writes them).  Zero payload is written
    unless ``payload`` is False, in which case the file ends after the directory.
    """

    header = bytearray(MAGIC)
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(metadata) + 1)  # + general.alignment
    header += _string("general.alignment") + struct.pack("<I", GGUF_UINT32) + struct.pack("<I", alignment)
    for key, vtype, value in metadata:
        header += _string(key) + struct.pack("<I", vtype) + _value(vtype, value)

    directory = bytearray()
    offsets: List[int] = []
    cursor = 0
    for name, dims, type_name in tensors:
        count = 1
        for d in dims:
            count *= d
        nbytes = count * ELEMENT_BYTES[type_name]
        offsets.append(cursor)
        cursor += ((nbytes + alignment - 1) // alignment) * alignment
        directory += _string(name)
        directory += struct.pack("<I", len(dims))
        for d in dims:
            directory += struct.pack("<Q", d)
        directory += struct.pack("<I", GGML_TYPE[type_name])
        directory += struct.pack("<Q", offsets[-1])

    body = bytes(header) + bytes(directory)
    pad = (-len(body)) % alignment
    with open(path, "wb") as fh:
        fh.write(body)
        fh.write(b"\x00" * pad)
        if payload:
            fh.write(b"\x00" * cursor)
    return len(body) + pad + (cursor if payload else 0)


# --------------------------------------------------------------------------- #
# Architecture fixtures
# --------------------------------------------------------------------------- #


def llama_fixture(path: str, layers: int = 4, d: int = 256, ffn: int = 688, vocab: int = 512, kv_heads: int = 2,
                  experts: int = 0, name: str = "fixture-llama") -> int:
    """A Llama-shaped (optionally MoE) GGUF with a full tensor directory.

    Shapes follow llama.cpp's naming so a plan built from the directory looks like a
    plan built from a real checkpoint: ``attn_q`` is [d, n_head*d_head] in GGUF
    order, expert tensors are 3-D ``ffn_*_exps``.
    """

    head_dim = d // 8
    n_head = 8
    meta: List[Tuple[str, int, object]] = [
        ("general.architecture", GGUF_STRING, "llama"),
        ("general.name", GGUF_STRING, name),
        ("llama.block_count", GGUF_UINT32, layers),
        ("llama.embedding_length", GGUF_UINT32, d),
        ("llama.feed_forward_length", GGUF_UINT32, ffn),
        ("llama.attention.head_count", GGUF_UINT32, n_head),
        ("llama.attention.head_count_kv", GGUF_UINT32, kv_heads),
        ("llama.context_length", GGUF_UINT32, 8192),
        ("tokenizer.ggml.model", GGUF_STRING, "gpt2"),
        ("tokenizer.ggml.tokens", GGUF_ARRAY, (GGUF_STRING, ["<unk>", "<s>", "</s>"] + [f"t{i}" for i in range(6)])),
    ]
    if experts:
        meta.append(("llama.expert_count", GGUF_UINT32, experts))
        meta.append(("llama.expert_used_count", GGUF_UINT32, 2))

    tensors: List[Tuple[str, Sequence[int], str]] = [("token_embd.weight", [d, vocab], "f32")]
    for layer in range(layers):
        pre = f"blk.{layer}."
        tensors += [
            (pre + "attn_norm.weight", [d], "f32"),
            (pre + "attn_q.weight", [d, d], "f32"),
            (pre + "attn_k.weight", [d, head_dim * kv_heads], "f32"),
            (pre + "attn_v.weight", [d, head_dim * kv_heads], "f32"),
            (pre + "attn_output.weight", [d, d], "f32"),
            (pre + "ffn_norm.weight", [d], "f32"),
        ]
        if experts:
            tensors += [
                (pre + "ffn_gate_inp.weight", [d, experts], "f32"),
                (pre + "ffn_gate_exps.weight", [d, ffn, experts], "f32"),
                (pre + "ffn_up_exps.weight", [d, ffn, experts], "f32"),
                (pre + "ffn_down_exps.weight", [ffn, d, experts], "f32"),
            ]
        else:
            tensors += [
                (pre + "ffn_gate.weight", [d, ffn], "f32"),
                (pre + "ffn_up.weight", [d, ffn], "f32"),
                (pre + "ffn_down.weight", [ffn, d], "f32"),
            ]
    tensors += [("output_norm.weight", [d], "f32"), ("output.weight", [d, vocab], "f32")]

    return write_gguf(path, meta, tensors, payload=False)
