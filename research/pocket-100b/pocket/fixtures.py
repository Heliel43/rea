"""A model inspection document in REA's schema, for testing this lab without a 40 GB file.

The field names follow REA's own contract (`src/domain/model/modelArtifact.ts`:
``artifact``, ``format``, ``tensors`` with ``name``/``type``/``shape``/``file_offset``/
``byte_length``/``parameter_count``, plus ``metadata`` rows of ``{key, type, value}``).
Shapes are those of a real published architecture family; the payload does not
exist, which is the point -- a *declaration* is all the scheduler needs.

If you have the REA CLI, prefer the real thing:

    rea inspect-model-artifact model.gguf --json --detail summary \\
        --evidence_output model-evidence.json

and feed that file to ``python3 -m pocket.reaadapter``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

# Llama-3.1-70B-shaped and gpt-oss-120b-shaped declarations, scaled down in
# *vocabulary* only (vocabulary does not change the per-token byte budget much,
# but it dominates the file listing for no analytical gain).
PRESETS: Dict[str, dict] = {
    "dense-70b": dict(layers=80, d=8192, ffn=28672, heads=64, kv_heads=8, experts=0, top_k=0,
                      vocab=128256, bits=4.0, name="fixture-llama-3.1-70b-shaped"),
    "moe-120b": dict(layers=36, d=2880, ffn=2880, heads=64, kv_heads=8, experts=128, top_k=4,
                     vocab=201088, bits=4.25, name="fixture-gpt-oss-120b-shaped"),
    "dense-100b": dict(layers=96, d=8192, ffn=32768, heads=64, kv_heads=8, experts=0, top_k=0,
                       vocab=128256, bits=4.0, name="fixture-dense-100b-shaped"),
}


def _row(index: int, name: str, shape: Sequence[int], bits: float, file_offset: int) -> dict:
    params = 1
    for d in shape:
        params *= int(d)
    nbytes = int(round(params * bits / 8.0))
    return {
        "index": index,
        "name": name,
        "type": f"Q4_K" if abs(bits - 4.5) < 0.01 else ("Q4_0" if abs(bits - 4.0) < 0.01 else "Q8_0"),
        "type_id": 12,
        "shape": [str(d) for d in shape],
        "data_offset": file_offset,
        "file_offset": file_offset,
        "byte_length": nbytes,
        "parameter_count": str(params),
        "byte_length_state": "derived",
        "storage_validation": "consistent",
        "source_offset": index * 64,
        "source_length": 64,
        "json_pointer": None,
    }


def inspection_document(
    preset: str = "dense-70b",
    tensors: Optional[Sequence[Tuple[str, Sequence[int]]]] = None,
    bits: Optional[float] = None,
    path: str = "/models/fixture.gguf",
) -> dict:
    """Build a schema-shaped inspection document."""

    cfg = dict(PRESETS[preset])
    bits = bits or cfg["bits"]
    layers = cfg["layers"]
    d = cfg["d"]
    ffn = cfg["ffn"]
    heads = cfg["heads"]
    kv_heads = cfg["kv_heads"]
    experts = cfg["experts"]
    top_k = cfg["top_k"]
    vocab = cfg["vocab"]
    head_dim = d // heads

    rows: List[Tuple[str, Sequence[int]]] = [("token_embd.weight", [d, vocab])]
    for layer in range(layers):
        pre = f"blk.{layer}."
        rows.append((pre + "attn_norm.weight", [d]))
        rows.append((pre + "attn_q.weight", [d, d]))
        rows.append((pre + "attn_k.weight", [head_dim * kv_heads, d]))
        rows.append((pre + "attn_v.weight", [head_dim * kv_heads, d]))
        rows.append((pre + "attn_output.weight", [d, d]))
        rows.append((pre + "ffn_norm.weight", [d]))
        if experts:
            rows.append((pre + "ffn_gate_inp.weight", [d, experts]))
            rows.append((pre + "ffn_gate_exps.weight", [d, ffn, experts]))
            rows.append((pre + "ffn_up_exps.weight", [d, ffn, experts]))
            rows.append((pre + "ffn_down_exps.weight", [ffn, d, experts]))
        else:
            rows.append((pre + "ffn_gate.weight", [d, ffn]))
            rows.append((pre + "ffn_up.weight", [d, ffn]))
            rows.append((pre + "ffn_down.weight", [ffn, d]))
    rows.append(("output_norm.weight", [d]))
    rows.append(("output.weight", [d, vocab]))
    if tensors:
        rows = list(tensors)

    offset = 0
    tensor_rows = []
    for i, (name, shape) in enumerate(rows):
        r = _row(i, name, shape, bits, offset)
        offset += r["byte_length"] + ((-r["byte_length"]) % 32)
        tensor_rows.append(r)

    metadata = [
        {"index": 0, "key": "general.architecture", "type": "string", "value": "llama",
         "element_type": None, "element_count": None, "preview_complete": True,
         "source_offset": 0, "source_length": 0, "json_pointer": None},
        {"index": 1, "key": "general.name", "type": "string", "value": cfg["name"],
         "element_type": None, "element_count": None, "preview_complete": True,
         "source_offset": 0, "source_length": 0, "json_pointer": None},
        {"index": 2, "key": "llama.block_count", "type": "uint32", "value": layers,
         "element_type": None, "element_count": None, "preview_complete": True,
         "source_offset": 0, "source_length": 0, "json_pointer": None},
        {"index": 3, "key": "llama.embedding_length", "type": "uint32", "value": d,
         "element_type": None, "element_count": None, "preview_complete": True,
         "source_offset": 0, "source_length": 0, "json_pointer": None},
        {"index": 4, "key": "llama.attention.head_count", "type": "uint32", "value": heads,
         "element_type": None, "element_count": None, "preview_complete": True,
         "source_offset": 0, "source_length": 0, "json_pointer": None},
        {"index": 5, "key": "llama.attention.head_count_kv", "type": "uint32", "value": kv_heads,
         "element_type": None, "element_count": None, "preview_complete": True,
         "source_offset": 0, "source_length": 0, "json_pointer": None},
    ]
    if experts:
        metadata += [
            {"index": 6, "key": "llama.expert_count", "type": "uint32", "value": experts,
             "element_type": None, "element_count": None, "preview_complete": True,
             "source_offset": 0, "source_length": 0, "json_pointer": None},
            {"index": 7, "key": "llama.expert_used_count", "type": "uint32", "value": top_k,
             "element_type": None, "element_count": None, "preview_complete": True,
             "source_offset": 0, "source_length": 0, "json_pointer": None},
        ]

    total_params = sum(int(r["parameter_count"]) for r in tensor_rows)
    total_bytes = sum(int(r["byte_length"]) for r in tensor_rows)
    return {
        "artifact": {
            "path": path,
            "file_size": total_bytes,
            "header_sha256": hashlib.sha256(json.dumps(tensor_rows, sort_keys=True).encode()).hexdigest(),
            "header_length": offset,
            "sha256": None,
            "bytes_read": offset,
            "read_scope": "header",
        },
        "format": "gguf",
        "version": 3,
        "byte_order": "little",
        "alignment": 32,
        "tensor_data_offset": 0,
        "metadata": metadata,
        "tensors": tensor_rows,
        "totals": {
            "confidence": "derived",
            "parameter_count": str(total_params),
            "tensor_bytes": total_bytes,
            "by_type": [
                {"type": "Q4_K", "tensors": len(tensor_rows), "parameter_count": str(total_params),
                 "bytes": total_bytes}
            ],
        },
        "coverage": {
            "header": "complete",
            "tensor_data": "not-interpreted",
            "array_values": "complete",
            "tensor_ranges": "complete",
            "storage_sizes": "complete",
        },
        "limitations": [
            "Fixture document for the pocket-100b lab: shapes are from a published architecture, "
            "payload bytes are not present.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Write a fixture model inspection document")
    ap.add_argument("--preset", default="moe-120b", choices=sorted(PRESETS))
    ap.add_argument("--bits", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    doc = inspection_document(args.preset, bits=args.bits)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    print(f"wrote {args.out}: {len(doc['tensors'])} tensors, "
          f"{int(doc['totals']['parameter_count'])/1e9:.1f}B params declared")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
