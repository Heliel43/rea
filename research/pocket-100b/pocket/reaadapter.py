"""Turn a REA model inspection into a streaming plan.

REA already answers "what is in this file?" without loading it: ``inspect-model-artifact``
reads a GGUF v2/v3 or safetensors header and reports the tensor directory -- names,
types, shapes, offsets, byte lengths -- plus typed metadata.  That is exactly the
input a streaming scheduler needs and exactly what it usually cannot get cheaply.

This module closes the loop:

    rea inspect-model-artifact model.gguf --json --detail summary \\
        --evidence_output model-evidence.json --view '{"collection":"tensors"}'
    python3 -m pocket.reaadapter model-evidence.json --device laptop-16gb --bits 4

It is deliberately tolerant about the *shape* of the document: it searches for the
tensor rows and metadata pairs wherever they sit, so both a complete artifact
record and a paged summary view work.  What it will not do is invent numbers --
every byte in the plan traces back to a ``byte_length`` or ``parameter_count`` REA
reported, or to a documented assumption printed next to it.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import catalog, roofline
from .catalog import Device, ModelSpec

LAYER_RE = re.compile(r"(?:^|\.)blk\.(\d+)\.")
EXPERT_RE = re.compile(r"(exps|experts)\.weight$")


# --------------------------------------------------------------------------- #
# Tolerant extraction
# --------------------------------------------------------------------------- #


def _walk(node, depth: int = 0):
    if isinstance(node, dict):
        yield node
        if depth > 12:
            return
        for value in node.values():
            yield from _walk(value, depth + 1)
    elif isinstance(node, list):
        for value in node:
            if depth <= 12:
                yield from _walk(value, depth + 1)


def find_tensor_rows(doc: dict) -> List[dict]:
    """Find the richest list of tensor rows anywhere in the document."""

    best: List[dict] = []
    for node in _walk(doc):
        for key in ("tensors", "rows", "items", "tensor_declarations"):
            rows = node.get(key)
            if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "name" in rows[0]:
                if len(rows) > len(best):
                    best = [r for r in rows if isinstance(r, dict)]
    return best


def find_metadata(doc: dict) -> Dict[str, object]:
    """Flatten REA metadata rows (``{key, type, value}``) into a plain dict."""

    out: Dict[str, object] = {}
    for node in _walk(doc):
        rows = node.get("metadata")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and "key" in row:
                    out[str(row["key"])] = row.get("value")
    return out


def _as_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _count(row: dict) -> int:
    for key in ("parameter_count", "parameterCount", "params"):
        if key in row:
            n = _as_int(row[key])
            if n is not None:
                return n
    shape = row.get("shape") or []
    total = 1
    for dim in shape:
        n = _as_int(dim)
        if n is None:
            return 0
        total *= n
    return total if shape else 0


# --------------------------------------------------------------------------- #
# Model shape from a tensor directory
# --------------------------------------------------------------------------- #


class ArtifactShape:
    def __init__(self, doc: dict):
        self.doc = doc
        self.tensors = find_tensor_rows(doc)
        if not self.tensors:
            raise ValueError(
                "no tensor rows found; pass the JSON from `rea inspect-model-artifact --json`"
            )
        self.metadata = find_metadata(doc)
        self.layers = self._layer_count()
        self.params = sum(_count(t) for t in self.tensors)
        self.bytes = sum(int(t["byte_length"]) for t in self.tensors if t.get("byte_length") is not None)
        self.expert_params = sum(_count(t) for t in self.tensors if EXPERT_RE.search(str(t.get("name", ""))))
        self.expert_count, self.top_k = self._experts()
        self.kv_bytes_per_token = self._kv_bytes_per_token()

    # -- pieces ------------------------------------------------------------ #
    def _layer_count(self) -> int:
        idx = set()
        for t in self.tensors:
            m = LAYER_RE.search(str(t.get("name", "")))
            if m:
                idx.add(int(m.group(1)))
        if idx:
            return max(idx) + 1
        for key in ("llama.block_count", "general.block_count", "*.block_count"):
            for k, v in self.metadata.items():
                if k.endswith("block_count"):
                    n = _as_int(v)
                    if n:
                        return n
        return 0

    def _experts(self) -> Tuple[int, int]:
        # Metadata is authoritative; a tensor shape is a fallback and a dangerous
        # one: [d, ffn, n_expert] has three candidate dimensions and the *last* is
        # the expert axis (GGUF order is fastest-varying first).
        declared = 0
        top_k = 0
        for key, value in self.metadata.items():
            if key.endswith("expert_count"):
                declared = _as_int(value) or 0
            elif key.endswith("expert_used_count") or key.endswith("expert_used"):
                top_k = _as_int(value) or 0
        if declared:
            return declared, top_k
        count = 0
        for t in self.tensors:
            if EXPERT_RE.search(str(t.get("name", ""))):
                shape = [_as_int(d) for d in (t.get("shape") or [])]
                shape = [n for n in shape if n]
                if len(shape) >= 3:
                    count = max(count, shape[-1])
        return count, top_k

    def _kv_bytes_per_token(self) -> int:
        meta = self.metadata
        heads = 0
        head_kv = 0
        d_model = 0
        for key, value in meta.items():
            if key.endswith("attention.head_count"):
                heads = _as_int(value) or heads
            elif key.endswith("attention.head_count_kv"):
                head_kv = _as_int(value) or head_kv
            elif key.endswith("embedding_length"):
                d_model = _as_int(value) or d_model
        head_dim = (d_model // heads) if heads and d_model else 128
        kv_heads = head_kv or max(1, (heads or 8) // 4)
        return 2 * kv_heads * head_dim * max(1, self.layers) * 2  # K and V, fp16

    # -- derived ----------------------------------------------------------- #
    def effective_bits(self) -> float:
        if self.params and self.bytes:
            return 8.0 * self.bytes / self.params
        return 0.0

    def to_model_spec(self, name: Optional[str] = None) -> ModelSpec:
        total_b = self.params / 1e9
        if self.expert_count > 1:
            active_b = (self.params - self.expert_params) + self.expert_params * (self.top_k or 1) / self.expert_count
            active_b /= 1e9
            spec = ModelSpec(
                name or str(self.metadata.get("general.name") or "artifact"),
                total_b,
                active_b,
                max(1, self.layers),
                n_experts=self.expert_count,
                top_k=self.top_k or 1,
                kv_bytes_per_token=self.kv_bytes_per_token,
                source="REA inspect-model-artifact",
                notes=f"{self.expert_count} experts, top-{self.top_k}",
            )
        else:
            spec = ModelSpec(
                name or str(self.metadata.get("general.name") or "artifact"),
                total_b,
                total_b,
                max(1, self.layers),
                kv_bytes_per_token=self.kv_bytes_per_token,
                source="REA inspect-model-artifact",
            )
        return spec

    def globals_and_layers(self) -> Tuple[List[dict], List[List[dict]]]:
        """Split tensors into the resident set and per-layer groups."""

        globals_: List[dict] = []
        layers: Dict[int, List[dict]] = {}
        for t in self.tensors:
            name = str(t.get("name", ""))
            m = LAYER_RE.search(name)
            entry = {
                "name": name,
                "bytes": int(t["byte_length"]) if t.get("byte_length") is not None else 0,
                "params": _count(t),
                "type": t.get("type"),
            }
            if m:
                layers.setdefault(int(m.group(1)), []).append(entry)
            else:
                globals_.append(entry)
        return globals_, [layers.get(i, []) for i in range(max(layers) + 1 if layers else 0)]


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def plan_from_inspection(
    doc: dict,
    device: Device,
    bits: Optional[float] = None,
    ctx: int = 4096,
    kv_bits: float = 8.0,
    storage_bw_gbs: Optional[float] = None,
) -> Dict[str, object]:
    """Full plan: model shape, per-layer schedule, residency, throughput."""

    shape = ArtifactShape(doc)
    spec = shape.to_model_spec()
    effective_bits = bits if bits is not None else (shape.effective_bits() or 4.0)
    globals_, layers = shape.globals_and_layers()

    if storage_bw_gbs:
        device = Device(
            device.name, device.ram_gb, device.ram_bw_gbs, storage_bw_gbs,
            device.storage_rand_gbs, device.vram_gb, device.vram_bw_gbs, device.pcie_gbs,
            device.effective_flops_tflops, device.cpu_flops_tflops, device.notes,
        )

    p = roofline.plan(spec, device, bits=effective_bits, ctx=ctx, kv_bits=kv_bits)

    resident_names = {"token_embd.weight", "output.weight", "output_norm.weight"}
    resident = [t for t in globals_ if t["name"] in resident_names]
    resident_other = [t for t in globals_ if t["name"] not in resident_names]
    layer_bytes = [sum(t["bytes"] for t in layer) for layer in layers]
    largest = max((t for layer in layers for t in layer), key=lambda t: t["bytes"], default=None)

    warnings: List[str] = []
    if not layers:
        warnings.append("no `blk.N.` tensors found: this may be a non-transformer or a renamed layout")
    if resident:
        emb_bytes = sum(t["bytes"] for t in resident)
        warnings.append(
            f"{emb_bytes/1e9:.2f} GB of token embedding / head weights are read by token id, not by "
            f"layer order: keep them resident (or gather single rows) -- they cannot be prefetched "
            f"sequentially like a block"
        )
    if shape.expert_count > 1:
        warnings.append(
            f"{shape.expert_count} experts per layer, top-{shape.top_k}: only the selected experts need "
            f"to move, so the cache-hit curve (not the file size) decides throughput"
        )
    if largest and largest["bytes"] > 256 * 1024 * 1024:
        warnings.append(
            f"largest tensor is {largest['bytes']/1e6:.0f} MB ({largest['name']}); a single read that "
            f"large cannot be overlapped with a layer's compute on a small machine -- split it"
        )

    # prefetch depth: one buffer per block is enough to hide a block's transfer if
    # the reader can stay a block ahead; depth 2 is the practical default.
    per_layer_transfer_s = (
        (layer_bytes[0] / 1e9) / device.storage_bw_gbs if layer_bytes and device.storage_bw_gbs else 0.0
    )
    per_layer_compute_s = (
        (2.0 * spec.active_params_b * 1e9 / max(1, spec.layers)) / (device.effective_flops_tflops * 1e12)
        if device.effective_flops_tflops
        else 0.0
    )
    depth = 2
    if per_layer_compute_s > 0:
        depth = max(2, min(8, int(1 + per_layer_transfer_s / per_layer_compute_s + 0.999)))

    return {
        "artifact": {
            "name": spec.name,
            "params_b": shape.params / 1e9,
            "bytes_gb": shape.bytes / 1e9,
            "effective_bits_per_weight": effective_bits,
            "layers": shape.layers,
            "experts": shape.expert_count,
            "top_k": shape.top_k,
            "tensors": len(shape.tensors),
            "metadata_keys": sorted(shape.metadata)[:12],
        },
        "model_spec": {
            "params_b": spec.params_b,
            "active_params_b": spec.active_params_b,
            "expert_share": spec.expert_share,
        },
        "plan": p,
        "schedule": {
            "layer_bytes_mb": [round(b / 1e6, 3) for b in layer_bytes],
            "total_layer_mb": round(sum(layer_bytes) / 1e6, 3),
            "resident_set_mb": round(sum(t["bytes"] for t in resident + resident_other) / 1e6, 3),
            "resident_names": [t["name"] for t in resident + resident_other],
            "largest_tensor": None if not largest else {"name": largest["name"], "mb": round(largest["bytes"] / 1e6, 3)},
            "prefetch_depth": depth,
            "per_layer_transfer_ms": round(per_layer_transfer_s * 1000, 3),
            "per_layer_compute_ms": round(per_layer_compute_s * 1000, 3),
        },
        "warnings": warnings,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Plan streaming inference from a REA model inspection")
    ap.add_argument("inspection", help="JSON from `rea inspect-model-artifact --json`")
    ap.add_argument("--device", default="laptop-16gb")
    ap.add_argument("--bits", type=float, default=None, help="override the file's effective bits")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--storage-gbs", type=float, default=None, help="override storage bandwidth")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    with open(args.inspection, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    dev = catalog.device(args.device)
    out = plan_from_inspection(doc, dev, bits=args.bits, ctx=args.ctx, storage_bw_gbs=args.storage_gbs)

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    a, p, s = out["artifact"], out["plan"], out["schedule"]
    print(f"artifact   {a['name']}: {a['params_b']:.1f}B params, {a['bytes_gb']:.1f} GB, "
          f"{a['layers']} layers, {a['effective_bits_per_weight']:.2f} effective bits/weight")
    print(f"device     {p['device']} ({dev.notes})")
    print(f"strategy   {p['strategy']} -> {p['tok_s']:.2f} tokens/s, bottleneck {p['bottleneck']}")
    print(f"per token  {p['bytes_per_token']/1e9:.3f} GB moved, {p['resident_gb']:.1f} GB resident")
    print(f"schedule   {s['total_layer_mb']:.0f} MB across layers, largest tensor "
          f"{s['largest_tensor']['mb'] if s['largest_tensor'] else 0:.0f} MB, suggested prefetch depth {s['prefetch_depth']}")
    for note in p["notes"] + out["warnings"]:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
