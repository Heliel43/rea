"""Reference models and devices, with the assumptions written down.

Two rules for this file:

1. Every number is either *nominal spec* (a datasheet) or *published* (a model
   card / paper).  They are inputs to a cost model, not measurements.  The
   measurements happen in ``hwprobe.py`` and in ``experiments/``.
2. Where a model card publishes total and active parameters but not the split
   between attention and experts, the split is *derived* so that the published
   active-parameter count is reproduced exactly.  That keeps the cost model
   honest about the one number that decides everything: how many bytes a token
   actually has to touch.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class ModelSpec:
    """A model's shape, reduced to the quantities that matter for streaming."""

    def __init__(
        self,
        name: str,
        params_b: float,
        active_params_b: float,
        layers: int,
        n_experts: int = 1,
        top_k: int = 1,
        kv_bytes_per_token: int = 160_000,
        source: str = "",
        notes: str = "",
    ):
        self.name = name
        self.params_b = params_b
        self.layers = layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.kv_bytes_per_token = kv_bytes_per_token
        self.source = source
        self.notes = notes
        if n_experts > 1:
            # experts are the share of the model that routing can skip
            denom = 1.0 - top_k / n_experts
            share = (params_b - active_params_b) / (params_b * denom) if denom > 0 else 0.0
            # 0.995, not 0.98: gpt-oss-120b really is ~98.7% expert weights, and a
            # clamp below that would silently break the published active count.
            self.expert_share = min(max(share, 0.0), 0.995)
        else:
            self.expert_share = 0.0
        self.active_params_b = (
            params_b * (1.0 - self.expert_share) + params_b * self.expert_share * top_k / n_experts
            if n_experts > 1
            else params_b
        )

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 1

    @property
    def expert_params_b(self) -> float:
        return self.params_b * self.expert_share

    def weight_bytes(self, bits: float) -> float:
        return self.params_b * 1e9 * bits / 8.0

    def active_weight_bytes(self, bits: float) -> float:
        return self.active_params_b * 1e9 * bits / 8.0

    def kv_bytes(self, context: int, batch: int = 1, kv_bits: float = 16.0) -> float:
        return self.kv_bytes_per_token * context * batch * (kv_bits / 16.0)

    def dense_equivalent_params(self) -> float:
        "Parameters a dense model would need to match this one's per-token work."

        return self.active_params_b

    def describe(self) -> str:
        if self.is_moe:
            return (
                f"{self.params_b:.0f}B MoE ({self.n_experts} experts, top-{self.top_k}, "
                f"{self.active_params_b:.1f}B active/token)"
            )
        return f"{self.params_b:.0f}B dense"


class Device:
    def __init__(
        self,
        name: str,
        ram_gb: float,
        ram_bw_gbs: float,
        storage_bw_gbs: float,
        storage_rand_gbs: float,
        vram_gb: float = 0.0,
        vram_bw_gbs: float = 0.0,
        pcie_gbs: float = 28.0,
        effective_flops_tflops: float = 0.05,
        cpu_flops_tflops: float = 0.1,
        notes: str = "",
    ):
        self.name = name
        self.ram_gb = ram_gb
        self.ram_bw_gbs = ram_bw_gbs
        self.storage_bw_gbs = storage_bw_gbs
        self.storage_rand_gbs = storage_rand_gbs
        self.vram_gb = vram_gb
        self.vram_bw_gbs = vram_bw_gbs
        self.pcie_gbs = pcie_gbs
        self.effective_flops_tflops = effective_flops_tflops
        self.cpu_flops_tflops = cpu_flops_tflops
        self.notes = notes


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

# kv_bytes_per_token = 2 (K and V) * kv_heads * head_dim * layers * 2 bytes (fp16)
MODELS: Dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        "Llama-3.1-8B", 8.03, 8.03, 32, kv_bytes_per_token=163_840,
        source="https://arxiv.org/abs/2407.21783",
        notes="dense, GQA 32Q/8KV, head_dim 128",
    ),
    "llama-3.1-70b": ModelSpec(
        "Llama-3.1-70B", 70.6, 70.6, 80, kv_bytes_per_token=327_680,
        source="https://arxiv.org/abs/2407.21783",
        notes="dense, GQA 64Q/8KV, head_dim 128",
    ),
    "mixtral-8x7b": ModelSpec(
        "Mixtral-8x7B", 46.7, 12.9, 32, n_experts=8, top_k=2, kv_bytes_per_token=131_072,
        source="https://arxiv.org/abs/2401.04088",
        notes="MoE, 8 experts top-2",
    ),
    "mixtral-8x22b": ModelSpec(
        "Mixtral-8x22B", 141.0, 39.0, 56, n_experts=8, top_k=2, kv_bytes_per_token=262_144,
        source="https://mistral.ai/news/mixtral-8x22b",
        notes="MoE, 8 experts top-2",
    ),
    "qwen3-235b-a22b": ModelSpec(
        "Qwen3-235B-A22B", 235.0, 22.0, 94, n_experts=128, top_k=8, kv_bytes_per_token=96_256,
        source="https://huggingface.co/Qwen/Qwen3-235B-A22B",
        notes="MoE, 128 experts top-8, GQA 64Q/4KV, head_dim 64",
    ),
    "gpt-oss-120b": ModelSpec(
        "gpt-oss-120b", 117.0, 5.1, 36, n_experts=128, top_k=4, kv_bytes_per_token=147_456,
        source="https://openai.com/index/introducing-gpt-oss/",
        notes="MoE, 128 experts top-4, native MXFP4 experts",
    ),
    "deepseek-v3": ModelSpec(
        "DeepSeek-V3", 671.0, 37.0, 61, n_experts=256, top_k=8, kv_bytes_per_token=174_080,
        source="https://arxiv.org/abs/2412.19437",
        notes="MoE, 256 routed experts + 1 shared, top-8",
    ),
    "dense-100b": ModelSpec(
        "Dense-100B (hypothetical)", 100.0, 100.0, 96, kv_bytes_per_token=393_216,
        source="synthetic",
        notes="the request in the brief: a 100B dense model on a consumer device",
    ),
}


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #

# Bandwidths are nominal datasheet values rounded down; storage figures are
# "sequential read / 64 KiB random read" as typically measured on the device class.
DEVICES: Dict[str, Device] = {
    "phone-8gb": Device(
        "Phone, 8 GB", ram_gb=7.0, ram_bw_gbs=68.0, storage_bw_gbs=3.5, storage_rand_gbs=0.25,
        effective_flops_tflops=0.03, cpu_flops_tflops=0.02,
        notes="LPDDR5X ~68 GB/s, UFS 4.0 ~3.5 GB/s sequential",
    ),
    "laptop-16gb": Device(
        "Laptop, 16 GB", ram_gb=14.0, ram_bw_gbs=89.6, storage_bw_gbs=6.5, storage_rand_gbs=1.2,
        effective_flops_tflops=0.15, cpu_flops_tflops=0.10,
        notes="DDR5-5600 dual channel = 89.6 GB/s, NVMe Gen4 ~6.5 GB/s",
    ),
    "desktop-32gb": Device(
        "Desktop, 32 GB", ram_gb=30.0, ram_bw_gbs=96.0, storage_bw_gbs=7.0, storage_rand_gbs=1.5,
        effective_flops_tflops=0.25, cpu_flops_tflops=0.20,
        notes="DDR5-6000 dual channel, NVMe Gen4",
    ),
    "desktop-32gb-4090": Device(
        "Desktop 32 GB + RTX 4090", ram_gb=30.0, ram_bw_gbs=96.0, storage_bw_gbs=7.0,
        storage_rand_gbs=1.5, vram_gb=23.0, vram_bw_gbs=1008.0, pcie_gbs=26.0,
        effective_flops_tflops=160.0, cpu_flops_tflops=0.25,
        notes="4090: 24 GB GDDR6X at 1008 GB/s; PCIe 4.0 x16 ~26 GB/s effective",
    ),
    "desktop-128gb-5090": Device(
        "Desktop 128 GB + RTX 5090", ram_gb=120.0, ram_bw_gbs=96.0, storage_bw_gbs=7.0,
        storage_rand_gbs=1.5, vram_gb=31.0, vram_bw_gbs=1792.0, pcie_gbs=55.0,
        effective_flops_tflops=400.0, cpu_flops_tflops=0.25,
        notes="5090: 32 GB GDDR7 at 1792 GB/s; PCIe 5.0 x16 ~55 GB/s effective",
    ),
    "mac-studio-96gb": Device(
        "Mac Studio 96 GB", ram_gb=88.0, ram_bw_gbs=546.0, storage_bw_gbs=6.0, storage_rand_gbs=1.4,
        effective_flops_tflops=55.0, cpu_flops_tflops=0.4,
        notes="M3 Ultra class unified memory ~546 GB/s",
    ),
}


def model(name: str) -> ModelSpec:
    key = name.lower()
    if key not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(sorted(MODELS))}")
    return MODELS[key]


def device(name: str) -> Device:
    key = name.lower()
    if key not in DEVICES:
        raise KeyError(f"unknown device {name!r}; known: {', '.join(sorted(DEVICES))}")
    return DEVICES[key]


def device_from_probe(profile: dict, name: str = "this machine") -> Device:
    """Build a Device from ``hwprobe.profile()`` output (all measured, no datasheet)."""

    cpu = profile.get("cpu", {})
    ram_mb = cpu.get("ram_mb") or cpu.get("ram_available_mb") or 4096
    derived = profile.get("derived", {})
    seq = derived.get("storage_seq_mb_s", 500.0) / 1000.0
    rnd = derived.get("storage_random_mb_s", 100.0) / 1000.0
    copy = derived.get("copy_gbs", 10.0)
    matvec = profile.get("python_matvec", {}).get("m_mac_per_s", 20.0)
    # 2 MAC = 2 flops; the kernel is interpreter-bound, which is the honest
    # number for a stdlib-only runtime.
    flops = matvec * 1e6 * 2 / 1e12
    return Device(
        name,
        ram_gb=ram_mb / 1024.0,
        ram_bw_gbs=max(copy * 2, 1.0),  # copy moves 2 B per B of data
        storage_bw_gbs=max(seq, 0.01),
        storage_rand_gbs=max(rnd, 0.01),
        effective_flops_tflops=max(flops, 1e-4),
        cpu_flops_tflops=max(flops, 1e-4),
        notes="measured by pocket.hwprobe on this machine",
    )
