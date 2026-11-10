# Pocket 100B: running models that do not fit, on machines that cannot hold them

*A measurement-first study of large-model inference on ordinary hardware — and an
honest account of where the wall actually is.*

---

## The short version

You can run a 100B-class model on a laptop. You cannot run a **dense** 100B model
on a laptop at a speed you would call usable, and no scheduling trick changes that.
The distinction is not marketing: it is one division.

For single-stream decoding, a model's arithmetic is irrelevant. What matters is how
many bytes have to reach the arithmetic per token:

```
tokens/second  =  bandwidth  /  bytes touched per token          (batch 1, decode)
```

Every technique in the literature — quantization, offloading, expert caching,
windowing, speculation — is a way of shrinking the denominator or amortising it over
more tokens. Nothing shrinks it for free.

Put numbers on it and the picture resolves immediately:

| model | weights @4-bit | bytes/token, dense | bytes/token, routed | laptop SSD 6.5 GB/s | laptop RAM 90 GB/s |
|---|---|---|---|---|---|
| Llama-3.1-70B (dense) | 35 GB | 35 GB | 35 GB | **0.18 tok/s** | 2.6 tok/s (doesn't fit) |
| Dense 100B (hypothetical) | 50 GB | 50 GB | 50 GB | **0.13 tok/s** | 1.8 tok/s (doesn't fit) |
| Qwen3-235B-A22B (MoE) | 118 GB | — | 11 GB | 0.73 tok/s | doesn't fit |
| gpt-oss-120b (MoE) | 59 GB | — | 2.6 GB | 3.8 tok/s | doesn't fit |

Those numbers come from `python3 -m pocket.roofline --bits 4` — upper bounds assuming
peak bandwidth, so 50-80% of them in practice — and they are the honest starting
point: **a dense model bigger than RAM is a 0.1-1 token/second device, and a
"tokens/second" figure without a bytes/token figure next to it is not a measurement
of anything.**

(On a 96 GB unified-memory machine where the last two rows *do* fit, the same model
reaches ~5 tok/s for Qwen3-235B and ~39 tok/s for gpt-oss-120b; the laptop columns
are there to show that the file, not the arithmetic, is what refuses to fit.)

But look at the last two rows. The models people actually run on consumer hardware
this year — gpt-oss-120b, Qwen3-235B, DeepSeek-V3, Mixtral — are mixture-of-experts
models, and a token touches a *routed subset*, not the whole file. That is the
opening, and it is why "100B on a consumer device" went from a joke to a product in
about eighteen months.

This lab measures the three things that decide whether a given model/device pair can
do it, on hardware you can buy, using code small enough to read:

1. **How many bits does a weight need before quality notices?**
   (`pocket/quant.py`, `exp_quant`)
2. **How local is the routing, really?** — i.e. does an expert cache have anything to
   cache? (`pocket/analysis.py`, `exp_moe`)
3. **What does a bounded-memory streaming runtime actually cost per token?**
   (`pocket/streaming.py`, `exp_stream`, `exp_transport`)

---

## What the measurements say (details in [RESULTS.md](RESULTS.md))

Six results from the full run on a 2-vCPU / 4 GB container, all reproducible with
`python3 run_experiments.py`:

1. **int8 is free, int4 is cheap, 2-bit is not.** Against a trained subject, int8
   cost +0.001 nats of held-out loss and kept 99% of argmax decisions; int4 cost
   +0.024 nats at 5.0 bits/weight (77% agreement); 2-bit cost +0.46 nats and kept
   38%. Post-hoc 2-bit is a different model, not a compressed one.
2. **Metadata, not bits, sets the low-bit floor.** At 32-wide groups, 2-bit and 4-bit
   both cost ~4-5 bits/weight. Widening groups to 128 takes int4 to 4.26 bits/weight
   with *no* measurable quality loss, and int2 to 2.51 bits/weight — where the
   quality bill finally appears (+0.63 nats).
3. **A 6-expert toy does not have a hot-expert distribution** (Gini 0.085; a cache
   holding 39% of the pool hits 43%). Production 128-expert MoEs report 70-90% hit
   rates at 10% of the pool. The lab prints the measured curve, the synthetic
   reference and the planner's assumption side by side precisely so you can tell
   which regime you are in — with your trace, not mine.
4. **Streaming is exact and bounded.** A quantized store executed block-by-block
   produces *bit-identical* logits to the resident path over the same decoded weights
   (asserted in `tests/test_streaming.py`), at 13 MB peak RSS in a fresh process,
   versus 798 MB for the same bytes through `mmap`.
5. **Prefetch did not pay off here, and the report says so.** 74 → 64 tokens/s at
   depth 1, with a 2.5× run-to-run spread on a single configuration. What is reported
   instead is the *bound*: overlap can recover at most the transfer time.
6. **The wall is the wall.** Dense 100B at 4-bit is 50 GB/token of traffic: 0.07
   tok/s on a phone, 0.13 on a laptop, 2.25 on a 4090 with CPU offload. gpt-oss-120b
   at 2.55 GB/token: 1.7 / 3.8 / 52 tok/s on the same three devices. Same code, same
   table, 20× difference — and it is entirely the denominator.

---

## What is in this directory

```
pocket/nn.py          pure-stdlib transformer: dense or MoE, forward + hand-written
                      backward, verified against finite differences
pocket/quant.py       block quantization codecs (fp16/int8/int4/int2/ternary)
pocket/store.py       checkpoint format + page-friendly quantized weight store
pocket/streaming.py   bounded-residency execution: prefetch ring, pread vs mmap,
                      per-layer stall accounting
pocket/analysis.py    probes: router statistics, LRU cache curve, channel
                      concentration, end-to-end quantization quality
pocket/roofline.py    the cost model and planner (this is the part to argue with)
pocket/catalog.py     reference models and devices, every assumption written down
pocket/hwprobe.py     measures *your* machine: RAM, memcpy, sequential/random read
pocket/reaadapter.py  REA GGUF/safetensors inspection -> an execution plan
pocket/gguf.py        minimal GGUF v3 writer, for fixtures without a 40 GB download
pocket/fixtures.py    schema-shaped inspection documents (Llama-70B / gpt-oss-120B)
run_experiments.py    runs everything, renders figures/, writes RESULTS.md
tests/                unit + gradient + round-trip + scheduler tests
```

No third-party dependencies. Not "few" — none. `python3` and the standard library is
the whole requirement, because the first thing a reader should be able to do is run
it on the phone they are holding.

---

## Method: four levers, in order of leverage

### 1. Bits (and why 4-bit is not the end of the story)

Quantization divides the denominator. It is the cheapest lever and the only one that
needs no runtime changes at all.

The interesting part is where it stops being free. `exp_quant` measures, on trained
weights rather than Gaussian noise, the trade at each width — weight error,
held-out loss, argmax agreement, logit cosine — plus the group-size metadata
overhead that makes very low bit rates less impressive than their headline:

* at 32-weight groups, 2-bit carries 8 bytes of scale/zero-point per 32 weights — so
  **"2-bit" costs exactly as much as 4-bit** (4.00 vs 5.00 bits/weight measured), and
  there is no reason to pay the quality bill. At 128-wide groups it drops to 2.51,
  which is where sub-4-bit starts to make sense;
* below ~3 bits, error stops being a rounding artefact and starts being a modelling
  change (ternary/BitNet-style models are *trained* for that regime; post-hoc
  ternarization of a 4-bit model is not the same thing);
* quality damage is not uniform across tensors — embedding and output matrices
  tolerate far less error than FFN blocks, which is why every serious format
  (Q4_K, MXFP4, AWQ/GPTQ scales) is mixed-precision in practice.

Measured tables are in [RESULTS.md](RESULTS.md).

### 2. Sparsity: route, don't read everything

A 128-expert top-4 model reads 1/32 of its expert weights per token. That is a 32×
cut in the denominator, given away for free by the architecture. But it only becomes
*throughput* if the runtime can read the selected experts fast enough and reuse the
rest — which turns the problem into a cache problem, and a cache problem is only
soluble if routing is skewed.

So the model is treated as a test subject. `exp_moe` records, for every token and
every layer, which experts fired; then:

* the load histogram and Gini coefficient (is selection uniform? then caching is
  pointless and every offloading paper is measuring noise);
* the **LRU hit rate against cache size** over the real request stream — the number
  the planner needs and cannot guess;
* **channel concentration** in the FFN: what fraction of a block's output magnitude
  lives in what fraction of channels, which is the licence for
  relevance-driven loading (PowerInfer's hot/cold neurons).

The planner ships with a one-parameter stand-in curve for when you have no trace
(`default_hit_rate`), and RESULTS.md prints the measured curve next to it so you can
see how wrong the assumption is on this subject. On a 6-expert toy it is *too small
to be skewed*; on production MoEs the same measurement gives a 77%+ hit rate from
~10% of the pool. Shipping the assumption without that comparison would be exactly
the kind of overclaim this lab exists to avoid.

### 3. Speculation: amortise the pass you already paid for

In a streaming engine the expensive event is *a weight pass*, not a token. Speculative
decoding converts one pass into up to `k+1` accepted tokens:

```
E[tokens per pass] = 1 + α + α² + … + α^k        (α = per-token acceptance)
```

At α = 0.7, k = 4, one pass yields 3.77 tokens instead of 1 — the same bytes,
3.8× the output. `exp_speculation` Monte-Carlos this against the closed form to
confirm the arithmetic before it goes anywhere near a plan. Note the asymmetry that
makes speculation *especially* attractive here: the draft model is small and stays
resident, while the target model is the one whose bytes you are amortising. That is
precisely the regime where speculative decoding stops being a latency trick and
starts being a bandwidth trick.

### 4. Scheduling: overlap what can be overlapped, and measure the rest

`pocket/streaming.py` runs a model whose weights live in a quantized file, one block
at a time, with a background reader filling a ring of buffers. Two design decisions
are load-bearing, and both are justified by measurements in RESULTS.md:

**`pread` into reusable buffers, not `mmap`.** Mapped pages count against RSS until
the kernel decides otherwise, so a file larger than RAM thrashes instead of
streaming. On this lab's 768 MB transport blob, `mmap` peaked at **798 MB of RSS**
— essentially the file — while the ring-buffer reader held **27-33 MB** for the same
bytes at comparable or better throughput. That is why production engines expose
`--no-mmap`, and why this one does not use it.

**Overlap is bounded by how much there is to hide.** The gain from depth `d ≥ 2` is
at most `1/(1 − transfer_share)`; if a block's transfer is 12% of its compute, no
amount of pipelining buys more than 1.14×. This lab reports the measured share next
to the measured gain instead of asserting that prefetching "helps".

One more result worth stating up front, because it is the least convenient: with a
*pure-Python* decoder, prefetching made throughput **worse** (74 → 64 tokens/s at
depth 1, on the int8 store). The reader thread and the kernels share one GIL, there
is nothing to hide behind the ~3.5 MB/s reference decoder, and there is a thread
handoff to pay for. The scheduler is not the problem; the kernel is. Saying so is
more useful than reporting the one run where the wind blew the other way.

---

## What the plan looks like

`python3 -m pocket.roofline` answers "what will this model do on this device, and
which placement wins", by trying every placement that fits and keeping the fastest:

| strategy | what it means | when it wins |
|---|---|---|
| `vram` | weights + KV resident in VRAM | small model, big GPU |
| `ram` | resident in system memory, CPU arithmetic | unified memory, or model < RAM |
| `moecache` | hot experts in VRAM, cold experts computed on the CPU, overflow from disk | the MoE sweet spot on a consumer box |
| `pcie` | hot experts in VRAM, cold experts copied over PCIe | weak CPU, strong GPU |
| `stream` | read weights from storage per token | dense model bigger than RAM (i.e. never *fast*) |

and it prints the reason, not just the number:

```
$ python3 -m pocket.roofline gpt-oss-120b --device laptop-16gb --bits 4
model                      device                     strategy  GB/token    tok/s  bottleneck
gpt-oss-120b               Laptop, 16 GB              stream        2.96     3.29  storage+ram
  gpt-oss-120b / Laptop, 16 GB: 13.7 GB of hot experts cached in ram (35% hit rate);
  1.90 GB/token read from storage
```

The lever stack — same model, same hardware, one technique at a time — is the
reusable part. On a 24 GB RTX 4090, `gpt-oss-120b` goes from *does not fit* (fp16
resident) to 40+ tokens/s with expert caching, which is within the range production
offloaders report for that exact pairing. On the same GPU, a dense 100B stops at
~2 tokens/s no matter what you do to it, because 27.8 GB/token still has to arrive.

---

## Using it with REA

REA already inspects model files without loading them. `inspect_model_artifact` reads
a GGUF v2/v3 or safetensors header and reports every tensor's name, type, shape,
offset and byte length — which is the entire input a streaming scheduler needs:

```bash
rea inspect-model-artifact model.gguf --json --detail summary \
    --evidence_output model-evidence.json
python3 -m pocket.reaadapter model-evidence.json --device laptop-16gb
```

The adapter answers questions the file listing makes answerable: how many bytes per
block, which tensors must stay resident because they are indexed by token id rather
than by layer order (the embedding and output matrices — they *cannot* be prefetched
sequentially), how deep the prefetch window should be given your storage, and which
single tensor is so large that reading it as one chunk guarantees a stall.

No GGUF to hand? `python3 -m pocket.fixtures --preset moe-120b --out fix.json`
writes a schema-shaped inspection document for a gpt-oss-120b-shaped or
Llama-3.1-70B-shaped architecture, and `pocket/gguf.py` writes real GGUF v3 fixtures
for parser testing.

---

## What this does *not* show

Being precise about the limits of the evidence is the whole point, so:

* **The subject is a toy.** 30k–67k parameters, trained on a synthetic corpus for a
  minute. It is big enough to exhibit quantization damage, non-uniform routing and
  channel concentration — and far too small to predict a real model's numbers. Every
  such number in RESULTS.md is a *demonstration of the method*, not a claim about
  Llama-3.
* **The kernels are pure Python**, ~1000× slower than native. Absolute tokens/second
  here are meaningless; the *ratios* (bits saved, bytes moved, RSS flatness) are not.
  The roofline takes device numbers as parameters for exactly this reason: run
  `pocket.hwprobe` on your own machine and plug them in.
* **The storage in this sandbox is fast and RAM-backed.** Measured 1.4 GB/s
  sequential from a cold page cache, versus ~1.5 GB/s random on a good consumer NVMe
  and ~0.25 GB/s on a phone's UFS. Run the probes; do not trust a number from a
  datasheet, including the ones in `catalog.py`.
* **The cache-hit curve is an assumption unless you measure it.** One parameter,
  printed next to a measured curve, and labelled as a shape not a fact.
* **No KV-cache paging, no prefill batching, no tensor parallelism.** All three
  matter at long context and all three are future work (see below). Context is
  modelled only as a lump of bytes charged against capacity.

---

## Related work, and what is different here

The ingredients are not new. What is missing from most write-ups is the arithmetic,
the measurement, or the honesty about which one you got.

| work | what it contributes | relationship |
|---|---|---|
| [LLM in a flash](https://arxiv.org/abs/2312.11514) (Apple, 2023) | windowing + row-column bundling; 4–5× CPU, 20–25× GPU over naive loading with half the model in DRAM | the clearest statement that flash-backed decode is IO-bound; this lab's transport measurement is the same effect with the sizes parameterised |
| [FlexGen](https://arxiv.org/abs/2303.06865) (2023) | offloading with throughput-oriented scheduling at batch > 1 | batch-1 streaming (this lab) is the opposite regime: nothing to amortise across the batch |
| [PowerInfer](https://arxiv.org/abs/2312.12456) (SOSP'24) | hot/cold *neuron* locality on a consumer GPU; 13.2 tok/s average on OPT-175B-class models on one RTX 4090, up to 11.69× over llama.cpp | the source of the concentration measurement in `analysis.py`; this lab measures the same statistic on a subject small enough to audit |
| [PowerInfer-2](https://arxiv.org/abs/2406.06282) (2024) | the same idea on a phone, with flash-aware splitting | the endpoint this study's phone column is aimed at |
| [Fast MoE inference with offloading](https://arxiv.org/abs/2312.17238) (2023) | expert-level offload + speculative pre-fetch for Mixtral | the template for the `moecache` strategy |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `--n-cpu-moe`, `--no-mmap` | production implementation: CPU-computed cold experts, mmap last-resort | `moecache` models exactly this; the mmap-vs-pread measurement here is why `--no-mmap` exists |
| [ktransformers](https://github.com/kvcache-ai/ktransformers) | expert offload + quantized kernels for MoE on consumer hardware | the practical target for a native version of this scheduler |
| [In-flash processing](https://dl.acm.org/doi/10.1145/3695053.3731073) (MICRO'24) | do the GEMV inside the SSD controller; 5.74 tok/s for 20B | says out loud that SSD bandwidth alone caps offloaded dense decode at ~0.1–0.2 tok/s, which is the wall in Section 1 |
| [BitNet b1.58](https://arxiv.org/abs/2402.17764), [QMoE](https://arxiv.org/abs/2310.16795) | 1.58-bit and sub-1-bit weights | the 2-bit column is *post-hoc* quantization; these are what a model trained for that regime can do |
| [Speculative decoding](https://arxiv.org/abs/2211.17192) (2022) | draft/verify acceptance arithmetic | lever 3, including the closed form this lab verifies by simulation |

The contribution of this directory is not a new algorithm. It is a *runnable,
dependency-free apparatus* that measures the three decisive quantities on the machine
in front of you, refuses to quote a tokens/second figure it did not produce, and
plugs into REA's existing model inspection so a plan can be made from a file listing
in one step.

---

## Running it

```bash
cd research/pocket-100b

python3 -m unittest discover -s tests -t .   # ~10 s, no dependencies
python3 run_experiments.py --quick           # ~2 min, small budgets
python3 run_experiments.py                   # ~10 min on 2 cores; writes RESULTS.md

python3 -m pocket.hwprobe                    # what your machine actually does
python3 -m pocket.roofline gpt-oss-120b --device laptop-16gb --bits 4
python3 -m pocket.fixtures --preset moe-120b --out /tmp/fix.json
python3 -m pocket.reaadapter /tmp/fix.json --device desktop-32gb-4090
python3 -m pocket.streamprobe --store artifacts/subject-dense.int4.pksw --tokens 8 --depth 2
```

`--quick` and the full run differ only in budgets, so CI can afford the shape of the
result without the wall-clock.

---

## Where this goes next

Ordered by expected payoff per unit of work, and by how much of it is already
scaffolded here:

1. **A native dequantize + matvec kernel** (NEON/AVX2, or a `ctypes` shim over
   ggml). The current decoder is the bottleneck by two to three orders of magnitude;
   everything else in the pipeline is already instrumented.
2. **Process-parallel decode.** The GIL result in Section 4 says a reader thread
   cannot help a Python decoder; a forked reader that releases nothing *can*, and the
   measurement that proves the ceiling is already in `exp_transport`.
3. **A real router trace** from an open 128-expert MoE, used to replace
   `default_hit_rate` with a measured curve — the single assumption with the largest
   effect on the plan.
4. **KV-cache paging and quantization.** At 32k context the cache is a substantial
   fraction of the byte budget, and it is currently charged as a lump.
5. **Prefill/verify batching.** Speculative decoding moves the workload from decode
   (memory-bound) to verify (compute-bound), which is the only lever that can make a
   compute-rich, bandwidth-poor device — a phone — competitive.

---

*Start with [RESULTS.md](RESULTS.md): every table there was generated by
`run_experiments.py` on the machine it describes.*
