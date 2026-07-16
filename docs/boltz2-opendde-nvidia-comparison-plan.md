# Boltz-2 + OpenDDE vs NVIDIA — fair-comparison plan (research)

Research/recommendation only. No GPU was rented and no benchmark was run for
this doc; it answers the four questions Moritz asked (Telegram, 2026-07-16) by
citing existing committed docs and NVIDIA's published NIM materials, and it
estimates the cost of the one run that is genuinely still missing.

## Provenance (read first)

Moritz recalls benchmarking Boltz-2 against an NVIDIA NIM container and a
"~5x" figure. That run is **not on record anywhere in tt-bio** — no instance
id, no raw timings, no NIM version, no recycling/sampling settings. The only
logged Boltz-2-vs-NVIDIA claim is `docs/boltz2-tt-vs-nvidia.md`'s informal
"Blackhole p150 ≈2.4–2.9x a DGX per dollar at L=512/1024", which that doc
explicitly flags as unverified (no rigorous table exists; see memory
`boltz2-nvidia-table`). The two numbers are **not reconcilable** without
Moritz's raw data: "5x" may be a latency ratio at one length, "2.4–2.9x" is a
\$-normalized throughput ratio at L=512/1024 — different metrics. This doc
does not silently pick either; it states the gap and asks Moritz for the raw
run, or for a clean re-run on NIM v1.8.0.

This mirrors the tt-atom Orb fair-comparison fix (`workstreams/tt-atom-orb-gpu-
fair-comparison.txt`, state `orb-gpu-fair.md`): compare the best *easy,
out-of-box* GPU path a user actually runs; disclose asymmetry; commit hard
evidence; no fabricated numbers.

## Q1 — Boltz-2 GPU-side optimization survey

**The NVIDIA NIM for Boltz2 (`nvcr.io/nim/mit/boltz2`, latest v1.8.0, June
2026) is the best officially-supported, easy GPU inference path. There is no
faster documented easy path.**

What the NIM does under the hood (NVIDIA NIM for Boltz2 docs, "Optimization"
and "Performance" pages, v1.6.0/v1.8.0):

- TensorRT engines for the pairformer (structure) and affinity modules;
  PyTorch fallback. Since v1.8.0 a **unified optimized backend** auto-selects
  TensorRT for short sequences (≤768 residues) and optimized PyTorch kernels
  for longer ones — no flag needed.
- Custom kernels: Fused AdaptiveLayerNorm, DualGemm sm80, gated sigmoid sm80,
  FAv2/v3 TriangleAttention / AttentionPairBias, plus pre-computed masks and
  Pairformer/DiT buffers.
- TensorFloat32 for diffusion (`NIM_BOLTZ_ENABLE_DIFFUSION_TF32=1`, default).
- Throughput tuning via request params `recycling_steps` / `sampling_steps` /
  `diffusion_samples`; concurrent requests for batch throughput.

TensorRT vs stock-OSS (plain PyTorch `boltz predict`) speedup, H100, structure
prediction (NIM v1.6.0 perf table): **1.45x–6.44x** across 18 targets (e.g.
186 res 11.07s→1.72s; 2033 res 123.07s→79.55s). v1.8.0 adds "up to 1.7x on
H100" from the unified backend. So the NIM is materially faster than stock
`boltz predict`; stock is the slow path.

Concrete NIM reference latencies (v1.6.0, TensorRT, no templates), for sizing
the comparison:

| seq len | H200 | B200 | H100 |
|---|---|---|---|
| 186  | 1.44s  | 1.56s  | 1.72s  |
| 530  | 5.66s  | 6.01s  | 6.63s  |
| 858  | 12.47s | 14.11s | 14.56s |
| 1464 | 25.77s | 35.55s | 29.49s |
| 2033 | 71.01s | 83.43s | 79.55s |

**Verdict: if Moritz ran the NIM, he ran the best easy GPU path.** vLLM-style
serving does not apply (not a decoder LM); a separate TensorRT export *is*
what the NIM already does. Nothing faster/newer exists officially.

Caveats to disclose before trusting any Boltz-2 ratio (these are the real
fairness levers, not the choice of path):

1. **NIM version.** v1.8.0's unified backend is up to ~1.7x faster on H100
   than pre-1.8.0. If Moritz's run was on an older NIM, re-running on 1.8.0
   H200/B200 could move the GPU number by up to ~1.7x. **Confirm the version.**
2. **recycling_steps / sampling_steps must match both sides.** The TT side
   (`docs/boltz2-tt-vs-nvidia.md`) used `--fast` at Boltz-2 defaults
   (recycling 3, sampling 200). The NIM's benchmark-table settings are not
   fully published; the NIM's own test example uses recycling 3 / sampling 50.
   If the NIM run used fewer sampling steps than the TT side, the GPU number
   is unfairly fast. Align both sides to the same (recycling, sampling,
   samples) before quoting a ratio.
3. **Precision defaults are the fair "easy" path on both sides** — NIM TF32 +
   TRT (GPU) vs TT `--fast` bf8 trunk (TT). Both are what a user gets out of
   the box; disclose, do not equalize by hand-tuning (per the orb precedent).
4. **Hardware class.** The NIM requires ≥48 GB VRAM (A100/H100/H200/B200/
   GB200/RTX PRO 6000 class). A 24 GB RTX 3090/4090 cannot run it. Moritz's
   "H200 or B200-class" note is consistent with the NIM actually running.

## Q2 — OpenDDE-vs-Boltz-2 architectural similarity

Both are AF3-lineage. In tt-bio, Boltz-2, Protenix-v2, and OpenDDE all reuse the
same `tenstorrent.PairformerModule` (trunk), `DiffusionModule` (DiT diffusion),
`MSAModule`, and `ConfidenceHead` primitives. OpenDDE literally instantiates
`tt_bio.protenix.Protenix` and reuses its `Trunk` + `DiffusionModule` verbatim
(`tt_bio/opendde.py`: `self._protenix = Protenix(...)`). OpenDDE adds exactly
one novel compute seam between trunk and diffusion: `StructuralTokenExpander`
+ a 4-block refiner Pairformer (`expand_and_refine`). Boltz-2 has no such seam.

Quantified wall-clock breakdown (from `docs/opendde-port.md` "Speed vs
Boltz-2", same 117-residue `examples/prot.yaml`, warm, single Blackhole card,
each model's default settings):

| stage | OpenDDE r10/200 | Boltz-2 r3/200 |
|---|---|---|
| trunk (Pairformer, shared arch) | 7.98s (60.5%) | 1.79s (32%) |
| expand_and_refine (OpenDDE-only) | 0.47s (3.6%) | — |
| diffusion (DiT, shared arch) | 3.69s (27.9%) | 3.55s (63%) |
| confidence (shared arch) | 0.11s (0.8%) | 0.16s (3%) |
| **total (worker)** | **13.2s** | **5.6s** |

- **Shared/equivalent architecture** (trunk + diffusion + confidence, same
  primitives): ~89% of OpenDDE's production fold (11.78/13.2s) and ~98% of
  Boltz-2's (5.50/5.6s).
- **OpenDDE-novel compute** (StructuralTokenExpander + refiner): ~3.6% of
  OpenDDE's fold (0.47/13.2s); the expander alone is 2.21% and host/upload-bound
  (`docs/opendde-kernel-scout.md`). Small.
- **The production settings differ and matter.** OpenDDE runs 10 recycles /
  200 steps; Boltz-2 runs 3 / 200 (`_resolve_recycling_steps`, `tt_bio/main.py`).
  This **flips the bottleneck**: Boltz-2 is diffusion-bound (63%), OpenDDE is
  trunk-bound (60.5%). The diffusion stage is essentially identical between
  the two (3.69 vs 3.55s — same DiT, same 200 steps). The trunk differs because
  OpenDDE runs 10 cycles (vs 3) and a wider Pairformer (c_z=384 vs Boltz-2's
  c_z=128, ~3x pair channels; `docs/opendde-port.md` "Redundancy").

So: architecturally ~89–98% shared primitives and the OpenDDE-specific compute
is small (~3.6%), but the production compute *profile* differs materially
because the recycling count and pair width shift the bottleneck from diffusion
(Boltz-2) to trunk (OpenDDE).

## Q3 — Does the Boltz-2 TT-vs-NVIDIA perf/\$ ratio transfer to OpenDDE?

**No, not cleanly.** Two independent reasons:

1. **Compute-bound vs dispatch-bound balance flips.** Boltz-2 at r3 is
   diffusion-bound (63%): the DiT is 200 small kernel dispatches, the regime
   where TT is known to lose to GPU at small/medium N (esmc/protenix precedent;
   memory `opendde-trace-replay` notes OpenDDE diffusion is "compute-bound at
   this scale, not dispatch-bound like Protenix @L256"). OpenDDE at r10 is
   trunk-bound (60%): the Pairformer is O(L³) large matmuls, compute-bound —
   the regime where TT's bf8 Pairformer is most competitive with a GPU. The
   TT/GPU throughput ratio is therefore governed by different stages between
   the two models, so one ratio does not predict the other.

2. **The GPU software optimization level is asymmetric — and this is
   decisive.** Boltz-2's fair GPU path is the NIM (TensorRT engines + custom
   kernels + TF32; 1.45–6.44x over stock PyTorch). **OpenDDE has no NIM, no
   TensorRT path, and no official optimized serving container.** OpenDDE's
   only official GPU path is stock `opendde pred` (PyTorch, fp32 default,
   cuEquivariance triangle kernels on CUDA; `aurekaresearch/opendde:v1`
   Docker or `pip install opendde[gpu]`). The 4-GPU Fold-CP path is explicitly
   a demo, "still being actively optimized for performance and memory
   capacity" (OpenDDE `docs/inference_instructions.md`). So the GPU side for
   OpenDDE is the *slow, unoptimized* path, while the GPU side for Boltz-2 is
   the *fast, NIM-optimized* path. The Boltz-2 ratio (TT vs NIM-GPU) is not
   comparable to an OpenDDE ratio (TT vs stock-PyTorch-GPU): the GPU
   denominator is optimized to a different degree.

**Direction of the bias.** OpenDDE's GPU side (stock PyTorch) is slower relative
to its potential than Boltz-2's GPU side (NIM). So TT would look *better*
against stock-PyTorch-OpenDDE than against NIM-Boltz-2. Citing the Boltz-2
ratio for OpenDDE is therefore conservative-for-TT on raw throughput, but it
hides a real asymmetry (TT-OpenDDE beats stock PyTorch; TT-Boltz-2 had to beat
a NIM). That is not a clean transfer and is not honest to present as
equivalent.

**Reasoned verdict.** The ratio is plausibly in the same order of magnitude —
same model family, ~89% shared compute, and TT still wins on perf/\$ because a
p150 is ~23x cheaper than a DGX-class card. But the **magnitude is not
transferable** and the sign of the error is not knowable without a run: it
could move ~1.5–2x either way depending on (a) how much of OpenDDE's
trunk-bound time a NIM-equivalent optimization would have saved (none exists,
so the GPU trunk runs at OSS-ish speed, which helps TT), and (b) whether the
10-recycle trunk on GPU hits a memory-bandwidth or compute limit that the
NIM's TRT engines specifically avoid for Boltz-2.

## Q4 — Recommendation

**No — intellectual honesty requires an independent OpenDDE-vs-GPU run; we
cannot cleanly cite the Boltz-2 number for OpenDDE.** Reasons: (a) the compute
profile flips (diffusion-bound → trunk-bound), (b) the GPU software path is
asymmetric (NIM for Boltz-2 vs stock PyTorch for OpenDDE — no OpenDDE NIM
exists), and (c) the production recycling count differs (3 vs 10), changing
the bottleneck. Citing the Boltz-2 ratio for OpenDDE would be a fabricated/
unfair number under this project's "no fabricated/unfair numbers on public
claims" standard.

The good news: the run is **cheap and the harness already exists**. The P11
reference leg (`docs/opendde-port.md` "P11", state `opendde-9dsg-reference-
dockq.md`) already ran stock OpenDDE on a rented vast.ai RTX 4090 at production
settings (10 recycles / 200 steps / best-of-5, 9dsg) — that *is* the fair GPU
comparison point (stock OpenDDE PyTorch, the best easy official path). It just
didn't record clean wall-clock timings, and it ran on a 24 GB consumer card,
not the H200/B200-class hardware the Boltz-2 NIM ratio is sized against.

**Cost/effort estimate for a clean OpenDDE-vs-GPU perf run** (vast.ai access
per memory `vast-ai-access`; ~$8.31 credit remains):

- **Hardware.** Match the Boltz-2 NIM hardware class so the price ratio stays
  the same ~23x: vast.ai on-demand H200 141 GB (~$1.5–3/hr) or B200 if
  available. An RTX 4090 24 GB (~$0.40/hr) works for ≤700-res targets at bf16
  (fp32 OOMs at 200 steps on 24 GB, per P11) but is not the right perf/\$
  basis against a p150-vs-DGX-class claim.
- **Software.** Stock `opendde pred` (official `aurekaresearch/opendde:v1`
  Docker or `pip install opendde[gpu]`), bf16, cuEquivariance triangle kernels
  (the package default on CUDA). **No hand-tuning** — per the orb precedent,
  stock is the fair easy path, and there is no faster official path to use.
- **Targets.** Reuse `examples/prot.yaml` (117-res, matches the Boltz-2 speed
  table in `docs/opendde-port.md`) plus a size sweep (512 / 1000 / 2000) to
  characterize the curve and crossover, mirroring the orb methodology.
- **Settings.** 10 recycles / 200 steps / 1 sample (OpenDDE production) **and**
  a 3-recycle leg to isolate the recycling-count effect; same MSA input as the
  TT side; record raw per-stage timings, instance id, exact commands, git SHA.
- **Time/cost.** ~1–2 hr rental (env build + MSA + size sweep). ~$2–6 on H200,
  ~$0.5–1 on RTX 4090 (small targets only). Well inside the remaining credit.
- **Then** compute perf/\$ on the *same* price basis as the Boltz-2 NIM
  comparison (p150 vs H200/B200), and commit a results JSON like the orb task
  did (`benchmarks/orb_perf_dollar_*.json`).

**Reconciling "5x" vs "2.4–2.9x".** Moritz's NIM run is not on record (no
version, instance id, raw timings, or settings). The only logged Boltz-2 GPU
claim is the "2.4–2.9x a DGX per dollar" in `docs/boltz2-tt-vs-nvidia.md`,
explicitly flagged unverified. These are not reconcilable without Moritz's
raw data, and they may be different metrics (latency ratio vs \$-normalized
throughput). Do **not** pick one silently. Either get Moritz's raw NIM run, or
re-run the Boltz-2 NIM leg cleanly on v1.8.0 H200/B200 alongside the OpenDDE
leg so both numbers come from the same harness.

## Methodology guardrails (inherited from the orb precedent)

- Best easy/out-of-box GPU path only: NIM for Boltz-2, stock `opendde pred` for
  OpenDDE. No hand-tuned CUDA on either side.
- Disclose the asymmetry: the GPU optimization level differs between the two
  models (NIM/TRT vs stock PyTorch). State it in any public doc, don't hide it.
- Every number cited: either from a committed tt-bio doc (linked above) or
  from NVIDIA's published NIM perf/optimization pages (linked). No invented
  figures.
- Same recycling/sampling steps on both sides of each comparison; align the
  NIM's default sampling_steps with the TT side's 200 before quoting a ratio.

## Next step for Moritz

Decide one of:

1. **Commission the ~\$2–6 OpenDDE-vs-GPU run on H200/B200** (recommended) —
   gives a clean, independent, evidenced OpenDDE perf/\$ number with the same
   harness and price basis as the Boltz-2 NIM leg.
2. **Accept the Boltz-2 ratio as a stand-in** with an explicit caveat —
   "OpenDDE shares ~89% of Boltz-2's compute and the same trunk/diffusion
   primitives, but its GPU path is stock PyTorch (no NIM exists), so this is a
   conservative-for-TT stand-in, not a measured OpenDDE number." Not
   recommended for a public claim — the compute-profile flip and the
   asymmetric GPU optimization make it an unfair-as-presented number.

Either way, first confirm the NIM version + settings of Moritz's existing
Boltz-2 run so the Boltz-2 side of the ratio is itself evidenced.
