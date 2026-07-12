# HunyuanImage-3.0 — Bring-up & Optimization (tt_hw_planner evaluation)

**Status: STOPPED at sizing gate — Galaxy access required, not yet confirmed available. No bring-up attempted.**
**Date: 2026-07-10. Requestor: Dalar Vartanians (via Moritz, Slack).**

## Step 0 — Sizing research (done before touching any hardware, per Dalar's explicit instruction to independently verify his "needs BH Galaxy" hunch rather than assume it)

### Model architecture & size (Tencent Hunyuan team, arXiv:2509.23951, official repo github.com/Tencent-Hunyuan/HunyuanImage-3.0)
- **80B total parameters, 13B activated per token**, Mixture-of-Experts with **64 experts**.
- **Not a DiT** (unlike most image-gen models in tt-metal today). It's a **native unified autoregressive multimodal model** ("MoE + Transfusion"-style) that handles understanding and generation in one AR framework — a genuinely new architecture class for tt-metal, not an incremental variant of anything already ported.
- Variants: Base (≥240GB VRAM per vendor), Instruct and Instruct-Distil (≥640GB VRAM per vendor) — we only need to size the Base model for bring-up.

### Memory footprint — three independent estimates, all point the same direction
| Source | Estimate |
|---|---|
| Official repo (Tencent-Hunyuan/HunyuanImage-3.0 README) | **≥ 240 GB minimum** (stated as "3 × 80GB-class GPUs", multi-GPU inference explicitly recommended for Base) |
| Community estimate, unquantized BF16 | **~177–181 GB** (80B × 2 bytes ≈ 160GB weights + KV-cache/activation overhead) |
| Community INT4 quant (unofficial, no vendor-supported recipe, no tt-metal kernel path) | **~45 GB** |

### QB2 actual hardware (verified live via `tt-smi -s` on qb2, 2026-07-10, not assumed)
- 4 devices, `dram_speed: "16G"` each → **16GB GDDR6 per card**.
- **Aggregate device memory across all 4 cards: 64GB.**
- (Board reports as `p300c` — this is qb2's known firmware misdetection of Blackhole P150 pairs, not an actual P300; see prior fleet notes. Physical config is still 4× P150, 16GB each.)

### Closest existing TTNN reference in tt-metal (checked `models/` tree on the fork, `feature/tt-hw-planner` branch)
- **`models/tt_transformers` — Mixtral-8x7B**: the only existing MoE model in the repo. 47B total params, **13B active/token — coincidentally the same active-parameter count as HunyuanImage-3.0**. Telling data point: per `models/model_targets.yaml` and the `tt_transformers/README.md` hardware column, Mixtral-8x7B's minimum supported target is **QuietBox/T3K-class (8-chip)** hardware — not a 4-card subset. A *smaller* MoE model already doesn't fit in QB2's aggregate memory in tt-metal's own precedent.
- **`models/demos/stable_diffusion_xl_base`**: closest image-generation pipeline shape (VAE/tokenizer + diffusion steps), but ~3.5B params, not MoE, not autoregressive — useful only for the image-tokenizer/VAE plumbing pattern, not for the core architecture.
- No existing reference covers the AR+diffusion unified architecture itself — this part of bring-up would be **NEW**, not REUSE/ADAPT, regardless of which hardware it lands on.

### Verdict
**HunyuanImage-3.0 does NOT fit a single QB2.** 64GB aggregate device memory is:
- ~2.8–3.75× short of the lowest (unofficial, unquantized-precision-adjacent) BF16 estimate (177GB),
- ~3.75–6× short of the vendor's own stated minimum (240GB),
- and even the hypothetical INT4 floor (45GB, no vendor or tt-metal support for this architecture) would leave under 20GB headroom for KV-cache, 64-expert MoE routing buffers, and image-tokenizer/VAE activations — not a credible starting point for a first-pass bring-up of a brand-new architecture class.

tt-metal's own existing precedent reinforces this independently: Mixtral-8x7B, a *smaller* MoE model with the same 13B active-parameter count, already requires a full 8-chip QuietBox/T3K system rather than 4 cards.

**Dalar's hunch is correct: this genuinely needs the BH Galaxy 4×8 mesh, not QB2.** Per hardware-routing instructions, this is a legitimate stop point, not a failure — proceeding onto Galaxy requires:
1. Confirming Galaxy is actually free (shared production resource, active external customer, Moritz on Germany time).
2. Confirmed SSH access/credentials for the Galaxy (as of the Slack thread, Moritz's key was sent to Dalar's team but not yet confirmed added — "we will add your key soon", no confirmation since). No Galaxy hostname or credentials exist in this environment.

**Action taken:** Telegrammed Moritz immediately with this verdict and the access blocker, rather than guessing at Galaxy connection details.

## Due diligence summary (combined pre-work deliverable)
- **Architecture**: 80B/13B-active, 64-expert MoE, unified AR multimodal (image gen + understanding), not DiT.
- **Closest TTNN reference**: Mixtral-8x7B (MoE routing precedent, wrong hardware class already at 47B) + SDXL-base (image pipeline plumbing only). No AR+diffusion-unified reference exists — core bring-up is NEW work either way.
- **Realistic perf target**: not yet establishable — no representative published GPU throughput numbers were found beyond qualitative "multi-GPU inference recommended"; this needs to be pulled from the technical report or measured on a reference GPU setup once Galaxy access exists, before setting a tt-metal perf target.
- **Expected bottlenecks** (architectural, independent of hardware): 64-expert MoE routing/token-dropping at this activation ratio (13B/80B ≈ 16% active) will stress collective-communication (CCL) and expert-placement logic across whatever mesh it lands on; the AR decode loop (KV-cache) combined with diffusion-style image generation is a genuinely novel control-flow shape for tt-metal's existing model families, which are either pure-AR-LLM or pure-DiT today — expect this to land mostly in the "auto-onboard NEW family" path of tt_hw_planner rather than "scaffold from closest demo family".

## tt_hw_planner tool evaluation notes (partial — no run attempted)
Reviewed `GETTING_STARTED.md` on the `feature/tt-hw-planner` branch (worktree created at `/home/ttuser/tt-metal-hunyuan`, no changes made, no push). The tool's `auto-up <model> --box <B> --mesh <M>` and `optimize <model> --devices all` CLI is hardware-topology-aware by design (explicit `--box`/`--mesh` flags), so once Galaxy access exists the same tool should be usable there without modification — this wasn't tested, just confirmed from the flag surface. No attempt was made to run scaffold/prepare-plan/auto-onboard against real inputs since there is no hardware target to run it on yet.

## Model Completion Checklist — status
| Item | Status |
|---|---|
| Hardware selection justified by size/architecture/existing references | **Done** — this document |
| Functional end-to-end run | Not started (blocked on Galaxy) |
| Correct outputs on real inputs | Not started |
| Working demo | Not started |
| Performance in acceptable range | Not started |
| Tracy-validated perf + manual op-cost analysis | Not started |
| Trace support | Not started |
| 2CQ support | Not started |
| Documented model-specific blockers | This document — Galaxy access is the blocker |

## 7-criteria table row (honest partial — most stages not reached)
| Criterion | Score/Notes |
|---|---|
| Bring-up Efficiency | N/A for actual bring-up. Sizing-gate research itself was fast (~30 min): live `tt-smi` check + vendor repo + tt-metal `models/` grep gave an unambiguous answer without needing to touch hardware. |
| Optimization Effectiveness | N/A — not reached. |
| Final Performance Achievement | N/A — not reached. |
| Optimization Discovery | N/A — not reached. |
| Engineering Insight Quality | Partial — tool's `--box`/`--mesh` flag design suggests it's meant to generalize across hardware topologies including Galaxy, but this is read from docs, not verified in practice. |
| Workflow Usability | Partial — CLI surface reads cleanly from `GETTING_STARTED.md` alone; real usability (env/PATH auth inheritance, Python version pinning issues seen in the parallel XTTS-v2 pass) unverified for this model, no run attempted. |
| Adoption Intent | Cannot assess — blocked on Galaxy access confirmation. |

## Bottom line
Correctly identified, via verified evidence (live hardware check + vendor docs + existing tt-metal precedent), that HunyuanImage-3.0 does not fit QB2 and genuinely requires Galaxy. Stopped per explicit instruction rather than guessing at Galaxy access. Telegrammed Moritz. No fabricated numbers — every figure above is either read from `tt-smi -s` output captured live on qb2, or from the cited vendor/repo sources.
