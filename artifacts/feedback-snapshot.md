# tt_hw_planner evaluation — feedback

Tool under evaluation: `tt_hw_planner` (tenstorrent/tt-metal PR #46283, fork `apande-TT/tt-metal`).
Evaluator: Moritz's coworker agent, on behalf of Dalar Vartanians.
Host: **qb2 (tt-quietbox2)**, single Blackhole P150, physical card 1, `TT_VISIBLE_DEVICES=1`.
All numbers below are from live runs executed on qb2 on **2026-07-10** — none restated from the shipped reference report.

---

## Model A — `coqui/XTTS-v2` (multilingual TTS)

This is the **second extension pass** on Model A. A first pass earlier the same day cleared blockers
(b) and (c) and got partial per-component PCC + a single-op Tracy profile, but stopped short of the
real end-to-end pipeline (it only found the broken `demo.py` scaffold and concluded "no e2e wiring").
This pass found and exercised the model's **actual** wired pipeline (`tt/pipeline.py`,
`demo/demo_tts.py`, `tests/e2e/test_e2e_tts.py` — not `demo.py`, which is a dead scaffold) and pushed
all the way to a passing, PCC-gated, Tracy-profiled end-to-end run on real qb2 hardware, per Dalar's
"push through the dead-ends" direction.

### 0. Pre-work due-diligence (BACKFILLED — flagged)

> Backfilled: XTTS-v2 bring-up scaffolding already existed (shipped in the fork, dated 2026-07-05)
> before this due-diligence step was requested. Reconstructed honestly after the fact.

- **Architecture / size (measured, not estimated):** loaded the real reference checkpoint and summed
  parameters directly: **466.87M total** — `gpt` (GPT-2-style autoregressive text→mel-code decoder,
  30 blocks + conditioning-perceiver cross-attn) = **441.02M**, `hifigan_decoder` (HiFi-GAN vocoder +
  nested ResNet speaker encoder) = **25.86M**. Matches the ~470M figure in public XTTS-v2 writeups.
- **Closest existing TTNN reference:** correcting the earlier pass's assumption — `bringup_status.json`'s
  actual `common_reuse` entries are **generic infra helpers**, not `tt_transformers` attention/MLP:
  `models/common/rmsnorm.py`, `models/common/lightweightmodule.py`, `models/common/tensor_utils.py`,
  `models/common/utility_functions.py`. `sibling_hf_id` points at a local path
  (`/local/ttuser/apande/models/XTTS-v2-hf`) with no resolved `model_type` — i.e. **the tool found no
  good sibling template for this model at all**, which is why 29/32 components are `NEW`.
- **Component split:** `REUSE 3 / ADAPT 0 / NEW 29`.
- **Realistic perf target (published, cited):** XTTS-v2 on GPU achieves RTF **≈0.3×** (≈3× faster than
  real-time) with **150–400 ms** first-chunk latency depending on GPU (320 ms on an RTX 5090); CPU RTF
  is **≈1.41×** (slower than real-time). [GIGAGPU TTS latency benchmarks](https://gigagpu.com/tts-latency-benchmarks/),
  [GIGAGPU XTTS-v2 VRAM](https://gigagpu.com/xtts-v2-vram-requirements/). That is the bar a tuned TT
  bring-up would eventually be judged against — **not approached in this pass** (see §2, decode is
  ~117 s for 4 tokens at reduced depth; far from real-time).
- **Expected bottleneck (confirmed, see §2):** many small parametrized `conv1d`/`conv_transpose1d`/
  weight-norm ops with no native fused ttnn 1-D-conv equivalent, **plus** — newly identified this pass —
  the AR GPT-2 decode loop has **no KV-cache**: it is a repeat-prefill (recomputes the growing sequence
  from scratch every generated token), which dominates wall-clock far more than the conv ops do.

### 1. The three documented blockers

**(c) qb2 fabric `TT_FATAL @ tt_cluster.cpp:281` — CLEARED (re-verified).** Same fix as before:
```
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```
Confirmed again from a clean shell this pass: without it, `ttnn.open_device` hits
`TT_FATAL: Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type`
(P150 misdetected as a 1-chip P300). With it: clean `MeshDevice(1x1 grid, 1 devices)` open/close.

**(b) Python 3.10-vs-3.12 / `TTS` package — CLEARED, and confirmed to be a non-issue in the built env.**
The built `python_env` (uv-managed CPython 3.12.12, matching the `_ttnn.so` ABI) **already has
`coqui-tts==0.27.5`** installed (the maintained idiap fork, `TTS`-namespace-compatible, supports 3.12 —
unlike upstream coqui's dead `TTS` package which caps at `<3.12`). `import TTS.tts.models.xtts.Xtts`
and the full `_reference_loader.py` self-test **work out of the box**, loading real trained weights
from the HF cache already warm on this box. **No second interpreter, no code change, no pip install
needed this pass** — GETTING_STARTED.md's 3.10 assumption is simply stale for this checkout.

**(a) tool's `claude` auth in its subprocess — REAL blocker, confirmed again, escalated to Moritz.**
`claude -p "..."` as `ttuser` on qb2 returns **"Not logged in · Please run /login"** — no
`ANTHROPIC_API_KEY`, no `~/.claude/.credentials.json`. New this pass: **the tool's own environment
doctor lies about this.** `models/experimental/perf_automation/setup_env.sh` reports
`[ok] auth: claude CLI login found` — but its check (line 109) only tests for the *existence* of
`~/.claude.json` (created by `claude --version` alone, with no login), not
`~/.claude/.credentials.json` (the actual token). This is a **false-positive in the tool's own
self-check** that would mask exactly this blocker for a new user running the doctor script and
concluding they're ready. Sent to Moritz via Telegram in real time this pass; no credential-copying
attempted per the "ask before duplicating auth" policy. Until resolved, none of the tool's LLM-driven
steps (`up --auto`, `promote`, `optimize`'s repair loop) can run unattended on this box — everything
below was done by direct execution/investigation instead (the "additional agentic work" fallback).

### 2. What we actually ran live (own numbers, this pass)

**Full per-component PCC suite (all 29, not a sample):**
```
1 failed, 13 passed, 15 skipped, 1 warning in 217.63s
```
- **13 PASS** — native ttnn ports numerically verified against real trained weights on-device.
- **1 FAIL** — `test_mel_spectrogram`: `RuntimeError: The size of tensor a (20480) must match the
  size of tensor b (19712)`. A real framing/shape mismatch between the ttnn mel front-end and the
  torch reference on this box — contradicts the shipped `RUN_REPORT.md`'s "PCC-verified" claim for
  this component. Not investigated further (root-cause would need diffing STFT padding/centering
  behavior across torch/torchaudio versions — flagged as a genuine open bug, not fixed this pass).
- **15 SKIP** — the auto-generated test harness's synthetic-input shape guesser (`_make_arg_for()`)
  produces tensors incompatible with the real submodule signature (e.g. `attend` gets a 3-D q/k/v when
  the real call needs 4-D; `conv1_d` gets `(64,3072)` against a `1024×3072` weight). **These are test
  harness issues, not stub bugs** — but they mean **over half the shipped "29/29 PCC-verified"
  components cannot be independently re-verified per-component on a fresh box.** This is the single
  most important tool-quality finding, confirmed at full scale (not just a 7-component sample).

**End-to-end pipeline test (`tests/e2e/test_e2e_tts.py::test_e2e_tts`) — PASSED, real numbers:**
```
invoked 29/29 graduated stubs; missing=[]
speaker_embedding_pcc = 0.9710
cond_latent_pcc       = 0.9989
ar_token_match        = 1.0
ar_per_step_logits_pcc= 0.9993
latents_pcc           = 0.9995
waveform_pcc          = 0.9909   <- gates on this, target >=0.95, PASSED
full_chain_waveform_pcc (supplementary, not gated) = 0.7349
```
This is the real finding the previous pass missed: **the model does have a working, fully-wired,
device-only, gate-passing end-to-end pipeline** — it's just not the file the prior pass looked at
(`demo.py`). `tt/pipeline.py` explicitly documents that the demo and the e2e test share the exact same
wiring, so a passing e2e test is a passing demo by construction.

**Functional demo (`demo/demo_tts.py`) — ran, produced real audio:**
```
e2e PCC=0.5545  (different text/seed path than the e2e test's "hello world.")
wrote TT waveform -> /tmp/xtts_tt.wav  (44544 samples @ 24000 Hz, valid 16-bit PCM WAV)
```
Deliverable #1 (functional e2e audio) is **met** via this file. Note the PCC is sentence-dependent —
0.99 for the e2e test's short greeting vs 0.55 for a longer sentence — real measured variance, not
investigated further (plausibly AR-sampling divergence compounding over more decoded tokens).

**Canonical demo (`demo/demo.py::test_demo`) — still fails, re-confirmed:**
```
ValueError: Unrecognized model in coqui/XTTS-v2. Should have a `model_type` key in its config.json...
```
Same failure as the prior pass: this is an auto-generated CPU-only scaffold that calls generic
`AutoModel.from_pretrained` on a non-HF-native (coqui-runtime) checkpoint, and is even mis-labeled
category `STT`/`==ASR`. **It was never rewired after all 29 components graduated** — a genuine gap in
the tool's own workflow (bring-up completing doesn't retarget the canonical demo entrypoint at the
real pipeline). The *real* functioning demo lives in a hand-authored sibling file the tool's own
report never points to.

**Tracy profile (real, whole-pipeline, not a single op) + trace/2CQ:**
Ran `python -m tracy -r -p -m pytest tests/e2e/test_tts_perf.py::test_tts_perf` with
`TT_PERF_TRACE=1`, `TT_PERF_NUM_CQ=2`, `TT_PERF_MAX_NEW_TOKENS=4`, `TT_PERF_LAYERS=2` (the test's own
reduced-depth perf configuration). Real captured numbers:
```
FORWARD_WALL_MS = 117559.98        # 117.6s for a 4-token decode at reduced depth — far from real-time
AICLK clamped at 800 MHz (nominal 1350 MHz) — thermal, ASIC ~47-50°C, same as the prior pass
TRACE_REPLAY_SKIPPED = AttributeError("pipeline exposes no decode_step(state); its decode is
    repeat-prefill — run the structural decode lever to add a cached single-token step")
```
Device-level op breakdown from the generated `ops_perf_results_*.csv` (21,752 op invocations,
247.9 ms total device-kernel time, 114.0 ms total host-dispatch time), aggregated by op code —
**highest-cost modules, named as requested:**

| op | % of device time | count | % of host-dispatch time |
|---|---|---|---|
| `MatmulDeviceOperation` | **28.2%** | 3094 | 11.4% |
| `UntilizeWithUnpaddingDeviceOperation` | **16.8%** | 2810 | 8.9% |
| `BinaryNgDeviceOperation` (elementwise) | 11.4% | 4100 | 7.0% |
| `ReshapeViewDeviceOperation` | 7.5% | 1382 | 3.0% |
| `TilizeWithValPaddingDeviceOperation` | 7.2% | 1728 | 4.9% |
| `PermuteDeviceOperation` | 5.2% | 1074 | 4.2% |
| `LayerNormDeviceOperation` | 5.2% | 622 | — |
| `SliceDeviceOperation` | only 3.2% device, but **34.0% of ALL host-dispatch time** | 3390 | **34.0%** |

**Reading this:** Matmul dominating device time is expected (GPT-2 attention/MLP + HiFi-GAN convs as
matmul). But **Untilize/Tilize layout-conversion ops together cost ~25% of device time** — a real,
actionable fusion/layout opportunity (the 29 independently-graduated stubs weren't optimized to share
a consistent tile layout across component boundaries). And **`SliceDeviceOperation` is the single
biggest host-dispatch cost (34%) despite being cheap on-device** — this is the smoking gun for the
`TRACE_REPLAY_SKIPPED` message above: because decode is repeat-prefill (no KV-cache), every generated
token re-slices a **new, uniquely-shaped** growing causal-mask/sequence tensor, which the resident
weight-cache (keyed by exact shape) cannot reuse, forcing a fresh host round-trip per token. This is
also why the mask cache generation log showed a **new tensor upload for every sequence length from 43
to 87+, one per AR step** — direct evidence of the same root cause.

**Trace + 2CQ — documented precisely, per Dalar's ask:**
- **AR decode stage: does NOT support trace/2CQ**, and the tool's own perf harness says exactly why —
  `PipelineDecodeAdapter` requires a `decode_step(state)` method for a fixed-shape, cacheable single
  step; this pipeline's decode is repeat-prefill with a growing shape every step, so there is no stable
  program to capture/replay. Fixing this needs a structural change (add real KV-cache + a
  `decode_step`), which the tool's own doc calls "the structural decode lever" — i.e. **the tool
  already knows this is the right next optimization**, it just hasn't been run (blocked by (a)).
- **Non-AR stages (speaker encoder, conditioning encoder, HiFi-GAN vocoder): architecturally
  fixed-shape, single-shot forwards — plausible trace/2CQ candidates in principle** — but this was
  **not independently tested this pass** (the tool's built-in perf adapter only targets the decode
  stage; testing the encoder/vocoder in isolation would need new instrumentation, out of scope for
  this pass's time budget). Flagged honestly as unexplored rather than claimed.

### 3. Completion checklist status (this pass)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Functional e2e producing correct TTS audio | ✅ **met** | `demo_tts.py` wrote a valid 24kHz WAV; `test_e2e_tts` PASSED, waveform_pcc=0.9909 |
| 2 | Working demo | ⚠️ **partial** | the *real* demo (`demo_tts.py`) works; the *canonical* scaffold (`demo.py::test_demo`) still fails — tool never rewired it post-graduation |
| 3 | Perf in range + Tracy-validated | ⚠️ **partial** | real whole-pipeline Tracy numbers + named top-cost ops; but FORWARD_WALL_MS=117.6s for 4 tokens is nowhere near the ~0.3x-RTF GPU bar |
| 4 | Trace + 2CQ support | ✅ **documented precisely** | AR decode: no, with exact tool-diagnosed reason (repeat-prefill, no `decode_step`); non-AR stages: untested, flagged as such |
| 5 | Automatic vs manual documented | ✅ met | §4 below |

### 4. Tool-automatic vs manual intervention (core of the eval)

**Automatic (the tool did this, upstream, unattended):** discovered the 32-component module tree,
classified `REUSE(3)/ADAPT(0)/NEW(29)`; scaffolded 29 ttnn stubs + 29 self-configuring PCC tests;
built the actual **wired end-to-end pipeline** (`tt/pipeline.py`) with a resident on-disk weight-upload
cache (genuinely good, idiomatic ttnn — this is why the pipeline is "everything on device" and
trace-capturable in principle); wrote a perf test (`test_tts_perf.py`) that already wires up Tracy,
2-CQ, and an attempted trace-replay with a **self-diagnosing failure message** when trace can't apply.
That failure message (repeat-prefill / no `decode_step`) is itself a high-quality automatic finding —
the tool correctly named its own next optimization lever without a human debugging it.

**Manual / "additional agentic work" this pass (everything needed to get real numbers on qb2):**
export the fabric-descriptor workaround; verify (not build — it was already present) the 3.12-native
`TTS` dependency stack; discover that `demo.py` is a dead scaffold and that `demo_tts.py` +
`test_e2e_tts.py` are the real artifacts (this required reading `tt/pipeline.py`'s own docstring — not
discoverable from `RUN_REPORT.md` alone); run the full 29-component PCC suite and the e2e test to
completion; run the Tracy profiler correctly (first attempt broke on a `-o`/pytest flag collision,
second attempt hit a stale-sysmem device error requiring `tt-smi -r`, both fixed); aggregate the raw
2.19 GB Tracy CSV + the smaller `ops_perf_results_*.csv` by hand to name top-cost ops (the tool does
not do this aggregation itself — GETTING_STARTED says naming the hottest op needs the Tracy GUI or a
manual post-step); connect the Slice/host-dispatch numbers to the repeat-prefill diagnosis. **None of
the tool's own LLM-driven loop (`up --auto`/`optimize`) ran — still blocked by (a).**

**Net vs the previous pass:** the earlier pass concluded items 1/2/4 were "not met" because it only
looked at the broken scaffold and a partial PCC sample. Pushing through — reading the pipeline source,
running the *actual* e2e test and demo, running the full profiler — flips 1 to met and 3/4 to
evidenced-and-precise instead of "not reached." The tool's generated artifacts were sufficient once a
human (or a working agentic auth) actually explored past the first dead end.

### 5. Seven-criteria scorecard (Model A) — updated

| Criterion | Score /5 | Evidence |
|---|---|---|
| Bring-up Efficiency | 3 | 29 stubs + tests + a genuinely correct wired e2e pipeline scaffolded automatically, with zero edits needed to reach a passing e2e PCC gate. Undercut by 3 real env blockers before any of it could run on a fresh box. |
| Optimization Effectiveness | 1 | `optimize`'s agentic loop never ran (blocker a). The perf test's own trace-replay attempt correctly self-diagnosed the blocking structural issue (no KV-cache) but could not act on it. |
| Final Performance Achievement | 1 | Real Tracy numbers now exist (247.9ms device time / 114ms host time / 117.6s wall for a 4-token truncated decode) but are far (orders of magnitude) from the ~0.3x-RTF GPU bar; AICLK thermally clamped to 800MHz (59% of nominal) on this box, a real caveat on any number here. |
| Optimization Discovery | 4 | Upgraded from 2: the tool's own `TRACE_REPLAY_SKIPPED` message named the exact right lever (add `decode_step`/KV-cache) unprompted, and its e2e gate's torch-fallback enumeration + this pass's Slice/host-dispatch data independently corroborate the same root cause from two directions. |
| Engineering Insight Quality | 4 | Where it ran: PCC 0.95-0.999 across all e2e stages, a real 466.87M-param model producing correct audio, sound REUSE picks. Undercut by the still-uncorrected ASR mis-categorization and the one real mel_spectrogram PCC regression. |
| Workflow Usability | 2 | Upgraded evidence, same score: a new user hits the fabric TT_FATAL, then a false-green `setup_env.sh` auth check that hides blocker (a), then a demo entrypoint that fails outright while the real working demo sits in an unreferenced sibling file. Nothing here is unfixable, but nothing is currently a clean happy path either. |
| Adoption Intent | 3 | Worth adopting as a scaffolder + per-component-verifier + e2e-pipeline-builder once (a) headless auth is real (not just PATH), (b) the env doctor's auth check is fixed to check `.credentials.json`/API key not just file existence, (c) `demo.py` gets rewired to the real pipeline post-graduation, and (d) the structural KV-cache lever it already diagnosed gets a chance to run. |

### 6. Recommendations to the tool authors (new items this pass, in addition to the prior 4)
5. **Fix `setup_env.sh`'s auth check** (`models/experimental/perf_automation/setup_env.sh:109`): it
   treats the mere existence of `~/.claude.json` as "login found," which is created by running
   `claude --version` with zero auth. Check `~/.claude/.credentials.json` or do a live `claude -p`
   probe instead — otherwise the tool tells a new user they're ready when they are not.
6. **Rewire `demo.py` after full graduation.** `scaffold_demo_folder` emits a CPU-only HF-`AutoModel`
   skeleton for models with no template match; once all components graduate and an e2e pipeline
   exists, the canonical demo entrypoint should be repointed at it (or at minimum `RUN_REPORT.md`
   should link to the real demo file instead of leaving the stale scaffold as the only advertised
   entrypoint).
7. **Surface the Tracy op-cost breakdown automatically.** The tool already generates
   `ops_perf_results_*.csv`; a one-line top-10-ops-by-device-time summary in the perf report would
   remove the manual CSV-aggregation step this pass needed to name the hottest ops.
8. **Act on `TRACE_REPLAY_SKIPPED`'s own diagnosis.** The tool already detects "no `decode_step`,
   repeat-prefill" precisely — that's most of the work of the "structural decode lever" it names.
   Wiring that detection to actually attempt the KV-cache rewrite (even just for the AR loop) would
   likely be the single highest-leverage automatic optimization for this model class.

_Environment: qb2, single P150 @ 800 MHz (thermal-clamped, both passes), one 3.12 venv (ttnn +
coqui-tts, already present in the checkout). `TT_MESH_GRAPH_DESC_PATH=…/p150_mesh_graph_descriptor.textproto`.
No files changed in `tt-metal-xtts`; all fixes were environment/invocation-level. Nothing pushed to
any apande-TT/tenstorrent remote (per Moritz's company-repo caution)._

---

## Model B — `tencent/HunyuanImage-3.0` (80B / 13B-active MoE, unified AR multimodal image model)

Bring-up attempted on the **BH Galaxy** (`bh-glx-exp-b03u14`, 32× Blackhole, shared `tt-admin`
account, all work confined to `/home/tt-admin/mthuening`). This is the model the original sizing
gate stopped on (see the standalone `/tmp/hunyuan_report.md`); both blockers that stopped it were
reported cleared and this pass is the actual bring-up attempt. Host: pc (`moritz`), driving the
Galaxy over `ssh galaxy`. Every number below is from a live command I ran on 2026-07-13 (UTC+2 pc)
or the prior session ran on 2026-07-11 — none restated from the sizing report without re-verification.

### 0. Pre-work due-diligence (re-verified live this pass, not assumed)

- **Architecture / size:** Tencent HunyuanImage-3.0, arXiv:2509.23951. **80B total / 13B activated
  per token, 64-expert MoE**, native unified autoregressive multimodal (image generation + understanding
  in one AR framework, "MoE + Transfusion"-style). **Not a DiT** — a genuinely new architecture class
  for tt-metal (existing image-gen ports are pure-diffusion DiT/UNet). The tool's own `plan`
  re-confirmed the size live today: **83.01B params (exact), 168.54 GB bf16 on-disk** — matches the
  vendor/repo figures and the prior sizing research.
- **Closest existing TTNN reference:** the tool's `compat` answers this directly and correctly —
  8/9 building blocks are drop-in from `models/tt_transformers/tt/*` (GQA attention, RMSNorm, token
  embedding, LM head, checkpoint remap, tokenizer, generator/inference loop, top-k sampling); the one
  gap is **MoE routing** (`models/tt_transformers/tt/mixtral_moe.py` exists but hard-codes
  `num_devices=8` and top-2, needs a small refactor for Hunyuan's 64-expert / different-top-k config).
  Repo discovery: **UNKNOWN** — not referenced anywhere in `models/` yet (fresh port). `compat`'s
  overall verdict: **FEASIBLE WITH WORK.**
- **Component split (tool's `scaffold` dry-run, 2026-07-11):** `REUSE 5 / ADAPT 1 / NEW 2` — far
  better than XTTS's 3/0/29. NEW: `image3_decoder_layer`, `top_k_gate` (both ref
  `transformers/src/transformers/models/Hunyuan/modeling_Hunyuan.py`); ADAPT: `mo_e`
  (`mixtral_moe.py`). Caveat below — the scaffold picked the **wrong sibling template**.
- **Hardware fit (re-verified live this pass):** `plan ... --box GalaxyBH` → single-chip TP=1 does
  NOT fit (-137.82 GB headroom); sharded across the canonical **[4,8] mesh (32 chips): FITS
  (comfortable), ~6.19 GB/chip, 22.41 GB headroom.** Tool confidence tag: **LOW** (category-level
  estimate, recommends a smoke-test before deciding) — consistent with, and reconfirming, the prior
  sizing verdict that this genuinely needs the Galaxy, not QB2.
- **Realistic perf target:** not establishable this pass — no device run (see §1). Published GPU
  numbers for this model are qualitative ("multi-GPU inference recommended"); a tt-metal perf target
  needs a reference GPU measurement, not yet taken.
- **Expected bottleneck (architectural):** 64-expert MoE routing at a 13B/80B ≈ 16% activation ratio
  stresses CCL + expert-placement across the mesh; the AR decode loop (KV-cache) combined with
  image-token generation is a novel control-flow shape vs tt-metal's existing pure-AR-LLM or
  pure-DiT families.

### 1. The blockers (two real, one external — re-verified, not the prior "cleared" claim)

**(d) Customer card occupancy — NOT actually cleared; re-verification caught it. STOPPED card work.**
The `galaxy-usage-check.md` snapshot that "cleared" blocker #2 concluded "all 32 cards free, 0 device
fds" from `ls -l /proc/*/fd | grep -c tenstorrent` = 0. **That check is misleading:** it misses
mmap-only holds. I re-ran `sudo lsof /dev/tenstorrent/*` myself before touching any card and it shows
customer process **`user1` PID 42168** (`uvicorn main:app --port 8000`, parked, sleeping, ~2d16h
elapsed) **still mmaps ALL 32 devices** (`mem` CHR mappings on /dev/tenstorrent/0..31). ASICs are
idle (AICLK 1350 = idle clock, low power) but the devices are **held**. Same blocker as the Jul 11
session. Per the hard safety rule (device held by someone else → stop, don't touch, Telegram Moritz),
**no card-touching command was run this pass.** Telegrammed Moritz at ~15:25 pc with the discrepancy.
The fd-count check should be replaced with `sudo lsof /dev/tenstorrent/*` (or a UMD device-lock probe)
— it false-negatives exactly the parked-mmap case that matters here.

**(e) `auto-up` isolation worktree + the 13:22 failure — REVISED this pass (last turn's
"isolation-venv bug" framing was incomplete).** `auto-up` locks `--isolation=worktree`; it creates a
fresh git worktree under `/tmp/`, and `python_env/` is gitignored (`.gitignore:18`) so the worktree
has no `python_env/` dir (confirmed: the leftover `/tmp/tt_hw_planner_tencent_HunyuanImage-3.0_1783948950`
has none). **But** the 13:22 `auto-up` failure (`transformers is not importable` / `huggingface_hub
not installed`) was PRIMARILY the PATH-shadowing footgun in (f), not the missing worktree venv: the
PROGRESS.md launch command did `export PATH="$HOME/.tenstorrent-venv/bin:$PATH"` *before*
`python -m scripts.tt_hw_planner auto-up …`, so `python` resolved to the tt-smi venv python (no
transformers/huggingface_hub) regardless of the worktree. Whether the isolation worktree would also
break when launched with the **correct** python is **not verified** this pass — `auto-up` opens
devices (blocked by (d)) so I could not run it to check. If the tool uses `sys.executable` (the
absolute `python_env/bin/python` path) for its worktree subprocess, the venv site-packages would
still resolve (venv python references site-packages by absolute path in `pyvenv.cfg`), and the
missing `python_env/` dir in the worktree would be harmless. Latent concern, downgraded from
"confirmed bug." **Workaround if it does bite:** `up --isolation none --auto` (in-place, real venv)
— `up` exposes `--isolation {worktree,none}` even though `auto-up` doesn't.

**(f) `huggingface_hub not installed` — ROOT-CAUSED this pass: a PATH-ordering footgun, not a tool
code bug.** `probe.py:314` exits on `ImportError` from `from huggingface_hub import HfApi`. By
patching `probe.py` to print `sys.executable` + `sys.path` in the failing process, I captured:
`sys.executable=/home/tt-admin/.tenstorrent-venv/bin/python` and
`sys.path=[…, /home/tt-admin/.tenstorrent-venv/lib/python3.10/site-packages]` — i.e. the **tt-smi
venv** python, not the tt-metal `python_env` python (no huggingface_hub/transformers/ttnn there).
Cause: GETTING_STARTED.md step 7 literally says `export PATH="$HOME/.tenstorrent-venv/bin:$PATH"`
(**prepend**) to expose `tt-smi`; but the tt-smi venv's `bin/python` then **shadows** the tt-metal
`python_env/bin/python` that `source python_env/bin/activate` had put first. So `python -m
scripts.tt_hw_planner …` runs under the wrong python. Last turn's "intermittent" appearance was
because my heredocs did *not* prepend `.tenstorrent-venv` (so `python` = `python_env` python →
worked) while my `python -m` CLI runs *did* prepend it (→ failed). **Fix (verified):** append
instead of prepend — `export PATH="$PATH:$HOME/.tenstorrent-venv/bin"`. With the fix, `which python`
= `python_env/bin/python`, `plan` passes **3/3 deterministically**, and `scaffold --apply` runs past
the probe (model loading, RAM climbing 139→156 GB). This unblocked `scaffold --apply` (now running
in the background, nohup'd on galaxy, survives this turn). This is a **documentation footgun**, not
a tool code defect — but it stopped the 13:22 `auto-up` and every CLI plan/scaffold run until found,
so it is the single most impactful "blocker" of the pass.

### 2. What actually ran (own numbers, this pass + verified prior session)

**`plan` — ran live today (after the intermittent failures cleared once):**
```
Total parameters: 83.01 B (exact)   On-disk weights: 168.54 GB (bf16 mixed)
GalaxyBH bf16  per-chip=6.19G  usable=28.60G  headroom=22.41G  FITS (comfortable)
RECOMMENDATION: GalaxyBH, [4,8] mesh, 32× Blackhole (BHGLX), 1024 GB total
CONFIDENCE: LOW  (category-level estimates only — smoke-test before deciding)
```
Re-confirms the sizing verdict on live hardware today, matching the Jul 11 run and the independent
sizing research. No new claim beyond what the tool itself reports.

**`compat` — ran 2026-07-11 (with `--skip-kernel-check`, see blocker g):** architecture correctly
identified as MoE, GQA-attention + MoE-routing decoder, 8/9 blocks drop-in from `tt_transformers`,
1 partial (MoE routing). Verdict **FEASIBLE WITH WORK**. Repo discovery **UNKNOWN** (not ported).

**`scaffold` (dry-run, 2026-07-11) — component plan produced, but wrong sibling template (see §4):**
`5 REUSE / 1 ADAPT / 2 NEW`; NEW stubs `image3_decoder_layer`, `top_k_gate`; ADAPT `mo_e`. Proposed
to copy ~50 files from `models/demos/vision/generative/stable_diffusion/wormhole/` (UNet/VAE/diffusion
scheduler plumbing). `--apply` was blocked on 2026-07-13 by the PATH footgun (f) until I root-caused
it; with the fix, `scaffold --apply` gets past the probe and loads the 83B model on CPU. **Not yet
completed this pass:** the `--apply` run does a ~30+-min CPU model-analysis pass (it instantiates the
full 83B reference model, ~160 GB RAM, then an op-classification walk — same as the Jul 11 dry-run
which ran 30+ min past model-load); my first relaunched attempt was timeout-killed at 20 min (no
files written yet). Relaunched with a 50-min timeout, nohup'd + setsid-detached on galaxy (survives
this turn — next relaunch reads `scaffold_apply5.log` + the written `hunyuanimage_3_0/` tree +
`BRING_UP_PLAN.md` + the two NEW stubs to evaluate the tool's actual stub-generation quality).

**Device bring-up (`auto-up`/`up --execute`/per-component PCC/demo/perf): NOT RUN** — blocked by (d).
No device fd was opened by this pass at any point (verified: `sudo lsof` unchanged, only customer
PID 42168 throughout).

### 3. Model Completion Checklist — status

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Hardware selection justified by size/architecture/existing references | ✅ **met** | live `plan` today + `compat` block analysis + independent sizing research (this §0) |
| 2 | Functional end-to-end run | ❌ not started | blocked by (d) customer cards (device iterate-loop can't run) |
| 3 | Correct outputs on real inputs | ❌ not started | blocked by (d) |
| 4 | Working demo | ❌ not started | blocked by (d); `scaffold --apply` past the probe after (f) root-caused but the ~30+-min CPU pass not yet completed (timeout-killed once, relaunched) |
| 5 | Perf in acceptable range | ❌ not started | blocked by (d) |
| 6 | Tracy-validated perf + manual op-cost analysis | ❌ not started | blocked by (d) |
| 7 | Trace support | ❌ not started | blocked by (d) |
| 8 | 2CQ support | ❌ not started | blocked by (d) |
| 9 | Documented model-specific blockers | ✅ **met** | (d) customer occupancy + (e) isolation-worktree latent concern + (f) PATH footgun root-caused + (g) compat top_k-list crash + (h) scaffold template mismatch (this §) |

### 4. Tool-automatic vs manual intervention (core of the eval)

**Automatic (the tool did this, unattended, no-device):** `plan` produced an exact 83.01B-param sizing
+ a mesh-fit verdict + an honest LOW confidence tag; `compat` correctly identified the architecture
(MoE / GQA-attention + MoE-routing decoder), mapped 8/9 building blocks to existing
`tt_transformers` modules, flagged the one partial (MoE routing hardcoding top-2/8-devices), and
returned a correct FEASIBLE-WITH-WORK verdict + UNKNOWN repo-discovery; `scaffold` (dry-run) emitted
a full component plan (5/1/2) and a file manifest. The `plan`+`compat` sizing/feasibility analysis is
genuinely good and independently corroborated — that is the tool's strongest automatic output for
this model.

**Manual / "additional agentic work" this pass:** re-verify occupancy myself before touching cards —
which caught that the "cards free" conclusion was based on a misleading fd-count check (`sudo lsof`
showed the customer still mmaps all 32); **root-cause the `huggingface_hub not installed` failures
to a PATH-ordering footgun** (GETTING_STARTED.md step 7's `export PATH="$HOME/.tenstorrent-venv/bin:$PATH"`
prepends the tt-smi venv, whose `bin/python` shadows the tt-metal `python_env/bin/python` — captured
`sys.executable=.tenstorrent-venv/bin/python` in the failing process; fix: append instead of prepend;
verified `plan` 3/3 and `scaffold --apply` runs with the fix); **revise the prior "isolation-venv
bug" diagnosis** (the 13:22 `auto-up` failure was primarily the same PATH shadowing, not the missing
`python_env/` in the worktree — downgraded (e) to a latent, unverified concern); root-cause the
`compat` `int(top_k)` crash on Hunyuan's list-valued `num_experts_per_tok`
(`kernel_constraints.py:~499`, workaround `--skip-kernel-check`); reproduce the `plan` sizing live
today; and identify that **`scaffold`'s sibling-template pick contradicts `compat`** — `compat`
correctly sees an AR MoE decoder (8/9 from `tt_transformers`) but `scaffold` keys off
`pipeline_tag=text-to-image` / category `Image` and picks **Stable Diffusion 1.4** (a UNet/VAE
diffusion model), proposing to copy ~50 irrelevant UNet/VAE/scheduler files. For a multimodal AR-MoE
model the right sibling is `tt_transformers`, not `stable_diffusion`. **None of the tool's
LLM-driven loop (`auto-up`/`up --auto`/`optimize`/`emit-e2e`) ran** — blocked by (d) (devices held
by customer).

### 5. Seven-criteria scorecard (Model B) — honest partial (device stages blocked)

| Criterion | Score /5 | Evidence |
|---|---|---|
| Bring-up Efficiency | 3 | `plan`+`compat`+`scaffold` no-device analysis ran and is genuinely useful, and once the PATH footgun (f) is fixed (one-line `export PATH` ordering change) the no-device stages run cleanly and deterministically (`plan` 3/3, `scaffold --apply` **completed** turn-3 rc=0 ~35 min and wrote a real tree+plan+stubs, §Scaffold output quality). The device iterate-loop never ran (customer, d). Undercut because (a) a user following GETTING_STARTED.md literally hits (f) at the first `python -m` command, and (b) the auto-generated demo tree is an irrelevant SD1.4 copy for this AR-MoE model — the useful output is the component plan, not the scaffolded files, and the pass is a single-shot ~50-min CPU run with no checkpointing. |
| Optimization Effectiveness | N/A | not reached (device stages blocked). |
| Final Performance Achievement | N/A | not reached. |
| Optimization Discovery | N/A | not reached (`optimize` never ran). |
| Engineering Insight Quality | 3 | `plan` sizing correct and matches independent research; `compat` architecture ID + 8/9 block mapping correct and useful. The real `scaffold --apply` output (§Scaffold output quality) **confirms** the component classifier is correct — 5 REUSE (`tt_transformers`/common) + 1 ADAPT (`mixtral_moe.py`, accurate hard-coded-num_devices/top-2 note) + 2 NEW (the MoE `top_k_gate` + `decoder_layer`), matching `compat`'s MoE-gap verdict, with real shape extraction for REUSE rows. Undercut by `scaffold`'s *file-scaffolding* layer picking the wrong template (SD1.4 for an AR-MoE model — confirmed in the real `--apply` tree, not just dry-run, §4) and the `compat` top_k-list crash (g): the plan and the scaffolded files disagree with each other. |
| Workflow Usability | 2 | The PATH footgun (f) is a documentation defect — GETTING_STARTED.md step 7's `export PATH="$HOME/.tenstorrent-venv/bin:$PATH"` (prepend) shadows `python` with the tt-smi venv and silently breaks every `python -m` invocation; it stopped the 13:22 `auto-up` and all my CLI runs until root-caused. Plus the `compat` top_k-list crash (g), the scaffold/compat template disagreement (h), and the fd-count occupancy check that false-negatives the parked-mmap case and misled even the pre-pass state file. One genuine improvement over the qb2 pass: headless `claude` auth works on galaxy (`setup_env.sh` auth check is a true positive here — unlike qb2's false positive on `~/.claude.json`). |
| Adoption Intent | 2 | The no-device `plan`/`compat` stages are worth adopting as a sizing/feasibility triage tool. For actual bring-up, a user following the doc literally hits the PATH footgun (f) at the first command, the `compat` top_k-list crash (g) at `compat`, and a wrong-template `scaffold` (h) — too many doc/tool defects before any device work, on top of the external customer-card blocker. |

### 6. Recommendations to the tool authors (Hunyuan-specific, in addition to Model A's 1–8)

9. **Fix GETTING_STARTED.md step 7's `PATH` instruction.** It says
   `export PATH="$HOME/.tenstorrent-venv/bin:$PATH"` (**prepend**), which makes the tt-smi venv's
   `bin/python` shadow the tt-metal `python_env/bin/python` that `source python_env/bin/activate`
   just set — so every later `python -m scripts.tt_hw_planner …` runs under the wrong python (no
   transformers/ttnn/huggingface_hub) and fails with `transformers is not importable` /
   `huggingface_hub not installed`. This stopped the 13:22 `auto-up` and every CLI run this pass
   until root-caused. Change it to **append** (`export PATH="$PATH:$HOME/.tenstorrent-venv/bin"`) or
   expose `tt-smi` via an alias / full path. Verified fix: with append, `which python` =
   `python_env/bin/python`, `plan` 3/3, `scaffold --apply` runs.
10. **(Downgraded from last turn's "isolation-venv bug".) `auto-up --isolation=worktree` + missing
    `python_env/` in the `/tmp` worktree is a *latent* concern, not a confirmed blocker.** The
    13:22 `auto-up` failure was primarily (9), not this. If the tool uses `sys.executable` (absolute
    `python_env/bin/python`) for its worktree subprocess, the venv site-packages still resolve via
    `pyvenv.cfg` and the missing `python_env/` dir is harmless. Not verified this pass (`auto-up`
    opens devices, blocked by the customer). If it does bite, `up --isolation none --auto` is the
    workaround (`up` exposes `--isolation {worktree,none}`; `auto-up` doesn't).
11. **Fix `compat`'s `check_moe`** (`kernel_constraints.py:~499`) to handle `num_experts_per_tok` as
    a **list** (per-layer top_k), not `int(top_k)` assuming a scalar — it crashes with `TypeError`
    on Hunyuan's config, forcing `--skip-kernel-check` and losing the kernel-constraint signal.
12. **Reconcile `scaffold`'s sibling-template pick with `compat`'s block analysis.** For
    HunyuanImage-3.0, `compat` correctly IDs an AR MoE decoder (8/9 from `tt_transformers`) but
    `scaffold` picks Stable Diffusion 1.4 (a UNet/VAE diffusion model) because it keys off
    `pipeline_tag=text-to-image` / category `Image`, copying ~50 irrelevant diffusion files. A
    multimodal AR model needs the `tt_transformers` sibling. The two stages disagreeing on the
    closest template is a real port-misdirection risk for any image-tagged AR model.
13. **Replace the card-occupancy fd-count check with `sudo lsof /dev/tenstorrent/*`** (or a UMD
    device-lock probe). `ls -l /proc/*/fd | grep -c tenstorrent` false-negatives the parked-mmap
    case (a process holding devices via `mmap`, no fd) — exactly the case that matters on a shared
    box with a parked customer server. This is technically outside the tool, but it's the check the
    fleet relied on and it misled the "blockers cleared" call for this very task.

#### Scaffold output quality (harvested)

The `scaffold --apply` launched turn-3 (`run_scaffold_once.sh`, fixed PATH append, `timeout 3000`)
**completed cleanly** — `scaffold_apply5.log` ends `end Mon Jul 13 02:39:47 PM UTC 2026  rc=0` /
`=== SCAFFOLD --apply SUCCEEDED ===`, ~35 min wall (16:04→16:39 pc). It wrote the full demo tree
under `models/demos/vision/generative/hunyuanimage_3_0/` plus `BRING_UP_PLAN.md`,
`bringup_status.json`, and two `_stubs/` files. So the tool *can* finish a single-shot CPU pass —
but only with an unbounded/≥3000 s timeout: the turn-1 launch used `timeout 1200` and was killed
at 20 min (rc=124) before writing anything, and the CPU model-analysis pass alone runs 30+ min
past model-load. There is **no checkpointing**: a timeout-kill loses everything and forces a full
restart (re-download/re-load ~40 GB of weights, re-run analysis). That is a real robustness note
for the tool authors — a ~50-min single-shot CPU pass with no resume is fragile on a shared box.

The SD1.4 sibling-template mismatch (rec 12) is **confirmed in the real `--apply` output**, not
just the dry-run, and it did **not** self-correct. `BRING_UP_PLAN.md` opens with "Backend template:
**Stable Diffusion 1.4** at `models/demos/vision/generative/stable_diffusion`" and the whole
`wormhole/tt/` tree is a verbatim copy of the SD demo — `ttnn_functional_unet_2d_condition_model_*`,
`_basic_transformer_block`, `_cross_attention`, `_resnetblock2d`, a full `vae/` subdir
(`ttnn_vae_decoder`/`_midblock`/`_upblock`/`_attention`/…), `sd_helper_funcs.py`,
`sd_pndm_scheduler.py`, a streamlit `web_demo/`. The `wormhole/README.md` is literally titled
"`# Stable_diffusion` … a latent text-to-image diffusion model". Only the directory path was
renamed to `hunyuanimage_3_0`; the contents are architecturally alien to HunyuanImage-3.0's
autoregressive-MoE decoder. An engineer following the plan's own checklist step 1 ("import the
sibling tt-module directly in the scaffolded demo's `tt/`") would find no `tt/` file corresponding
to any AR-MoE component — the scaffolded demo tree is dead weight for this model and would mostly
be discarded.

**However**, the component-classification layer inside `scaffold` correctly identified the AR-MoE
architecture, and this is the genuinely useful part of the output. `bringup_status.json` reports
`5 REUSE · 1 ADAPT · 2 NEW`:

- **REUSE** (5): `self_attention`, `mlp`, `m_l_p`, `r_m_s_norm`, `image3_s_d_p_a_attention` — all
  mapped to `models/tt_transformers/tt/attention.py` / `mlp.py` or `models/common/rmsnorm.py`,
  matching `compat`'s "8/9 from `tt_transformers`" verdict. Shape extraction worked for these
  (`hidden_size=4096`, `num_attention_heads=32`, `num_key_value_heads=8`, `vocab_size=133120`,
  `max_position_embeddings=22800`).
- **ADAPT** (1): `mo_e` → `models/tt_transformers/tt/mixtral_moe.py`, with the accurate note that
  it "hard-codes `num_devices=8` and top-2" and needs a small refactor for Hunyuan's
  64-expert / different-top-k config — exactly the MoE gap `compat` flagged.
- **NEW** (2): `image3_decoder_layer` and `top_k_gate` (`HunyuanTopKGate`, `submodule_path=
  layers.0.mlp.gate`) — precisely the AR-MoE-specific pieces with no analog in `tt_transformers`.

So `scaffold` contains **two disagreeing layers**: the backend-template picker keys off
`pipeline_tag=text-to-image` → SD1.4 → copies the entire SD demo tree; the component classifier
correctly sees an AR-MoE decoder → `tt_transformers` siblings. The plan and the scaffolded files
contradict each other for this model.

The two **NEW stubs** (`_stubs/image3_decoder_layer.py`, `_stubs/top_k_gate.py`) are pure
boilerplate: a docstring pointing at the HF reference
(`transformers/src/transformers/models/Hunyuan/modeling_Hunyuan.py`) plus a function that
`raise NotImplementedError`. Zero TTNN code, no shape constants (`new_shape: {}` for both — the
supplemental module-tree pass that classified them did not extract shapes, unlike the REUSE
rows), no skeleton of the gate / decoder-layer logic. They are signposts, not starting points; a
competent engineer would work from the HF reference and the `mixtral_moe.py` ADAPT target rather
than from these stubs.

Two smaller defects worth noting: (i) the per-component shape diff is empty across the board
(`sibling_shape: {}` for every component) because "Sibling config could not be fetched" — the
`CompVis/stable-diffusion-v1-4` fetch failed (HF_TOKEN / network), so no real numeric
sibling-vs-new comparison happened; the REUSE verdicts come from the class-name `reuse_registry`,
not a shape diff. The plan itself flags this and suggests setting `HF_TOKEN` / pre-downloading
the sibling. (ii) Component names are munged from class names — `HunyuanMoE`→`mo_e`,
`HunyuanMLP`→`m_l_p`, `HunyuanRMSNorm`→`r_m_s_norm`, `HunyuanImage3SDPAAttention`→
`image3_s_d_p_a_attention` — which would leak into file/import names if used verbatim.

**Net assessment:** as a *component-level bring-up plan + reuse map*, the `--apply` output is
solid (correct AR-MoE ID, accurate MoE-ADAPT note, correct `tt_transformers` reuse targets, real
shape extraction for classified components). As a *working starting scaffold* it is low-value for
this model: the auto-generated demo tree is an irrelevant SD diffusion copy that must largely be
discarded, and the NEW stubs are boilerplate-only. The useful artifact is `BRING_UP_PLAN.md` /
`bringup_status.json`, not the `tt/` files.

_Environment: BH Galaxy `bh-glx-exp-b03u14`, 32× Blackhole, shared `tt-admin` account, all work in
`/home/tt-admin/mthuening` (checkout `tt-metal-hunyuan`, branch `hunyuan-bringup` — one local
tool-improvement commit on top of `feature/tt-hw-planner`, NOT pushed). Env built 2026-07-11: venv
py3.10.19, `import ttnn` ok, transformers 5.10.2, `tt-smi` sees 32 devices, headless `claude` auth
verified. No device fd opened by this pass (customer PID 42168 held all 32 throughout). No files
pushed to any apande-TT/tenstorrent remote (per Moritz's company-repo caution). Telegrammed Moritz
in real time on the occupancy discrepancy._

### 7. Device bring-up retry (2026-07-15)

**Outcome: still blocked — no device work possible this pass.** Two days after the 2026-07-13
~15:25 check, I re-verified Galaxy occupancy myself before touching any card (per the hard safety
rule; the cached `galaxy-usage-check.md` snapshot was explicitly not trusted). The cards are **still
held by the same customer process**, so no card-touching command was run. This is the honest
outcome — no device-stage progress was fabricated.

**Fresh occupancy evidence (commands I ran live on `galaxy` at ~01:20 pc, 2026-07-15):**
- `sudo lsof /dev/tenstorrent/*` — **customer `user1` PID 42168 still mmaps ALL 32 devices**
  (`mem` CHR mappings on `/dev/tenstorrent/0..31`, plus open fds 29u–102u on every device). This is
  the check that matters (mmap holds show up here even when fd-count heuristics say "free"), and it
  is identical to the Jul 11 and Jul 13 findings — the blocker (d) never cleared.
- `ps -p 42168` — confirms it is the same parked server:
  `/home/user1/tt-metal/python_env/bin/python3 .../uvicorn main:app --lifespan on --port 8000`,
  state `Sl+` (sleeping), **elapsed 4d02h32m** (started ~Jul 11, i.e. it has been parked and holding
  the devices across this entire bring-up attempt, including both prior passes).
- `who; w` — **0 users logged in**, load average `0.00, 0.05, 0.03` (box idle) — but the devices are
  held by the backgrounded customer process regardless of who's logged in.
- `tt-smi -ls` (with `.tenstorrent-venv/bin` appended to PATH) — enumerates all 32 Blackhole devices,
  UMD chip IDs 0–31, series `tt-galaxy-…`; confirms the hardware is present and visible, just not
  releasable by us. (`tt-smi -s` returned no telemetry output this run; not needed for the gate —
  `lsof` is decisive and the box is demonstrably idle.)

**What this means for the Completion Checklist (items 2–8): unchanged, still ❌ blocked-by-(d).**
Every device stage — `auto-up` / `up --auto` iterate-loop, per-component PCC, demo, perf, Tracy,
trace, 2CQ — remains **NOT RUN**, exactly as in §3. No new device numbers exist because no device fd
was opened by this pass at any point (verified: the only holder in `lsof` throughout was customer
PID 42168). The no-device artifacts from the prior pass (§"Scaffold output quality": the
`scaffold --apply` tree, `BRING_UP_PLAN.md`, the two NEW stubs) are still the high-water mark; the
PATH footgun (f) fix and headless `claude` auth remain in place on galaxy, so the moment the
customer process releases the devices the bring-up can resume cleanly from where `scaffold --apply`
left off (`/home/tt-admin/mthuening/tt-metal-hunyuan`, branch `hunyuan-bringup`) via the tool's
`auto-up` / `up --isolation none --auto` flow with `export PATH="$PATH:$HOME/.tenstorrent-venv/bin"`
(append, not prepend).

**Action taken this pass:** Telegrammed Moritz at ~01:20 pc with the finding and the recommendation
that the next step requires the customer `user1` process (PID 42168) to be stopped/released — that
is Moritz's call, not something to do unilaterally on a shared customer box. No card was reset, no
device opened, nothing pushed to any `apande-TT`/`tenstorrent` remote. Re-verification timestamp
logged here so the next relaunch does not re-trust the stale "free" snapshot.

---

## Model C — `ACE-Step/Ace-Step1.5` (text-to-music, composite pipeline)

This is a **documentation-merge pass**, not a new device run. ACE-Step 1.5 was already evaluated on
qb2 on **2026-07-13** (workstream `tt-hw-planner-acestep`, concluded). That session hit a real,
well-documented tool wall, but the finding was captured only in
`~/.coworker/knowledgebase/memory/_global/tt-hw-planner-composite-model-gap.md` and never merged into
this running deliverable. This section merges it in, at the same depth/format as Models A/B, restated
faithfully from the verified artifacts of that concluded session (the worker log + the composite-model-gap
memory) — no new commands run, no new numbers fabricated, no device touched (correctly: this is a
no-card, read-only documentation task, and the structural blocker below means a device run was never
reachable anyway).

### 0. Pre-work due-diligence (verified on HuggingFace `ACE-Step/Ace-Step1.5`, 2026-07-13)

- **Architecture / size (measured, not estimated):** ACE-Step 1.5 is a **text-to-music** model (not
  TTS — the distinction matters and is exactly what the tool got wrong, see §1). **5.01B params /
  10.03 GB bf16.** It is a **composite, custom-code** repo (`trust_remote_code`, `model_type=acestep`)
  made of **four sub-models, each in its own HF subfolder**, loaded by a bespoke `ACEStepPipeline`
  with **no single model at the repo root**:
  - **DiT music generator** — `acestep-v15-turbo/` (the diffusion-transformer core; no existing
    tt-metal reference).
  - **DCAE VAE** — `vae/` (the audio codec/decoder).
  - **Qwen3 LM-planner** — `acestep-5Hz-lm-1.7B/` (the lyric/structure planner).
  - **Qwen3 embedding encoder** — `Qwen3-Embedding-0.6B/` (text conditioning).
- **Why it is structurally different from A/B:** Models A (XTTS-v2) and B (HunyuanImage-3.0) are each
  a single checkpoint loadable by one `from_pretrained` at the repo root (XTTS via its coqui runtime,
  Hunyuan via HF `AutoModel`). ACE-Step is **not** — it is a multi-submodel pipeline orchestrated by a
  hand-written `ACEStepPipeline` class with custom code (`configuration_acestep_v15.py`, etc.) living
  **inside subfolders**, not at the root. `tt_hw_planner` characterizes a model purely from its
  top-level `config.json` + `pipeline_tag` and assumes one root `from_pretrained` builds the whole
  thing; ACE-Step breaks that assumption in two places at once (multi-submodel + custom-code-in-subfolder).
- **Closest existing TTNN reference:** none for the DiT/VAE music core. The two Qwen3 sub-models are
  plain transformer-decoder LLMs and would map to `models/tt_transformers/` — but the tool never got
  far enough to split them out cleanly (see §1).
- **Hardware fit (re-verified by `plan` that session):** qb2 single-chip **FITS comfortably —
  3.51 GB/chip, ~25 GB headroom** — confirming Dalar's "single card is sufficient" claim. This is the
  one stage the tool got fully right.
- **Realistic perf target:** not establishable — no device run ever happened (blocked upstream at
  `scaffold`, see §1). No fabricated target.
- **Expected bottleneck (architectural, had a run happened):** the DiT music generator + DCAE VAE
  are the novel compute core (audio-rate diffusion + codec decode); the Qwen3 planner is a standard
  AR-LLM decode loop. None of this was reachable to profile.

### 1. What ran and what broke (this IS the finding — not "not attempted")

The static, no-device, no-auth stages (`plan` → `compat` → `scaffold`) ran clean on qb2 on 2026-07-13
and **are the eval signal**. Bring-up (`auto-up` / `emit-e2e` / `optimize`) was **not reached** —
blocked by two independent walls. No device run, no perf numbers, nothing fabricated.

**`plan` — PASS (no device / no auth):** sized correctly **5.01B / 10.03 GB**; **QB2 fits
comfortably, 3.51 GB/chip, ~25 GB headroom**, confirming the single-card claim. `CONFIDENCE: LOW`
(category-level estimate). **One real defect:** it mis-categorized the model as **TTS** off
`pipeline_tag=text-to-audio` and suggested a `qwen3_tts` sibling template — the model is text-to-**music**,
and this mis-categorization propagated downstream (see `scaffold`).

**`compat` — PASS, but blind to the music core (the confident-wrong answer):** reported
**"10 ready / 1 partial / 0 missing — FEASIBLE WITH WORK."** That verdict is **false confidence**:
`compat` mapped only the **transformer-decoder-shaped config fields** (GQA, RoPE, RMSNorm, SwiGLU,
vocab 64003, sliding_window 128) to `tt_transformers` LLM blocks and **silently dropped every
audio-specific field** from consideration — `in_channels: 192`, `fsq_dim: 2048`, `patch_size: 2`,
the lyric/timbre encoders, and `audio_decoder`. It **never opened the DiT/VAE subfolders**. It does
not know those fields exist, so it cannot flag them as missing — hence the misleading "0 missing."
Its kernel-constraint flags were real but **incomplete** (derived from the LM fields only): top-k
vocab-not-power-of-2, sliding + chunked-prefill, TP fails only at TP=32.

**`scaffold` — the wall, then a silent wrong degradation:** hard-failed with
`MODEL FAILED TO LOAD — OSError: … configuration_acestep_v15.py` — the custom code lives in a
subfolder, so `from_pretrained` at the repo root cannot build the model. **Instead of hard-stopping
here, the tool silently degraded**: it fell back to the TTS template its `plan` had mis-picked, whose
base `demo.py` was **missing on disk**, and emitted a **2-REUSE / 0-NEW** plan (just attention + mlp)
that **omits the entire diffusion/codec core**. A confident wrong answer on top of a failed load,
not a stop.

### 2. The blockers (two real, one upstream of the other)

**(g) Structural — composite pipeline incompatible with the tool's single-`from_pretrained`-at-root
assumption. UPSTREAM of auth.** ACE-Step's 4-sub-model custom-code layout cannot be constructed on
CPU by a single root `from_pretrained`, so `capture-inputs` and every PCC-gated stage are blocked
**even with valid auth**. This is a tool feature-gap: there is no composite / multi-submodel concept
and no subfolder-targeting. This is the durable finding (see the cross-model recommendation in §5/§6).

**(a) Access / auth — same as Model A's blocker (a).** The tool's agentic stages spawn `claude -p`
on the device host; qb2 has no claude CLI, no `~/.claude/.credentials.json`, no `ANTHROPIC_API_KEY`
→ "Not logged in." So `up --auto` / `optimize`'s repair loop cannot run headless. Telegrammed
Moritz that session. Secondary to (g) here: even with auth, the structural wall blocks the path
first. (Also noted: the tool's inner loop is hardwired to Anthropic auth — GLM cannot drive it.)

**(f) PATH-shadow `huggingface_hub` trap — same as Model B's blocker (f).** Documented in
`tt-hw-planner-venv-path-shadow.md`; it persisted on qb2 this session. Not the cause of the ACE-Step
stop (the structural (g) is), but a recurring usability footgun across all three models.

### 3. Model Completion Checklist — status

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Hardware selection justified by size/architecture/existing references | ✅ **met** | live `plan` 2026-07-13: 5.01B/10.03GB, QB2 fits 3.51 GB/chip ~25 GB headroom, CONFIDENCE LOW; matches Dalar's single-card claim |
| 2 | Functional end-to-end run | ❌ **N/A — tool structural gap** | `scaffold` hard-failed (`OSError: configuration_acestep_v15.py`); model cannot be built on CPU by a single root `from_pretrained`, so the whole PCC-gated path is blocked upstream of auth |
| 3 | Correct outputs on real inputs | ❌ **not reached** | blocked upstream by (g) |
| 4 | Working demo | ❌ **N/A — tool structural gap** | `scaffold` degraded to a TTS template whose base `demo.py` is missing on disk; emitted 2-REUSE/0-NEW plan omitting the diffusion/codec core |
| 5 | Perf in acceptable range | ❌ **not reached** | no device run |
| 6 | Tracy-validated perf + manual op-cost analysis | ❌ **not reached** | no device run |
| 7 | Trace support | ❌ **not reached** | no device run |
| 8 | 2CQ support | ❌ **not reached** | no device run |
| 9 | Documented model-specific blockers | ✅ **met** | (g) composite-pipeline structural gap + (a) headless-claude auth + (f) PATH-shadow trap, all root-caused with evidence (§1, §2) |

### 4. Tool-automatic vs manual intervention (core of the eval)

**Automatic (the tool did this, no-device, no-auth):** `plan` produced a correct 5.01B sizing + a
QB2 fit verdict + an honest LOW confidence tag. `compat` ran and produced a structured block map +
kernel-constraint list. `scaffold` ran (and failed, then degraded). The sizing stage is genuinely
good and independently consistent with Dalar's single-card claim — that is the tool's strongest
automatic output for this model, same as for Hunyuan.

**Manual / "additional agentic work" that session:** the *eval* value here is almost entirely in
**not trusting `compat`'s "0 missing / FEASIBLE" verdict** — a human had to open the HF repo, notice
the four subfolders each with their own config, recognize the bespoke `ACEStepPipeline`, and connect
that to the `scaffold` `OSError` to diagnose the structural gap the tool itself never names. The tool
reported confident success (`FEASIBLE WITH WORK`) on a model it had silently half-characterized; the
human supplied the architecture knowledge the tool dropped (DiT + VAE + 2× Qwen3) and the
"this-is-a-composite-pipeline" framing. **None of the tool's LLM-driven loop (`auto-up` / `optimize`)
ran** — blocked by (a), and upstream of that by (g).

**Net:** for a plain single-`from_pretrained` HF model, the tool's static stages are useful (Models
A/B show this). For a composite custom-code pipeline, the tool's automatic output is **confidently
incomplete** — it reports the subset it understood as the whole. The manual work this session was to
catch that and name it as a feature-gap rather than repeat the tool's verdict.

### 5. Seven-criteria scorecard (Model C) — honest, optimization stages not reached

| Criterion | Score /5 | Evidence |
|---|---|---|
| Bring-up Efficiency | N/A (blocked) | The static path (`plan`/`compat`) ran fast, but `scaffold` hit a hard model-load wall (g) before any bring-up could start. The tool cannot bring this model up as-is — not a tuning problem, a structural one. |
| Optimization Effectiveness | N/A | Not reached — `optimize`'s agentic loop never ran (blocked by (g) upstream, then (a)). No numbers, none fabricated. |
| Final Performance Achievement | N/A | Not reached — no device run, no perf numbers. |
| Optimization Discovery | N/A | Not reached. (`compat`'s kernel-constraint list is real but derived from the LM fields only — incomplete, blind to the DiT/VAE core.) |
| Engineering Insight Quality | 2 | `plan` sizing is correct and useful (5.01B, QB2 fit, LOW confidence honest). But `compat`'s insight is **confidently wrong**: it maps only the decoder-shaped fields, drops the entire music/diffusion/codec core, and reports "0 missing / 0 NEW / FEASIBLE" — i.e. it presents the subset it understood as the whole. The *eval's* insight in surfacing this failure mode is the valuable part (see §6 rec 14): a confident wrong answer is worse than a hard stop, and naming that clearly is the single most important cross-model recommendation from this whole evaluation. |
| Workflow Usability | 2 | Clean CLI and followable flow, and `plan`/`compat` run with no auth. But `scaffold` emits a **confident wrong plan on top of a failed load** (degrades to a missing-on-disk TTS template, 2-REUSE/0-NEW) instead of hard-stopping — the worst usability failure mode, because a user who didn't open the repo by hand would trust it. Plus the recurring PATH-shadow trap (f) and the headless-auth blocker (a) shared with A/B. |
| Adoption Intent | 1 (for this model class, as-is) | Static `plan`/`compat` are useful for standard single-`from_pretrained` HF models. For a composite pipeline model the tool cannot drive bring-up end-to-end: it can't load the model, can't see the music core, and scaffolds the wrong family while reporting success. Needs the composite-pipeline feature (rec 14) before it's adoptable for this class. |

### 6. Recommendations to the tool authors (ACE-Step-specific, in addition to Models A's 1–8 and B's 9–13)

14. **Detect composite / multi-submodel / custom-pipeline-class repos and HARD-STOP with a clear
    message instead of silently reporting false confidence.** This is the single most important
    cross-model recommendation from the whole eval: **a confident wrong answer is worse than a hard
    stop.** For ACE-Step 1.5, `compat` reported "10 ready / 1 partial / **0 missing** / FEASIBLE WITH
    WORK" by silently characterizing only the transformer-decoder-shaped fields it understood and
    never opening the DiT/VAE subfolders; `scaffold` then hard-failed on the custom code and
    **silently degraded** to a wrong TTS template (missing on disk) emitting a 2-REUSE/0-NEW plan that
    omits the entire diffusion/codec core. A user who didn't hand-inspect the HF repo would ship a
    bring-up plan for the wrong model family. Concrete detection signals the tool could use, all
    available before any model load:
    - **Multiple subfolders each with their own `config.json`** at the repo root (ACE-Step has
      `acestep-v15-turbo/`, `vae/`, `acestep-5Hz-lm-1.7B/`, `Qwen3-Embedding-0.6B/`).
    - A **custom `*Pipeline` class** (here `ACEStepPipeline`) that does not match any single HF
      auto-class, and/or `model_type` / `trust_remote_code` pointing at hand-written code.
    - Custom `.py` modules (`configuration_acestep_v15.py`, …) living **inside a subfolder**, not at
      the repo root — the exact thing that caused the `scaffold` `OSError`.
    On any of these, `compat`/`scaffold` should emit **"composite multi-submodel pipeline detected —
    not supported yet; characterize each sub-model separately or target a subfolder explicitly"** and
    stop — not map the one sub-config it can parse and report FEASIBLE.

15. **Don't degrade to a sibling template on a model-load failure.** `scaffold`'s fallback path
    (load fails → pick a template by `pipeline_tag` → emit a plan) produced a confident 2-REUSE/0-NEW
    plan on top of a failed load, with a base demo missing on disk. A load failure should be a hard
    stop with the captured `OSError`, not a silent template substitution — the substitution hides the
    real blocker (g) behind a plausible-looking but wrong artifact.

16. **Fix `plan`'s `pipeline_tag` → category mapping for `text-to-audio`.** It mapped
    `text-to-audio` → TTS → `qwen3_tts` template, but ACE-Step is text-to-**music** (a diffusion +
    codec pipeline, not a TTS vocoder). The mis-categorization is what seeded the wrong `scaffold`
    fallback. `text-to-audio` should not collapse to TTS by default; music-generation repos need a
    distinct category/template path (or the composite-detection in rec 14 catches it first).

_Environment: qb2 (tt-quietbox2), single Blackhole P150, physical card 1,
`TT_VISIBLE_DEVICES=1`, session 2026-07-13 (workstream `tt-hw-planner-acestep`, concluded). Static
stages only — `plan` / `compat` / `scaffold` — no device fd opened, no perf numbers taken (none
fabricated). This section is a 2026-07-15 documentation-merge of that concluded session's verified
findings into the running deliverable; no new commands run, no card touched. Findings also recorded
in `~/.coworker/knowledgebase/memory/_global/tt-hw-planner-composite-model-gap.md`. Report from the
original session pushed to `origin/wk/tt-hw-planner-acestep` (`workstreams/tt-hw-planner-acestep.md`)._

---

## Model D — `meituan-longcat/LongCat-Video` (13.6B video DiT, text-to-video composite)

Evaluated live on **qb1 (tt-quietbox)**, 4× Blackhole P150 (UMD chip IDs 0–3), physical card 0,
`TT_VISIBLE_DEVICES=0`, on **2026-07-15**. Every number below is from a live command I ran this pass
against the tool's own PR branch (`apande-TT/tt-metal` `feature/tt-hw-planner`, PR #46283) — none
restated from prior rounds.

> **Host note (qb1 vs the task's "qb2" line):** the task brief's WORKSPACE + CARD sections point at
> qb1 (`ssh ttuser@tt-quietbox`, worktree `/home/ttuser/.coworker/wt/tt-hw-planner-longcat-video`,
> card 0); only the stale HARDWARE line says qb2. qb1 has a free 4× P150 mesh (verified `tt-smi -ls`),
> equivalent to a QB2-class box, so the device fit verdict transfers. I flagged the discrepancy to
> Moritz in real time and ran on qb1 per the operational (worktree + CARD) instructions.

### 0. Pre-work due-diligence (verified by hand on the real HF repo, then by the tool)

- **Architecture (measured from the live HF repo tree, not assumed):** `meituan-longcat/LongCat-Video`
  is a **diffusion-transformer text-to-video model**, 13.6B params (Meituan's headline figure, the DiT
  core; arXiv:2510.22200). It is a **composite pipeline** with the standard diffusers *folder layout* —
  `dit/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`, `lora/` — each subfolder holding its own
  weights + `config.json`. The three real components:
  - **DiT** — `_class_name: LongCatVideoTransformer3DModel` (custom, not stock diffusers 0.32),
    `hidden_size=4096`, `depth=48`, `num_heads=32`, `in/out_channels=16`, `patch_size=[1,2,2]`,
    `caption_channels=4096` (fed by the T5), **Block Sparse Attention** (`bsa_params.sparsity=0.9375`,
    `chunk_3d_shape_q/k=[4,4,4]`). 6 safetensors shards, **~54.3 GB on disk (fp32)** → 13.6B params.
  - **VAE** — `AutoencoderKLWan` (Wan-style causal 3D VAE), `z_dim=16`, `base_dim=96`, ~0.5 GB.
  - **Text encoder** — `UMT5EncoderModel` (`google/umt5-xxl`), `d_model=4096`, 24 layers, 64 heads,
    vocab 256384, `torch_dtype=float32`, 5 shards ~22.7 GB (fp32) → ~5.7B params.
  - plus two LoRA adapters (`cfg_step_lora` 2.5 GB, `refinement_lora` 3.2 GB) under `lora/`.
- **The structural catch (why the tool breaks):** the repo uses the diffusers *folder layout* but its
  root `config.json` **and** `model_index.json` are both just `{"model_name": "LongCat-Video"}` — **not
  a valid diffusers pipeline manifest** (no `_class_name`, no per-component registry). So
  `DiffusionPipeline.from_pretrained` / `AutoModel.from_pretrained` at the repo root **cannot assemble
  the pipeline**; the reference repo loads it via a bespoke `run_demo_text_to_video.py` (`torchrun
  --checkpoint_dir=…`), not a HF auto-class. This is the discriminating case between "diffusers-native
  composite" (one `from_pretrained` works) and "bespoke composite" (it doesn't) — LongCat is the latter,
  same family as ACE-Step despite looking diffusers-native at a glance.
- **Closest existing TTNN reference:** none for the LongCat DiT / Wan VAE. The UMT5 text encoder is a
  plain transformer encoder and would map to `models/tt_transformers/`. There are existing
  `models/tt_dit/` and Wan-VAE references in-tree that a real port would reuse — the tool never got far
  enough to name them (see §1).
- **Hardware fit (tool's `plan`, live this pass):** single P150 **does NOT fit**; the **QB2 4-chip mesh
  FITS** — directly contradicting Saurabh's "should fit a single P150 card" sizing claim. Real `plan`
  numbers:
  ```
  On-disk: 83.3 GB   Parameters: 41.64 B (bf16-assumed)   Category: Video (text-to-video)
  QB2 bf16  per-chip=84.3 GB usable=29.2 GB headroom=-55.1 GB  → no            (single chip)
  QB2 bf16  per-chip=21.8 GB usable=28.6 GB headroom=+6.8 GB   → FITS (room)   (4-chip mesh)
  CONFIDENCE: LOW
  ```
  Telegrammed Moritz the discrepancy in real time per the brief's "if `plan` disagrees with the
  should-fit claim, report the real numbers before assuming which is right." **One real defect in
  `plan`:** it assumed bf16 and computed `Parameters = 41.64 B` from `on-disk/2`, but the DiT and T5
  weights are actually **fp32 on disk** (`text_encoder/config.json` says `torch_dtype: float32`; DiT
  shards are ~9.9 GB×6 = 54 GB for 13.6B params = 4 bytes/param). So the 41.64 B param figure is
  inflated ~2×; real total is ~19–20 B (13.6B DiT + ~5.7B T5 + VAE). The **fit verdict is still
  directionally correct** because it is driven by on-disk bytes (single chip: 84 GB ≫ 29 GB usable),
  but the param count is wrong for any fp32-stored repo — a sizing-trust defect worth noting.
- **Realistic perf target:** not establishable — no device run (blocked upstream at `scaffold`, §1).
  Published claim is "720p/30fps in minutes" on GPU; a tt-metal target needs a reference measurement.
- **Expected bottleneck (architectural, had a run happened):** the 48-layer DiT with Block Sparse
  Attention at high resolution (patchify → 3D attention over spacetime tokens) is the compute core;
  the Wan VAE decode (causal 3D conv) and the umt5-xxl text encode (run-once) are secondary. None
  reachable to profile.

### 1. What ran and what broke (the finding — confirmed composite wall, distinct mechanism)

The static no-device stages ran live on qb1 and **are the eval signal**. Bring-up (`auto-up` /
`emit-e2e` / `optimize`) was **not reached** — blocked upstream by the structural wall (g) and then
auth (a). No device fd opened, no perf numbers taken (none fabricated).

**`plan` — PASS (no device / no auth), with one real defect.** Sized the model from HF without
loading weights: **83.3 GB on-disk**, **Category = Video (text-to-video)** — the categorization is
**correct** this time (unlike ACE-Step's `text-to-audio`→TTS mis-categorization, Model C §1).
Verdict: single-chip **no fit**, 4-chip mesh **FITS (+6.8 GB headroom)**, `CONFIDENCE: LOW`. The one
defect: it assumed bf16 and reported **41.64 B params** (`on-disk/2`) for a repo whose weights are
fp32 on disk → ~2× overcount (see §0). The fit verdict is still right (byte-driven); the param figure
is not.

**`compat` — ran, but confidently mapped the WRONG architecture family (composite-wall, variant B).**
Because the root `config.json` is `{"model_name":"LongCat-Video"}` with **no `model_type`**, `compat`
returned:
```
architecture_family = "unknown (no model_type)"
closest_supported_model = null
overall = "UNKNOWN"
effort_summary = ""
```
So far more honest than ACE-Step's false "FEASIBLE WITH WORK / 0 missing." **But** `compat` then
emitted a full **27-block generic LLM-decoder template** and marked standard LLM blocks as
`needed:true / SUPPORTED / drop-in`: `Token embedding`, `Standard RoPE`, `RMSNorm (text)`,
`SwiGLU MLP`, `LM head`, `Generator / inference loop (prefill+decode with KV cache)`, `Top-k / sampling`.
**None of those match a video diffusion pipeline** — a video DiT has no LM head, no KV-cache
generator, no top-k sampling; it has a diffusion scheduler, a patchify+positional-embedding front,
Block Sparse Attention, and a VAE decoder. `compat` **never opened the `dit/`, `vae/`, or
`text_encoder/` subfolders** (the `text_encoder/config.json` it would have needed to correctly
characterize the UMT5 encoder sits right there in the repo, with a clean `model_type: umt5`). Its
`kernel_constraints.findings_by_tp` came back **empty for every TP** (`{1:[], 2:[], 4:[], 8:[], 32:[]}`)
— no kernel signal at all, because it had no architecture to derive constraints from. So: an honest
`UNKNOWN` headline, wrapped around a confidently-irrelevant LLM block list. Better than ACE-Step's
confident-wrong "FEASIBLE," but still not a characterization a user could port from.

**`scaffold` — HARD-FAIL on load, then the SAME silent-wrong degradation as ACE-Step (rec 15).** This
is the smoking gun, and it is essentially identical to Model C's `scaffold` failure:
```
========================================================================
  MODEL FAILED TO LOAD — cannot inspect 'meituan-longcat/LongCat-Video'
========================================================================
  reason: ValueError: Unrecognized model in meituan-longcat/LongCat-Video.
          Should have a `model_type` key in its config.json.
========================================================================
SCAFFOLDING meituan-longcat/LongCat-Video  (compat=FAMILY TEMPLATE (Video))
  Sibling:        hf_eager universal (Video)
  Compat note:    Component plan: 2 REUSE / 0 NEW
  Skipped: - backend demo path missing on disk: models/demos/hf_eager/demo.py
  [REUSE] self_attention  models/tt_transformers/tt/attention.py
  [REUSE] mlp             models/tt_transformers/tt/mlp.py
```
i.e. `AutoModel.from_pretrained` at the root **cannot construct the model** (no `model_type` →
`ValueError`), the tool correctly reports "could not even construct the model on CPU" — and then
**instead of hard-stopping, silently degrades** to an `hf_eager universal (Video)` sibling template
whose base `demo.py` is **missing on disk**, emitting a **2 REUSE / 0 NEW** plan (`self_attention` +
`mlp` from `tt_transformers`) that **omits the entire DiT, the VAE, the T5 text encoder, the diffusion
scheduler, Block Sparse Attention, and the patchify/positional-embedding layers**. A confident,
plausible-looking plan for an LLM decoder, on top of a failed load of a video diffusion pipeline.
Exactly the failure mode rec 15 (Model C) named: a load failure should be a hard stop, not a silent
template substitution that hides the real blocker behind a wrong artifact.

**`auto-up` / `prepare --execute` / `emit-e2e` / `optimize` — NOT REACHED.** Upstream of auth: the
model cannot be constructed on CPU, so `capture-inputs` and the entire PCC-gated iterate loop are
blocked **even with valid auth** (same as ACE-Step, Model C §2 blocker g). Auth (a) is also only
partially present on qb1 — `~/.claude/.credentials.json` exists but the `claude` CLI is not on `PATH`
(`which claude` empty), so the tool's `claude -p` subprocess would still fail. Moot here: the
structural wall blocks first.

### 2. The blockers (one structural, one auth, one sizing-trust)

**(g) Structural — composite pipeline with a non-standard root manifest; tool assumes one root
`from_pretrained` builds everything. UPSTREAM of auth, same family as Model C (g).** LongCat's root
`config.json`/`model_index.json` carry no `model_type` and no diffusers component registry, so no
single root `from_pretrained` can build it; the real loader is bespoke (`run_demo_text_to_video.py`).
`compat` can't see the architecture (returns `UNKNOWN`), never opens subfolders; `scaffold` hard-fails
on the load and silently degrades. This is **confirming evidence for the composite-model-gap finding**
(rec 14) on a **second, structurally different model class** (video DiT, vs ACE-Step's audio music
DiT) — the wall is not audio-specific; it is any composite/custom-loader repo where the root config
isn't a single auto-class manifest.

**(a) Auth — partial on qb1.** `~/.claude/.credentials.json` exists (unlike qb2/galaxy), but the
`claude` CLI binary is not on `PATH`, so the tool's `claude -p` agentic subprocesses (`auto-up` /
`optimize` repair loop) cannot run headless here. Secondary to (g): even with the CLI on PATH, the
structural wall blocks the path first. (Also: the tool's inner loop is hardwired to Anthropic auth;
this GLM-driven eval cannot drive the LLM stages regardless.)

**(i) `plan` param-overcount for fp32 repos — NEW this pass.** `plan` assumed bf16 and reported
`41.64 B` params for a repo whose weights are fp32 on disk (~2× over the real ~19–20 B). The fit
verdict (byte-driven) is still correct, but the param figure is not trustworthy when the repo stores
fp32. The tool should read each component's `torch_dtype` (and/or the safetensors index dtype) before
computing param count.

### 3. Model Completion Checklist — status

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Hardware selection justified by size/architecture/existing references | ✅ **met** | live `plan` 2026-07-15: 83.3 GB on-disk, single-chip no-fit (-55.1 GB), QB2 4-chip FITS (+6.8 GB), LOW confidence; contradicts Saurabh's single-P150 claim (telegrammed); hand-verified repo tree confirms DiT+VAE+T5 composite |
| 2 | Functional end-to-end run | ❌ **N/A — tool structural gap (g)** | `scaffold` hard-failed (`ValueError: … no model_type key`); model cannot be constructed on CPU by a root `from_pretrained`, so the PCC-gated path is blocked upstream of auth |
| 3 | Correct outputs on real inputs | ❌ **not reached** | blocked upstream by (g) |
| 4 | Working demo | ❌ **N/A — tool structural gap (g)** | `scaffold` degraded to `hf_eager universal (Video)` template whose base `demo.py` is missing on disk; emitted 2-REUSE/0-NEW LLM plan omitting the entire DiT/VAE/T5/scheduler |
| 5 | Perf in acceptable range | ❌ **not reached** | no device run |
| 6 | Tracy-validated perf + manual op-cost analysis | ❌ **not reached** | no device run |
| 7 | Trace support | ❌ **not reached** | no device run |
| 8 | 2CQ support | ❌ **not reached** | no device run |
| 9 | Documented model-specific blockers | ✅ **met** | (g) composite/non-standard-manifest structural gap + (a) partial claude-CLI auth on qb1 + (i) plan fp32 param-overcount, all root-caused with live evidence (§0, §1, §2) |

### 4. Tool-automatic vs manual intervention (core of the eval)

**Automatic (the tool did this, no-device, no-auth):** `plan` produced a correct on-disk sizing (83.3
GB), a **correct category** (Video / text-to-video — better than ACE-Step), and an honest single-chip
no-fit / 4-chip-fit verdict with a LOW confidence tag. `compat` ran and returned an honest
`architecture_family = "unknown (no model_type)"` / `overall = "UNKNOWN"` headline. `scaffold` ran,
correctly diagnosed "MODEL FAILED TO LOAD … could not even construct the model on CPU," and produced a
structured plan + `BRING_UP_PLAN.md`/`bringup_status.json` manifest. The `plan` sizing + category call
is the tool's strongest automatic output for this model.

**Manual / "additional agentic work" this pass:** the eval value is again in **not trusting the tool's
artifact at face value** — a human had to open the HF repo, see the diffusers folder layout, notice
that `model_index.json` is **not** a valid diffusers manifest (`{"model_name": …}` only), recognize
the bespoke `run_demo_text_to_video.py` loader, and connect that to the `scaffold` `ValueError` to
diagnose the composite wall the tool itself only half-names (it says "failed to load," not "this is a
composite pipeline with a non-standard manifest — target each subfolder"). The human also caught
`plan`'s fp32 param-overcount (§2 blocker i) — the tool reported 41.64 B confidently. And the human
supplied the architecture knowledge the tool dropped (DiT + Wan VAE + umt5-xxl T5 + BSA + scheduler)
and the "this-is-a-diffusion-pipeline-not-an-LLM-decoder" framing that makes the 27-block LLM template
in `compat` visibly wrong. **None of the tool's LLM-driven loop (`auto-up` / `optimize`) ran** —
blocked by (g) upstream, then (a). Env-wise: the tool's canonical `create_venv.sh` path needs a full
tt-metal C++ build (submodules + `build_metal.sh`, multi-hour); I instead built a py3.10 venv with the
tool's public PyPI pins (`torch==2.11.0+cpu`, `transformers==5.10.2`, `diffusers==0.33.0`) and reused
the host's existing built `ttnn` from `/home/ttuser/tt-metal` via `PYTHONPATH` + `LD_LIBRARY_PATH` to
run the no-device stages — an env-workaround a new external user would also need, since `create_venv.sh`
is a full-source-build path, not a quick install.

**Net:** `plan`/`compat`/`scaffold` all ran and produced *real* output, so this pass is further along
than "tool didn't run" — but the bring-up path is structurally closed at `scaffold` for the same root
cause as ACE-Step. The confirming finding: **the composite-model gap is not audio-specific; it hits
video diffusion too**, with a distinct tell (`UNKNOWN` headline + a generic LLM block list + a
silent-degraded 2-REUSE scaffold, vs ACE-Step's `FEASIBLE` + TTS template).

### 5. Seven-criteria scorecard (Model D) — honest, optimization stages not reached

| Criterion | Score /5 | Evidence |
|---|---|---|
| Bring-up Efficiency | 2 | `plan` ran clean and sized correctly (83.3 GB, correct Video category, honest single-vs-4-chip fit verdict). But `compat` returned `UNKNOWN` + a generic 27-block LLM-decoder template (no subfolder inspection), and `scaffold` hard-failed on load then silently degraded to a 2-REUSE/0-NEW LLM plan omitting the entire diffusion core — bring-up cannot start from the tool's output without a human re-characterizing the repo by hand. |
| Optimization Effectiveness | N/A | Not reached — `optimize`'s agentic loop never ran (blocked by (g) upstream, then (a)). No numbers, none fabricated. |
| Final Performance Achievement | N/A | Not reached — no device run, no perf numbers. |
| Optimization Discovery | N/A | Not reached. (`compat`'s `kernel_constraints.findings_by_tp` is empty for every TP — no kernel signal; `plan`'s LOW confidence is honest.) |
| Engineering Insight Quality | 2 | `plan` sizing + category are correct and useful (and the LOW confidence is honest). But `compat`'s insight is **misdirected**: an `UNKNOWN` headline wrapped around a confidently-irrelevant LLM block list — it never opens `dit/`/`vae/`/`text_encoder/`, so it cannot name the real components (DiT/VAE/T5/BSA/scheduler) a port would need. `scaffold`'s "failed to load" message is accurate but does not name the composite/non-standard-manifest root cause. The eval's insight (catching the fp32 param-overcount and the composite-wall tell) is the valuable part. |
| Workflow Usability | 2 | Clean CLI and `plan`/`compat`/`scaffold` run with no auth. But `scaffold` repeats the worst failure mode from Model C — a confident wrong plan (2-REUSE LLM template, sibling demo missing on disk) on top of a failed load, instead of a hard stop. Plus the canonical env is a full-source-build path (`create_venv.sh` + `build_metal.sh`), not a quick install, so reaching a runnable state on a fresh box is heavy. |
| Adoption Intent | 1 (for this model class, as-is) | Static `plan` is useful for sizing/category on any model. For a composite/custom-loader video pipeline the tool cannot drive bring-up: it can't load the model (non-standard manifest, no `model_type`), can't see the DiT/VAE/T5 core, and scaffolds an LLM-decoder family while reporting success. Needs the composite-pipeline feature (rec 14) — detect a non-standard `model_index.json` / missing `model_type` and hard-stop with "target each subfolder explicitly" — before it's adoptable for this class. |

### 6. Recommendations to the tool authors (LongCat-specific, in addition to Models A 1–8, B 9–13, C 14–16)

17. **Treat a non-standard `model_index.json` / missing root `model_type` as a composite-pipeline
    signal and HARD-STOP (extends rec 14).** LongCat's root `config.json` and `model_index.json` are
    both just `{"model_name": "LongCat-Video"}` — a diffusers repo by tag/folder-layout but **not** a
    valid diffusers pipeline manifest (no `_class_name`, no component registry). The tool already
    detects "no `model_type`" (`compat` returns `unknown (no model_type)`; `scaffold` raises
    `ValueError: … Should have a model_type key`) — that detection should route to **"composite /
    custom-loader pipeline detected — not supported yet; characterize each subfolder (`dit/`, `vae/`,
    `text_encoder/`) separately"** and stop, not emit a generic 27-block LLM template and a 2-REUSE
    scaffold. Concrete pre-load signals, all available without downloading weights: root
    `model_index.json` lacking `_class_name`/component entries; multiple subfolders each with their own
    `config.json` + `_class_name`; a `pipeline_tag` whose reference repo loads via a bespoke script
    rather than an HF auto-class.

18. **Do not emit an LLM-decoder block template for a non-LLM architecture.** When
    `architecture_family = "unknown"`, `compat` still marked `Token embedding / RoPE / RMSNorm /
    SwiGLU / LM head / KV-cache Generator / Top-k` as `needed:true/SUPPORTED/drop-in` — a generic LLM
    fallback that is architecturally wrong for a video diffusion pipeline (no LM head, no KV-cache
    decode, no top-k sampling; it has a scheduler + patchify + BSA + VAE). An `UNKNOWN` architecture
    should produce an `UNKNOWN` block list (or none), not a confident LLM one.

19. **Fix `plan`'s parameter count for fp32-stored repos (new, blocker i).** `plan` assumed bf16 and
    reported `41.64 B` params (`on-disk/2`) for LongCat, whose DiT and T5 weights are fp32 on disk
    (`text_encoder/config.json: torch_dtype=float32`; DiT shards ~9.9 GB×6 = 54 GB for 13.6 B params).
    Real total is ~19–20 B. The fit verdict is byte-driven and still correct, but the param figure is
    ~2× wrong. Read each component's `torch_dtype` (and/or the safetensors index dtype) before dividing
    bytes by the bytes-per-param assumption.

20. **Confirm rec 15 (no silent degradation on load failure) applies to the Video path too.**
    `scaffold`'s Video fallback (`hf_eager universal (Video)`, base `demo.py` missing on disk, 2-REUSE
    plan) is the same silent-wrong-degradation failure mode as ACE-Step's TTS fallback — a second
    independent instance. A model-load failure should hard-stop with the captured `ValueError`, not
    substitute a template that hides the real (composite) blocker behind a plausible but wrong plan.

_Environment: qb1 (tt-quietbox), 4× Blackhole P150 (UMD chip IDs 0–3), physical card 0,
`TT_VISIBLE_DEVICES=0`, session 2026-07-15 (workstream `tt-hw-planner-longcat-video`). Static stages
only — `plan` / `compat` / `scaffold` — no device fd opened, no perf numbers taken (none fabricated).
Tool checkout: isolated shallow clone of `apande-TT/tt-metal` `feature/tt-hw-planner` (PR #46283) under
the task worktree `/home/ttuser/.coworker/wt/tt-hw-planner-longcat-video/tt-metal`; env = py3.10 venv
with the tool's public PyPI pins (`torch==2.11.0+cpu`, `transformers==5.10.2`, `diffusers==0.33.0`)
reusing the host's pre-built `ttnn` from `/home/ttuser/tt-metal` via `PYTHONPATH`+`LD_LIBRARY_PATH`
(canonical `create_venv.sh` is a multi-hour full-source build, not used). Raw `plan`/`compat`/`scaffold`
outputs saved in the worktree (`compat.json`, `scaffold.out`, `scaffold.err`). No files pushed to any
`apande-TT`/`tenstorrent` remote (per Moritz's company-repo caution); task worktree commits on
`wk/tt-hw-planner-longcat-video` of `moritztng/tt-bio`. Sizing discrepancy vs Saurabh's single-P150
claim telegrammed to Moritz in real time._
