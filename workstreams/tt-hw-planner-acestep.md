# ACE-Step 1.5 — Bring-up & Optimization (tt_hw_planner evaluation)

**Status: STOPPED at `capture-inputs` wall. Two independent blockers, one structural (model won't load on CPU as the tool assumes) and one access (tool's agentic `claude` subprocess has no auth on qb2). Static analysis stages (`plan`/`compat`/`scaffold`) ran clean, no device, no auth — and are themselves the most useful eval signal this round.**
**Date: 2026-07-13. Requestor: Dalar Vartanians (via Moritz). Hardware: qb2 (tt-quietbox2), single Blackhole P150, card 1. Tool: tt-metal PR #46283, fork apande-TT, branch `feature/tt-hw-planner`.**

## Target model — ACE-Step 1.5 (verified from HuggingFace `ACE-Step/Ace-Step1.5`)
A **text-to-music** model, NOT text-to-speech. Composite, custom-code (`trust_remote_code`), `model_type="acestep"`, `architectures=["AceStepConditionGenerationModel"]`. Four sub-models, each in its own subfolder with its own config:
- `acestep-v15-turbo/` — the **Diffusion Transformer (DiT)**, the actual music generator. Custom code (`modeling_acestep_v15_turbo.py`, `configuration_acestep_v15.py`), `silence_latent.pt`. This is the compute core and has no existing tt-metal reference.
- `vae/` — the **DCAE** (Sana Deep-Compression AutoEncoder) audio latent decoder.
- `acestep-5Hz-lm-1.7B/` — the **LM planner** (Qwen3-based, 1.7B) that turns a query into a song blueprint.
- `Qwen3-Embedding-0.6B/` — text/lyric encoder.

Total 5.01 B params, 10.03 GB on disk (bf16). Runs in <4 GB VRAM per vendor; **a single qb2 card (16 GB) is more than sufficient** — Dalar's single-chip claim is correct (confirmed below by the tool's own fit-check).

The official load path is a bespoke `ACEStepPipeline` that loads each subfolder separately — there is **no modeling file and no single loadable model at the repo root** (root has only `.gitattributes`, `README.md`, `config.json`).

## What ran, literally, following GETTING_STARTED

Environment: reused the already-built XTTS checkout `~/tt-metal-xtts` on qb2 (same `feature/tt-hw-planner` HEAD `23e613b493`, tt-metal already compiled, Python 3.12 venv already provisioned from the XTTS round) rather than burning hours on a fresh from-source build for a tool-eval. Applied the known PATH-shadow fix from the Hunyuan round (append, not prepend, `~/.tenstorrent-venv/bin`) — `plan`/`compat`/`scaffold` ran deterministically with no `huggingface_hub` error.

### `plan ACE-Step/Ace-Step1.5` — PASS (no device, no auth, ~seconds)
Read the config correctly: **5.01 B params exact, 10.03 GB bf16**. Fit-check verdict per box:

| Box | per-chip | headroom | verdict |
|---|---|---|---|
| N150 | 11.03 G | -0.13 G | no |
| N300 | 6.01 G | 4.49 G | FITS (comfortable) |
| **QB2** | **3.51 G** | **25.09 G** | **FITS (comfortable)** |
| Galaxy / GalaxyBH | 1.31 G | ~9–27 G | FITS |

Recommendation: N300 (smallest sufficient). **QB2 fits comfortably** — matches Dalar. `CONFIDENCE: LOW — category-level estimates only`.
- **Finding (repeat of XTTS):** mis-categorized `Category: TTS` off `pipeline_tag=text-to-audio`, and points at `models/demos/qwen3_tts/` as the closest template. ACE-Step is a diffusion music model with a DiT + DCAE codec, architecturally nothing like a Qwen3 TTS decoder. The category is derived purely from the HF pipeline tag, not from the architecture.

### `compat ACE-Step/Ace-Step1.5` — PASS (no device, no auth) — but blind to the music core
`Architecture: unknown (acestep)`, `Overall verdict: FEASIBLE WITH WORK`, `Repo discovery: UNKNOWN (not yet ported)`. Section 1 reports **10 ready / 1 partial / 0 missing**:
token-embedding, GQA attention, RoPE, RMSNorm, SwiGLU MLP, LM head, sampling — all mapped to `models/tt_transformers/tt/*`. Kernel constraints flagged: `ttnn.topk` vocab 64003 not power-of-2 (single-core fallback), sliding-window+chunked-prefill incompatibility, `ttnn.embedding` tile-padding. TP divisibility OK for TP=1/2/4/8, fails TP=32 (16 heads / 8 KV).

- **Key finding:** `compat` is **transformer-decoder-centric**. The top-level `config.json` mixes decoder fields (`num_hidden_layers:24, hidden_size:2048, num_attention_heads:16, num_key_value_heads:8, sliding_window:128, vocab_size:64003`) with audio-specific fields (`in_channels:192, audio_acoustic_hidden_dim:64, fsq_dim:2048, patch_size:2, num_lyric_encoder_hidden_layers:8, num_timbre_encoder_hidden_layers:4, num_audio_decoder_hidden_layers:24, num_attention_pooler_hidden_layers:2`). The tool mapped **only the decoder-shaped fields** to tt_transformers LLM blocks and **silently dropped every audio-specific field**. It never inspects the sub-model folders. So it characterizes a 5B multimodal music generator as if it were a plain Qwen3 LLM — "10 ready / 0 missing" is true for the LM-shaped attention but omits the DiT denoiser, DCAE decoder, FSQ quantizer, patchify, and the lyric/timbre encoders entirely. An engineer trusting `compat` would badly underestimate the port.

### `scaffold ACE-Step/Ace-Step1.5` (dry-run) — the wall
**`MODEL FAILED TO LOAD — cannot inspect`**:
```
OSError: ACE-Step/Ace-Step1.5 does not appear to have a file named
configuration_acestep_v15.py.
```
The tool builds the model on CPU via a single `from_pretrained` at the repo root. ACE-Step's custom-code files live in the `acestep-v15-turbo/` subfolder, not root, so `trust_remote_code` can't resolve them and the model can't be constructed. **This blocks `capture-inputs` (run the HF model once with forward hooks to get real per-component I/O), which the entire PCC-verification / auto-iterate pipeline depends on.** It is independent of, and upstream of, the auth blocker.

Despite the load failure, `scaffold` proceeded with a degraded plan and produced a plainly-wrong result:
- `compat=FAMILY TEMPLATE (TTS)`, sibling `hf_eager universal (TTS)` — and even that sibling's base is **missing on disk**: `backend demo path missing on disk: models/demos/hf_eager/demo.py`.
- Component plan: **2 REUSE / 0 ADAPT / 0 NEW** — just `attention` and `mlp`, both pointed at `tt_transformers`. Zero NEW components surfaced for a diffusion music model. The DiT, VAE, encoders, FSQ — none appear.

## Blockers

1. **STRUCTURAL (upstream, not auth-related):** ACE-Step 1.5's composite repo layout — four sub-models, per-subfolder custom code, loaded by a bespoke `ACEStepPipeline` — is incompatible with the tool's "one HF model, one `from_pretrained` at repo root" assumption. The model can't be constructed on CPU, so `capture-inputs` and every PCC-gated stage downstream cannot run **even with valid auth**. The tool's own error text suggests a `cpu_compat.py` stand-in, but this is not a missing-accelerator-package case — it's a multi-model repo the tool has no concept of. Bringing up ACE-Step through this tool would require either (a) targeting each sub-model as its own repo id (the DiT, the VAE, the LM, the encoder separately — the tool has no subfolder syntax today), or (b) teaching the tool the composite/pipeline model shape. Both are tool-feature gaps, not user error.
2. **ACCESS (auth):** the agentic stages (`auto-up`, `emit-e2e`, `optimize`) spawn `claude -p` as a subprocess **on qb2**, which has no `claude` CLI and no creds / `ANTHROPIC_API_KEY` (pc holds the fleet's OAuth, qb2 does not). Same as XTTS blocker-a. Telegrammed Moritz for either (a) OK to install claude CLI on qb2 + copy the OAuth creds, or (b) an `ANTHROPIC_API_KEY` to export there. Pending his call — but note that even once cleared, blocker 1 stops the run at `capture-inputs`.

No bring-up run was attempted on device (the pipeline can't reach the device stage). No fabricated perf numbers.

## 7-criteria tool evaluation (honest — most stages not reached; scored only where evidenced)

| Criterion | Score | Evidence |
|---|---|---|
| **Bring-up Efficiency** | **N/A (blocked)** | Never reached on-device bring-up. Static path was fast (`plan`+`compat`+`scaffold` in minutes, no device, no auth), but `scaffold` hit a hard model-load wall. The tool cannot bring up this model as-is without a feature change for composite/multi-submodel repos. |
| **Optimization Effectiveness** | **N/A** | Not reached. |
| **Final Performance Achievement** | **N/A** | Not reached. No perf numbers — none fabricated. |
| **Optimization Discovery** | **N/A** | Not reached. (`compat`'s kernel-constraint list — topk power-of-2, sliding-window/chunked-prefill, TP divisibility — is the kind of pre-run insight it would feed the optimizer, but it's derived from the LM fields only, so it's incomplete for this model.) |
| **Engineering Insight Quality** | **2 / 5 (partial, mixed)** | Genuinely good: exact param/size accounting, per-box fit-check with per-chip dispatch/ccl/frag overhead modelled, TP-divisibility table, real kernel constraints (topk vocab, sliding-window). Genuinely misleading here: the whole analysis is transformer-decoder-centric — it reads only the decoder-shaped fields of a multimodal config, silently drops all audio/diffusion fields, never opens the sub-model folders, and reports "0 missing / 2 REUSE / 0 NEW" for a model whose entire novel core (DiT + DCAE + FSQ + encoders) it never saw. High-confidence-looking output that is wrong for composite models. |
| **Workflow Usability** | **3 / 5** | CLI is clean and the GETTING_STARTED flow is followable literally. Stop-and-tell-the-fix messages are good (the load-failure box explains what failed and why). Real friction: (a) the PATH-shadow `huggingface_hub` trap still present (worked around with the known append fix); (b) the agentic loop's dependence on a `claude` subprocess with ambient auth is a hard headless blocker on a device host that isn't the auth host; (c) `scaffold` produced confident output (template pick, component plan, next-steps) on top of a failed model load and a missing sibling base file — it should have hard-stopped, not emitted a plausible-looking wrong plan. |
| **Adoption Intent** | **Low for this model class, as-is** | For a standard single-`from_pretrained` HF model (the XTTS/LLM shape), the static planning stages are useful and worth keeping. For a composite custom-code pipeline model like ACE-Step 1.5, the tool cannot currently drive bring-up end-to-end — it can't load the model, can't see the music core, and scaffolds the wrong family. Would need the composite/multi-submodel feature before recommending it to Dalar for this model. |

## Bottom line
The tool's **no-device, no-auth static stages worked and gave a fast, correct memory-fit verdict (QB2 fits, 3.51 GB/chip — confirms single-card sufficiency).** But bring-up is blocked by two independent walls: a **structural** one (ACE-Step's four-sub-model custom-code repo can't be loaded by the tool's single-`from_pretrained` assumption, blocking `capture-inputs` and everything downstream regardless of auth) and an **access** one (the agentic `claude` subprocess has no auth on qb2, flagged to Moritz). The most valuable eval finding is not a perf number: it's that `plan`/`compat`/`scaffold` are **transformer-decoder-centric and silently degrade on a composite multimodal model** — they emit high-confidence output (fit table, "0 missing", 2-REUSE/0-NEW plan, TTS template) that is wrong for a diffusion music model, rather than flagging "this is a multi-model pipeline I can't characterize." Every number above is read live from tool output on qb2; nothing is fabricated.

## Durable lesson (for the orchestrator to save)
`tt_hw_planner`'s `plan`/`compat`/`scaffold` assume a single HF model loadable by one `from_pretrained` at the repo root and characterize a model purely from its top-level `config.json` + `pipeline_tag`. On a **composite custom-code model** (ACE-Step 1.5: DiT + DCAE-VAE + Qwen3 LM-planner + embedding encoder, each in its own subfolder, loaded by a bespoke pipeline) the tool (a) can't construct the model on CPU (`OSError: ... configuration_acestep_v15.py` — custom code is in a subfolder, not root), which blocks `capture-inputs`/PCC and thus the whole automated path even with valid auth; and (b) silently maps only the decoder-shaped config fields to tt_transformers LLM blocks, dropping the entire diffusion/audio-codec core and reporting "0 missing / 0 NEW". It emits confident but wrong output (TTS category, missing-on-disk sibling template) instead of hard-stopping. Composite/multi-submodel pipeline models are a real tool-feature gap. (Auth blocker on qb2 — agentic `claude -p` subprocess, no creds on the device host — is the same XTTS blocker-a and is secondary to the structural one here.)
