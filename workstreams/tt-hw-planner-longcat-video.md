# Workstream — tt_hw_planner LongCat-Video (Model D)

Eval of `tt_hw_planner` (apande-TT/tt-metal `feature/tt-hw-planner`, PR #46283) driving
`meituan-longcat/LongCat-Video` (13.6B video DiT, text-to-video composite) through bring-up.
Live on qb1 (tt-quietbox), 4x Blackhole P150, card 0, `TT_VISIBLE_DEVICES=0`, 2026-07-15.

Full Model D section (same depth/format as Models A/B/C) is in `artifacts/feedback-snapshot.md`
and has been merged into `/home/moritz/tt-hw-planner-feedback.md` on pc (the running deliverable).

## Outcome — hit the same composite-model wall as ACE-Step (Model C), confirmed on a video DiT

Real `plan`/`compat`/`scaffold` output captured live. Bring-up (`auto-up`/`emit-e2e`/`optimize`)
NOT REACHED — blocked upstream by the structural wall (g); no device fd opened, no perf numbers
fabricated. This is confirming evidence for the composite-model-gap finding (rec 14), not a wasted
task — the wall is not audio-specific; it hits video diffusion too.

### Real repo structure (hand-verified HF tree)
- diffusers folder layout: `dit/`, `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`, `lora/`.
- BUT root `config.json` and `model_index.json` are both just `{"model_name":"LongCat-Video"}` —
  NOT a valid diffusers pipeline manifest (no `_class_name`, no component registry).
- DiT: `LongCatVideoTransformer3DModel`, hidden 4096, depth 48, heads 32, in/out 16,
  patch [1,2,2], Block Sparse Attention (sparsity 0.9375, chunk [4,4,4]); 6 shards ~54.3 GB fp32.
- VAE: `AutoencoderKLWan`, z_dim 16, ~0.5 GB.
- Text encoder: `UMT5EncoderModel` (umt5-xxl), d_model 4096, 24 layers, ~22.7 GB fp32.
- Reference load is bespoke: `run_demo_text_to_video.py` (torchrun --checkpoint_dir=...),
  NOT a HF auto-class from_pretrained.

### `plan` (live) — PASS with one defect
- 83.3 GB on-disk; Category = Video (text-to-video) [CORRECT, unlike ACE-Step TTS miscategorization].
- Single P150: NO FIT (84.3 GB vs 29.2 GB usable, -55.1 GB). QB2 4-chip mesh: FITS (+6.8 GB).
  -> Contradicts Saurabh's "fits single P150" claim; telegrammed Moritz.
- Defect (i): assumed bf16 -> reported 41.64 B params (on-disk/2), but weights are fp32 on disk;
  real ~19-20 B. Fit verdict still correct (byte-driven).

### `compat` (live) — UNKNOWN headline + generic LLM-decoder block list
- `architecture_family = "unknown (no model_type)"`, `overall = "UNKNOWN"` (more honest than
  ACE-Step's false "FEASIBLE").
- But emitted a 27-block generic LLM template: Token embedding / RoPE / RMSNorm / SwiGLU / LM head /
  KV-cache Generator / Top-k marked needed+SUPPORTED+drop-in — none match a video diffusion pipeline.
- Never opened dit/vae/text_encoder subfolders. `kernel_constraints.findings_by_tp` empty for all TP.

### `scaffold` (live) — HARD-FAIL on load, then silent degradation (same as ACE-Step rec 15)
- `ValueError: Unrecognized model in meituan-longcat/LongCat-Video. Should have a model_type key
  in its config.json.` (root config has no model_type -> AutoModel.from_pretrained cannot build it).
- Then SILENTLY degraded to `hf_eager universal (Video)` sibling (base demo.py missing on disk),
  emitted 2 REUSE / 0 NEW (self_attention + mlp from tt_transformers) — omits the entire DiT, VAE,
  T5, scheduler, BSA, patchify layers. Confident wrong plan on top of a failed load.

## Blockers
- (g) Structural: composite pipeline + non-standard root manifest; tool assumes one root
  from_pretrained builds everything. Upstream of auth. Same family as Model C (g).
- (a) Auth: partial on qb1 — `~/.claude/.credentials.json` exists but `claude` CLI not on PATH;
  tool `claude -p` subprocesses cannot run headless. Moot (g blocks first).
- (i) plan fp32 param-overcount (new).

## Seven-criteria (Model D) — honest partial
| Criterion | Score |
|---|---|
| Bring-up Efficiency | 2 |
| Optimization Effectiveness | N/A |
| Final Performance Achievement | N/A |
| Optimization Discovery | N/A |
| Engineering Insight Quality | 2 |
| Workflow Usability | 2 |
| Adoption Intent | 1 (this model class, as-is) |

Full per-cell evidence in `artifacts/feedback-snapshot.md` (Model D section).

## Recommendations (in addition to A 1-8, B 9-13, C 14-16)
17. Treat non-standard `model_index.json` / missing root `model_type` as composite-pipeline signal ->
    HARD-STOP "characterize each subfolder" (extends rec 14).
18. Do not emit an LLM-decoder block template for a non-LLM / UNKNOWN architecture.
19. Fix `plan` param count for fp32-stored repos (read component `torch_dtype`).
20. Confirm rec 15 (no silent degradation on load failure) on the Video path too.

## Env note
Canonical `create_venv.sh` is a multi-hour full-source build; instead built a py3.10 venv with the
tool public PyPI pins (torch==2.11.0+cpu, transformers==5.10.2, diffusers==0.33.0) reusing the
host pre-built `ttnn` from /home/ttuser/tt-metal via PYTHONPATH+LD_LIBRARY_PATH to run no-device
stages. Raw outputs: artifacts/{compat.json,scaffold.out,scaffold.err}.
