# tt-bio perf — 2026-06-29 — SwiGLU gate/up fusion in the shared `Transition`

**Branch:** `exp/perf-20260629-swiglu-fusion` (reset to main `c892edb`; **no code shipped**)
**Card:** tt-quietbox `/dev/tenstorrent/0` (others left free)
**Verdict:** DEAD END — neutral on Boltz-2 (no measurable speedup) **and** breaks Protenix. Reverted.

## Idea
The shared `Transition` primitive (`tenstorrent.py`) computes a SwiGLU as **two**
separate matmuls reading the same `x_norm`:
```
x_1 = linear(x_norm, fc1, activation="silu")
x_2 = linear(x_norm, fc2)
x   = x_1 * x_2 ; out = linear(x, fc3)
```
`ttnn` exposes a native fused `ttnn.swiglu(t)` = `first * silu(second)`. So concatenating
the weights as `[fc2 | fc1]` lets a **single** matmul emit `[fc2(x) | fc1(x)]`, and
`ttnn.swiglu` then yields `fc2(x) * silu(fc1(x)) == silu(fc1(x)) * fc2(x)` — reading
`x_norm` once instead of twice, saving one matmul launch, with **no host split**
(structurally different from the ruled-out AdaLN concat-split fusion, which needed an
explicit copy of two output slices).

`Transition` is shared: Boltz-2 trunk (pairformer `transition_z`/`transition_s`), the MSA
module, diffusion conditioning, Protenix, and ESMFold2 all use it. A real win here would be
broad and elegant.

## What I did
- Concatenated `fc2`/`fc1` weights into `fc12_weight` in `Transition.__init__`; replaced the
  two matmuls + `multiply` with one matmul + `ttnn.swiglu` in `swiglu()`.
- `ttnn.swiglu` indexes rank-4 internally → added a reshape wrapper for the 3D (single-track)
  transitions.

## Results — parity (Boltz-2)
`test_tenstorrent.py::test_pairformer test_msa test_diffusion` → **9/9 passed** (repo's
rel-median-err < 0.1 gate). Fusion is numerically correct for Boltz-2 trunk / MSA / diffusion.

## Results — warm A/B (TT_STAGE_PROFILE, same session, card 0, seed 0)
| input | metric | main (baseline) | fused | Δ |
|---|---|---|---|---|
| 256 fast | total | 14.03s | 14.19s | +0.16s (noise) |
|          | trunk | 6.28s  | 6.29s  | +0.01s |
| 512 fast | total | 30.14s | 30.11s | −0.03s (noise) |
|          | trunk | 18.79s | 18.60s | −0.19s |
| 686 fast | total | 51.21s* | 51.56s | +0.35s (noise) |
|          | trunk | 33.50s* | 33.39s | −0.11s |
| 512 default | total | 40.70s* | 40.38s | −0.32s (noise) |
|             | trunk | 28.58s* | 28.15s | −0.43s |

\* 686 / default baselines from the 2026-06-26 journal (256/512 are same-session A/B). No OOM
at the ceiling; default path runs fine.

**Neutral at every size & both modes** — all deltas are within the documented run-to-run
variance (±1–3s from host contention + device nondeterminism). The transition is **FLOP-bound**:
fusing 2 matmuls into 1 does not reduce MACs; the only savings (one `x_norm` re-read + one
kernel launch) are negligible on the warm program-cache path, and `transition_*` is a minority
of the trunk (trimul + triattn dominate, per prior nights).

## Results — robustness (the kill shot)
`Transition` is shared, so I ran Protenix parity:
`test_protenix_trunk_pairformer.py`, `test_protenix_diffusion_cond.py` (3 tests).
- **main:** 3/3 passed.
- **fused:** 3/3 **FAILED** — `TT_FATAL: Invalid arguments to reshape`.

The rank-reshape wrapper around `ttnn.swiglu` does not generalize to Protenix-v2's transition
shapes (wider `c=256` channels / different dims). The change **regresses a shared primitive**.

## Decision
Neutral on the target model **and** breaks another model that shares the primitive → revert.
Branch reset to main; tree left clean (two pre-existing untracked WIP test files preserved).
**Recommendation: do not merge.** Even a Protenix-correct version would be a no-op on Boltz-2.

## Takeaway
Confirms the standing verdict (7+ nights): the Boltz-2 `--fast` trunk is **at ceiling** and its
ops are FLOP-bound, not launch/bandwidth-bound. Op-fusion that doesn't cut MACs buys nothing
warm. `ttnn.swiglu` is rank-4-only and not a drop-in for the variable-rank shared `Transition`.
The only remaining ≥5% lever is algorithmic / multi-device (trunk-only TP) — a multi-day
attended port, not an overnight knob.
