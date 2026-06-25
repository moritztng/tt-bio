# Perf run 2026-06-25 — Validating the ttnn-trace "lossless WIN": it regresses folds

Branch: `exp/perf-20260625-attnpairbias` (off main 8a40c42). Pushed for review. NOT merged.
Engineer: autonomous overnight run on tt-quietbox (4× Blackhole).

## TL;DR
The recorded **"ttnn TRACE = lossless WIN (trunk −14…−24%, e2e −7.6%)"** does **not** hold
for the trunk. The trace gives a real warm speedup, and every trace module is **per-op
bit-identical to eager (PCC=1.0, maxdiff=0)** — but enabling the **Pairformer+MSA (trunk)
trace reproducibly REGRESSES the folded structure** on a confident target (hemoglobin
~0.8 Å → ~2.4 Å Cα-RMSD, every seed tested). Per-op PCC misses this because the Boltz-2
fold is **chaotically sensitive** (eager-vs-eager spans 0.78–2.80 Å at a fixed seed), so the
trace replay's altered noise structure tips borderline folds into a worse basin. The
**diffusion-step trace** is the only per-step-bit-identical module and the only one with a
defensible (small-protein-only) win. **Code change:** the master `TT_BIO_TRACE` now enables
**only the diffusion trace**; the fold-regressing trunk traces require explicit per-module
opt-in. This prevents a bad merge and corrects the journal.

## Target & rationale
Boltz-2 --fast per-op kernels are at ceiling (LEARNINGS: trimul subblock, SDPA chunk,
exp_approx, bf8 weights, HiFi2 all dead ends). I verified the surface is mined:
AttentionPairBias (18.3% of trunk) is genuine per-layer compute (z changes each layer, no
reuse); `diffusion_samples=1` by default (no sample-batching fruit); the Boltz-2 diffusion
token-transformer pair bias is already precomputed (`compute_pair_bias=False`). So the only
unmerged "win" worth hardening was the **ttnn trace** (branch exp/perf-20260621-diffusion-
resident), validated previously at only ONE point (L=512 --fast). Moritz's gate is "works
for ALL inputs", so I validated the full matrix + re-checked losslessness rigorously.

## Warm speedup — full matrix (forward_total = trunk+diffusion+confidence; 2nd protein warm)
Master `TT_BIO_TRACE=1` (all modules; diffusion auto-gated off >384) vs OFF, clean symmetric
A/B pairs (median where noted):
| size | mode | OFF e2e | ON e2e | e2e Δ | trunk Δ | diffusion Δ |
|------|------|---------|--------|-------|---------|-------------|
| 256  | fast | 15.57s  | 13.35s | −14.3% | −6.4%  | −21.9% (diff trace on, 256<384) |
| 512  | fast | 33.4s   | ~31.6s | ~−5.3% (median of 4) | −9 to −13% | gated off |
| 686  | fast | 57.35s  | 51.62s | −10.0% | −14.9% | gated off |
| 256  | def  | 15.91s  | 14.65s | −7.9%  | ~0     | −16.8% |
| 512  | def  | 43.59s  | 39.33s | −9.8%  | −14.3% | gated off |
| 686  | def  | 75.29s  | 68.27s | −9.3%  | −14.6% | gated off |
No OOM at any size (2GiB trace region co-resident). Speedup is REAL. Per-module breakdown at
512 fast (trunk stage, contention-robust): Pairformer −9.5%, MSA −2.1%, all −10.0%; at 256 the
diffusion trace dominates. NB env-var gotcha: the Pairformer kind is `pairformer`
(TT_BIO_TRACE_PAIRFORMER), not `pf`.

## Losslessness — per-module, the playbook's prescribed test (in-process, byte-identical input)
Added a gated in-process check: on the trace-replay path, recompute eager on the SAME staged
input and compare. ALL THREE modules, both --fast and default:
- **Pairformer trace: PCC=1.00000000, maxdiff=0, bit-identical** (s and z)
- **MSA trace:        PCC=1.00000000, maxdiff=0, bit-identical**
- **Diffusion trace:  PCC=1.00000000, maxdiff=0, bit-identical**
So the trace replays the identical kernel stream — per-op lossless, as the journal said.

### Methodology trap I fell into (and corrected)
A naive **cross-run** trunk-output comparison (OFF-run vs trace-run) showed z-PCC 0.991–0.997
vs an OFF-vs-OFF floor of 0.9998 — looking like trace lossiness. This is WRONG: it conflates
run-to-run device nondeterminism (amplified through the 4-iter recycling feedback loop) with
any trace effect. The in-process same-input test (above) is the correct, noise-free test and
shows perfect bit-identity. **Lesson: never compare across runs on a nondeterministic device;
capture-then-eager-recompute in one process.**

## End-to-end fold quality — where the trunk trace fails (the real finding)
Per-op bit-identity is necessary but NOT sufficient. Folded hemoglobin (574 res, 4 chains,
ground-truth available), --fast, seed-paired, **byte-identical MSA** (md5-verified across all
runs), Cα-RMSD vs ground truth:

| config | seed 2 | seed 4 | other seeds |
|--------|--------|--------|-------------|
| OFF (eager)              | 0.78, 0.85, 0.87 ✓ | 0.84, 0.93 ✓ | 48: 0.79–2.80 (bimodal); 0:0.97; 7:1.04 |
| region reserved, no trace| 0.96, 0.90 ✓        | 2.59, 1.06 mixed | — |
| **trunk trace (PF+MSA)** | **2.31, 2.45, 2.40 ✗** | **2.29, 2.45 ✗** | 0:2.50, 7:2.34, 48:2.32/2.53 — all ✗ |
| diffusion trace (forced) | 2.78, 2.52 ✗        | 0.89, 0.87 ✓ | 48: 0.94/0.97 |

- **OFF reliably folds to ~0.8 Å at seeds 2,4 (5/5).** The **trunk trace NEVER does (5/5 bad
  at seeds 2,4; ~12/12 bad across seeds 1,2,3,4,48)** → reproducible regression, not noise.
- The Boltz-2 fold is **chaotically sensitive**: pure-eager (OFF) hemoglobin spans 0.78–2.80 Å
  at fixed seed 48 (device nondeterminism alone → different diffusion trajectory → different
  basin). This is exactly why fold metrics can't judge losslessness (LEARNINGS METHOD NOTES),
  and why the original journal — relying on per-step PCC — missed the fold regression.
- The diffusion trace and the bare region reservation have milder, **seed-dependent** effects
  within the chaotic envelope; the **trunk trace** is the consistent, robust regressor.

## Code change (minimal)
`tt_bio/tenstorrent.py`: master `TT_BIO_TRACE` now enables only the per-step-bit-identical
**diffusion** trace (`_TRACE_MASTER_SAFE_KINDS`). The fold-regressing **trunk** traces require
an explicit opt-in (`TT_BIO_TRACE_PAIRFORMER` / `TT_BIO_TRACE_MSA`). Default-off unchanged.
Corrected two misleading comments that claimed the trunk traces "still win at every size and
are unaffected." +18/−5 lines.

## Recommendation
- **Do NOT merge/enable the trunk (Pairformer+MSA) trace.** It is per-op bit-identical but
  reproducibly regresses fold quality on a confident target. The headline trunk speedup
  (−10…−14% at 512/686) is not free — it costs structure accuracy. The recorded "lossless
  WIN" claim must be qualified accordingly.
- **Diffusion trace** (master switch): per-step bit-identical, small-protein-only win
  (~−10% e2e at L≤384, auto-gated off above). Borderline on chaotic targets — validate fold
  quality per target before relying on it. Kept under the master switch as the only safe-ish
  component; default still OFF.
- **Follow-up (not done tonight):** root-cause why per-op-identical trunk replay biases the
  chaotic recycling+diffusion pipeline into a worse basin (RNG/noise-structure or trace memory
  layout). Until then the trunk trace stays opt-in only.

## Validation status vs playbook gate
Accuracy (per-module PCC): bit-identical ✓ (all 3 modules, both modes). All inputs:
256/512/686, --fast + default, no OOM ✓. End-to-end Cα-RMSD on a confident target: **trunk
trace REGRESSES** ✗ → not a win; diffusion trace within chaotic envelope. Honest verdict:
the trace is not the clean lossless win it was recorded as.
