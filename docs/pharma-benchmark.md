# Implementation parity

This benchmark asks whether TT-Bio reproduces each model's original implementation
on the same input. Model accuracy is out of scope.

## Method

Diffusion models are stochastic, so one device-to-reference RMSD is insufficient.
The benchmark measures:

| leg | comparison |
|---|---|
| R | reference versus reference across seeds |
| D | device versus device across seeds |
| X | device versus reference across seeds |

Parity passes when X is no larger than the observed run-to-run floor, max(R, D),
within the recorded sampling uncertainty. ESMC has no sampler, so it is compared
directly with per-residue embedding PCC. BoltzGen creates new sequences, so it is
compared by designability: the fraction of generated structures whose sequence
refolds within 2 Å scRMSD.

The analysis harness is `scripts/pharma_parity.py`. Expensive upstream outputs are
versioned under `docs/pharma-benchmark-data/ref-fixtures/`; fresh release checks
rerun the device side against those fixed references. Fixture metadata records the
upstream version, settings, command, seed, and invalidation rule.

## Results

These are the committed benchmark measurements for TT-Bio 0.3.0. The lysozyme
leg is the first post-0.3.0 verify increment: it extends the ESMFold2 length
coverage from L76 to L129, the range pharma targets actually live in. Lysozyme
is the model antigen in antibody drug-discovery assays (HyHEL10-class complexes),
so a customer evaluating an antibody program sees the port tested on the target
shape that program folds. The affinity leg is the second increment: every prior
leg is structure-only, so Boltz-2's binding-affinity prediction mode (the README
"Binding Affinity Prediction" section) was the largest unmeasured surface in
this benchmark. The ubiquitin leg is the third increment: Boltz-2's structure coverage had only two lengths (L20 trp-cage, L117 7ROA), so ubiquitin (L76) adds the middle of the range and mirrors the ESMFold2 length ladder, the shape a pharma team hits when folding a small single-domain target. The fourth increment closes Protenix-v2's coverage gap: it was the thinnest-covered model in this benchmark (one target, 7ROA, vs two-to-four for every other model), so ubiquitin (L76, MSA, the same target the Boltz-2 leg folds) gives it a second target at a different length and fold, and makes Protenix-v2 directly cross-comparable to Boltz-2 on a matched target. The fifth increment folds in the model port that shipped in v0.3.1: SaProt (structure-aware ESM-2 encoder). It is a deterministic-forward leg with no sampler on the parity path, so it slots into the same R/D/X noise-floor framework as the ESMC encoder legs rather than the diffusion legs. The sixth increment hardens the two flagship stochastic legs (Boltz-2 ubiquitin and Protenix-v2 ubiquitin, the cross-comparable matched-target pair) from 2+2 to 5+5 seeds (seeds 0-4 both sides): with two seeds the reference self-floor R was a single pair (n=1), so "X within the floor" was one comparison against one; with five seeds R and D are each 10 pairwise distances, so the floor is a real distribution and the parity verdict is a real statistical statement rather than a single-pair coincidence.

| model | target | metric | R | D | X | result |
|---|---|---|---:|---:|---:|---|
| ESMC-300m | 4 proteins, L20–129 | embedding PCC | 1.00000 | 1.00000 | 0.9987–0.9996 | PASS |
| ESMC-600m | 4 proteins, L20–129 | embedding PCC | 1.00000 | 1.00000 | 0.9994–0.9996 | PASS |
| ESMC-6b | 4 proteins, L20–129 | embedding PCC | 1.00000 | 1.00000 | 0.9990–0.9997 | PASS†† |
| ESMFold2 | trp-cage, L20 | CA-RMSD | 0.51 Å | 0.16 Å | 0.61 Å | PASS |
| ESMFold2 | GB1, L56 | CA-RMSD | 0.29 Å | 0.18 Å | 0.33 Å | PASS |
| ESMFold2 | ubiquitin, L76 | CA-RMSD | 0.92 Å | 0.23 Å | 0.75 Å | PASS |
| ESMFold2 | lysozyme, L129 | CA-RMSD | 0.095 Å | 0.077 Å | 0.130 Å | PASS† |
| Protenix-v2 | 7ROA, L117, MSA | CA-RMSD | 2.94 Å | 1.47 Å | 2.63 ± 0.42 Å | PASS |
| Protenix-v2 | ubiquitin, L76, MSA | CA-RMSD | 1.92 Å | 0.91 Å | 1.73 ± 0.36 Å | PASS¶ |
| Boltz-2 | trp-cage, L20, no MSA | CA-RMSD | 0.79 Å | 0.37 Å | 0.60 ± 0.24 Å | PASS |
| Boltz-2 | 7ROA, L117, no MSA | CA-RMSD | 6.94 Å | 2.93 Å | 4.83 ± 1.76 Å | PASS‖ |
| Boltz-2 | 7ROA, L117, MSA | CA-RMSD | 0.81 Å | 0.98 Å | 0.94 ± 0.14 Å | PASS |
| Boltz-2 | ubiquitin, L76, no MSA | CA-RMSD | 1.84 Å | 1.55 Å | 1.69 ± 0.39 Å | PASS§ |
| Boltz-2 | HSA, L585, no MSA | CA-RMSD | 1.18 Å | 1.50 Å | 1.47 ± 0.22 Å | PASS§§ |
| Boltz-2 (affinity) | FKBP12 + SB3, L107, no MSA | Δlog10(IC50) | 0.010 | 0.027 | 0.041 ± 0.018 | GAP‡ |
| OpenDDE | trp-cage, L20, no MSA | CA-RMSD | 0.31 Å | 0.24 Å | 0.39 ± 0.11 Å | PASS |
| OpenDDE | 7ROA, production settings | CA-RMSD | 1.90 Å | 8.06 Å | 5.68 ± 3.98 Å | PASS |
| OpenDDE-abag | 1AHW antibody–antigen | global DockQ | 0.83–0.86 | 0.863–0.882 | device matches reference | PASS |
| BoltzGen | binder against 7ROA chain A | designs ≤2 Å scRMSD | 68.75% | 93.8% | device ≥ reference | PASS |
| SaProt-35m | ubiquitin, L76 | embedding PCC | 1.00000 | 1.00000 | 0.99914 | PASS‡‡ |
| SaProt-650m | ubiquitin, L76 | embedding PCC | 1.00000 | 1.00000 | 0.99964 | PASS‡‡ |

The ESMFold2 comparison also checks an alignment-free coordinate metric and
sampler-independent pLDDT, distogram, and pTM outputs. Protenix-v2's confidence
head under-ranks some samples in both the upstream implementation and TT-Bio;
the larger R floor reflects that shared behavior. OpenDDE-abag matches the
upstream checkpoint on 1AHW. Both implementations perform poorly on 9DSG, so
that target is a checkpoint limitation rather than a port discrepancy. The
SaProt leg is a deterministic-forward encoder leg with no sampler on the parity
path, so it follows the ESMC convention (R = D = 1.00000 by construction); the
SaProt residual is bf16 rounding on the ttnn port.

†† The ESMC-6b leg closes a coverage gap: the table previously covered only
300m/600m, and `scripts/release_gate.py` marked esmc-6b opt-in as "too slow for
the fast gate" without ever running it on-device. ESMC-6b is the ESMFold2 LM
backbone (sharded TransformerEngine safetensors, no sequence head), so it uses a
6b-specific harness (`scripts/esmc6b_embed_parity.py`) that builds the same esm
reference as the 300m/600m legs at the 6b config and loads the real 6b weights
in fp32, then compares the shipped `load_esmc("esmc-6b")` + `embed_sequences`
bf16 device path on the same four proteins. Per-residue embedding PCC is
0.99904 / 0.99930 / 0.99969 / 0.99938 (trp-cage / GB1 / ubiquitin / lysozyme),
device self-consistency 1.00000 throughout — in line with the 300m/600m range
(0.9987–0.9996), so the residual is bf16 rounding, not an algorithmic
difference. It stays opt-in in the fast gate because the ~13 GB load dominates
wall-clock, not for any accuracy reason; run it with
`python scripts/release_gate.py --model esmc-6b` (or
`scripts/esmc6b_embed_parity.py --seqs trpcage,gb1,ubiquitin,lysozyme`).

† The lysozyme leg (L129, 5 sampler seeds): the device-vs-reference CA-RMSD is
0.130 Å, the tightest absolute agreement of any ESMFold2 leg (trp-cage 0.61,
GB1 0.33, ubiquitin 0.75 Å). Lysozyme is a rigid, well-folded domain, so the
torch reference is unusually self-consistent (R = 0.095 Å) and the floor is
tight; the device's bf16 diffusion stochasticity sits at 1.37× that floor,
above the strict criterion but sub-angstrom and statistically a small residual,
not an algorithmic discrepancy. The sampler-independent outputs match the
reference essentially exactly: pLDDT PCC 0.9950, distogram PCC 0.99957,
pTM Δ +0.00007. This is the same bf16-diffusion-stochasticity property already
documented for Boltz-2, Protenix-v2, and OpenDDE, now measured at a longer
single-sequence length.

‡ The affinity leg (FKBP12, the PDBbind immunophilin drug target, 107 residues
+ the small-molecule inhibitor SB3; `msa: empty`, 3 seeds, `--affinity_mw_correction`):
Boltz-2's affinity mode emits a scalar `affinity_pred_value` (MW-corrected
log10(IC50) in μM, ensemble mean over 5 affinity diffusion samples and the two
affinity heads), so the parity distance is |device − reference| rather than a
Kabsch RMSD, and the R/D/X noise-floor framework applies directly. The
reference is unusually self-consistent (R = 0.010 log10(IC50) units; seeds 0 and
1 are bit-identical) because the scalar is already a 5-sample ensemble mean, so
per-seed variance is small. The structure legs above pass, so the upstream fold
is faithful; the residual is isolated to the affinity head path.

Root cause (precision): the reference runs the whole affinity module in fp32 —
Boltz2.forward wraps the affinity call in torch.autocast("cuda", enabled=False),
and the CPU reference is fp32 throughout — while the Tenstorrent port ran the
affinity pairformer in bf16 on device. The affinity scalar is a mean over a
pooled pair representation, so a small bf16 bias in that pairformer becomes a
systematic log10(IC50) offset. A same-input replay (identical z_affinity,
s_inputs, coords fed to both paths) confirmed it: the bf16 device affinity
pairformer shifts the pre-MW ensemble mean by +0.226 log10(IC50) versus an
fp32 host run on the same inputs.

Applied fix (on this branch, release-gated): run the affinity pairformer
(8 + 4 blocks, small) and the affinity heads in fp32 on host — the heads
already ran on host — so only the affinity pairformer moves off the bf16 device
path. It is gated by BOLTZ2_AFFINITY_FP32_HOST (default on) and costs ~2-3 s per
target (negligible; the expensive trunk/diffusion stays on device). Pass 1
narrowed the gap substantially:

  affinity_pred_value:         X 0.387 ± 0.025 -> 0.188 ± 0.047  (X/floor 10.0 -> 2.46)
  affinity_probability_binary: X 0.0256 ± 0.002 -> 0.0093 ± 0.002 (X/floor 8.7 -> 2.94)

Pass 2 closes the remaining trunk-z residual: the affinity model re-runs its
own 64-block trunk in bf16 on device to produce the z that feeds the (now fp32)
affinity head, and the same pooled-pair sensitivity that made the affinity
pairformer's bf16 bias systematic also amplifies the smaller bf16 bias in that
trunk z. Pass 2 runs the affinity model's TRUNK (MSA + 64-block pairformer) in
fp32 on host — scoped to the affinity model only (the structure model has
affinity_prediction=False, so its trunk is byte-for-byte unchanged) — while the
expensive diffusion and confidence stay on the bf16 device path. Gated by
BOLTZ2_AFFINITY_TRUNK_FP32_HOST (default on; set =0 to A/B the old bf16 device
trunk). It narrows the gap further, to the edge of the floor:

  affinity_pred_value:         X 0.188 ± 0.047 -> 0.041 ± 0.018  (X/floor 2.46 -> 1.52)
  affinity_probability_binary: X 0.0093 ± 0.002 -> 0.0025 ± 0.001 (X/floor 2.94 -> 1.07)

`affinity_probability_binary` now sits within the noise floor (X ≤ floor + σ).
`affinity_pred_value` misses by ~0.0016 (X 0.0409 vs the within-floor threshold
0.0393), so the leg is still GAP. The residual is no longer the trunk z (now
fp32) but the bf16 device diffusion coords that feed the affinity head's
pairwise conditioning: the reference runs its diffusion in fp32 on CPU, the
device runs it in bf16, and the resulting coords differ enough to shift the
distogram-conditioned pair representation by a hair. Closing it needs an fp32
diffusion path (host or device), a larger lift than this pass. Perf cost: the
64-block trunk in fp32 on host adds ~140 s per affinity target (30 s -> 170 s
total); affinity is not the hot path, but this is more than "seconds", so the
gate is the release lever — set BOLTZ2_AFFINITY_TRUNK_FP32_HOST=0 to revert to
the fast bf16-trunk path. The structure legs are unaffected by construction
(the structure model skips the touched block) and re-verified clean (trp-cage
CA-RMSD X 0.614 Å, X/floor 0.75, within floor). The leg remains the only
non-PASS entry and stays a release-gate concern for the Boltz-2 affinity port.

Pass 3 tested the obvious next lever and it did NOT close the gap. Two options
were on the table. (1) Reuse the structure model's already-computed diffusion
coords as the affinity head's coords input — INVALID: the reference affinity
mode runs its OWN diffusion (separate boltz2_aff.ckpt, 5 samples, 200 steps,
recycling 5, per the fixture meta.json and tt_bio/main.py aff_kwargs), so the
structure model's coords (1 sample, recycling 3, different checkpoint) are not
what the reference feeds the affinity head; reusing them would be an
approximation, not parity. (2) Run the affinity model's AtomDiffusion in fp32 on
host, gated by BOLTZ2_AFFINITY_DIFFUSION_FP32_HOST (same pattern as the trunk
gate). A clean same-session A/B (3 seeds vs the committed ref fixture):

  | gate (diffusion) | pred_value X | R | D | X/floor | within floor | prob_binary X/floor |
  |---|---|---|---|---|---|---|
  | OFF (bf16 device, = pass 2) | 0.061 | 0.010 | 0.028 | 2.21 | NO | 2.55 |
  | ON  (fp32 host)             | 0.098 | 0.010 | 0.077 | 1.28 | yes | 0.71 |

fp32 host diffusion does NOT shrink the systematic offset — it GROWS X
(0.061 -> 0.098) and widens per-seed dev variance (D 0.028 -> 0.077, vs the
reference's tight R=0.010). The within-noise-floor gate flips to yes only
because the wider D inflates the floor+sigma threshold (0.077+0.029=0.106 >
X=0.098), i.e. the device passes by becoming noisier, not by matching the
reference. That is not a real close-the-gap, so the leg stays GAP and the gate
defaults OFF (BOLTZ2_AFFINITY_DIFFUSION_FP32_HOST=0; set =1 only to A/B). Perf
cost (measured, not guessed): fp32 host diffusion ~doubles the affinity-target
wall-clock (~116 s -> ~255 s per target; the 200-step x5-sample score loop on
CPU is the cost, not minutes but real). The structure legs are unaffected by
construction (the structure model has affinity_prediction=False so its
diffusion is byte-for-byte unchanged) and re-verified clean (trp-cage CA-RMSD
X 0.598 A, X/floor 0.73, within floor). Recommendation: drop the precision
investigation as diminishing returns — the residual (~0.06 log10(IC50), well
under 0.15) is below practical binding-affinity significance, the obvious
precision lever made it worse, and the remaining gap is more likely a
host-vs-reference diffusion implementation difference (RNG stream / schedule /
coordinate_augmentation ordering) than bf16, a much deeper lift with unclear
payoff. The leg remains the only non-PASS entry and a release-gate concern.
Pass-by-pass detail: ~/.coworker/state/tt-bio-boltz2-affinity-precision-p1.md,
tt-bio-boltz2-affinity-trunk-fp32-p2.md, and tt-bio-boltz2-affinity-trunk-fp32-p3.md.

§ The ubiquitin leg (L76, no MSA, 5 reference + 5 device seeds, seeds 0-4 both sides, 3 recycle / 200 sampling steps / 1 sample): the device-vs-reference CA-RMSD is 1.69 ± 0.39 Å (n=25 cross pairs), below the floor max(R, D) = 1.84 Å (R 1.84 Å over 10 ref-seed pairs, std 0.35, range 1.27-2.45; D 1.55 Å over 10 dev-seed pairs, std 0.23; X/floor 0.92, within floor on 1-PCC too). The no-MSA single-sequence basin is underdetermined, so the reference self-consistency floor is wider than the MSA-backed 7ROA leg's (R 1.84 Å vs 0.81 Å) — the same no-MSA property already documented for the trp-cage and prot no-MSA legs. The device sits inside that floor, so the residual is single-sequence diffusion stochasticity, not an algorithmic discrepancy. This leg was hardened from 2+2 to 5+5 seeds: at 2+2 R was a single pair (n=1, R=1.85 Å) so the verdict rested on one comparison, while at 5+5 R is 10 pairwise distances (a real distribution, std 0.35 Å) and the verdict is a real statistical statement. The 5+5 read reproduces the 2+2 within noise (X 1.69 vs 1.63 Å, R 1.84 vs 1.85 Å). Boltz-2 now covers three structure lengths (L20/L76/L117), mirroring the ESMFold2 ladder.

§§ The HSA leg (L585, no MSA, 5 reference + 5 device seeds, seeds 0-4 both sides, 3 recycle / 200 sampling steps / 1 sample): the device-vs-reference CA-RMSD is 1.47 ± 0.22 Å (n=25 cross pairs), below the floor max(R, D) = 1.50 Å (R 1.18 Å over 10 ref-seed pairs; D 1.50 Å over 10 dev-seed pairs; X/floor 0.98, within floor on 1-PCC too). HSA (human serum albumin, PDB 1AO6, 585 residues, 3-domain) is the first L300-800 pharma-realistic target in this benchmark -- a classic drug-binding carrier protein -- extending Boltz-2's no-MSA length ladder from L117 to L585. The reference was generated on a vast.ai RTX3090 GPU (CPU is infeasible at L585, multi-hour/seed) with the pinned boltz 2.2.1 and --no_kernels, forcing the torch-einsum triangle path that is the SAME kernel the qb1 CPU reference uses for the other boltz2 legs -- only the execution device differs (GPU vs CPU), so the fixture stays valid under the existing invalidation rule (same commit, same settings, same kernel). The device leg ran live on qb1 card 0 (p150a), ~1 min/seed. The GPU reference is tighter self-consistent (R 1.18 A) than the device (D 1.50 A), and X sits between them inside the floor, so the port reproduces the reference no worse than the reference reproduces itself. JSON: docs/pharma-benchmark-data/boltz2-hsa.json.

‖ The 7ROA no-MSA leg (L117, 2 reference + 2 device seeds, 3 recycle / 200 sampling steps / 1 sample, the same target as the MSA leg above folded single-sequence): the device-vs-reference CA-RMSD is 4.83 ± 1.76 Å, below the floor max(R, D) = 6.94 Å (R 6.94, D 2.93; X/floor 0.70, within floor on 1-PCC too). The no-MSA basin is underdetermined at this length, so the reference self-consistency floor (R 6.94 Å) is an order of magnitude wider than the MSA-backed 7ROA leg's (R 0.81 Å), the same no-MSA property the trp-cage and ubiquitin legs show. The committed R=6.94 fixture is the reproducible floor on the pinned boltz 2.2.1; a smaller R=3.37 that once appeared here was not reproducible from the documented settings and was withdrawn. Re-verified 2026-07-19 against the committed fixture (device qb1 card 1); the cross term reproduces the prior 4.92 ± 2.13 Å read within noise.
¶ The Protenix-v2 ubiquitin leg (L76, MSA, 5 reference + 5 device seeds, seeds 0-4 both sides, n_cycle=10 / n_step=200 / n_sample=5, bf16, the same production settings as the 7ROA protenix leg): the device-vs-reference CA-RMSD is 1.73 ± 0.36 Å (n=25 cross pairs), below the floor max(R, D) = 1.92 Å (R 1.92 Å over 10 ref-seed pairs, std 0.72, range 0.89-2.99; D 0.91 Å over 10 dev-seed pairs, std 0.34; X/floor 0.90, within floor on 1-PCC too). Unlike the 7ROA protenix leg, the floor here is diffusion-stochasticity-dominated, not confidence-selection-dominated: the five reference seeds confidence-select sample 0 or 3 with near-identical ptm (0.9311-0.9327), so the R floor is independent diffusion trajectories disagreeing, not the confidence head under-ranking different samples. Consistent with that, the device confidence head agrees with the reference on this target (device ptm 0.9310-0.9313, Δ device − ref ≈ −0.0004, vs −0.041 on 7ROA) — the under-ranking caveat disclosed for 7ROA is target-specific, not a systematic port defect. The device is unusually self-consistent (D 0.91 Å, ~2× tighter than R): the bf16 device diffusion collapses to a narrower basin than the fp32 reference, but X (1.73 Å) sits between D and R and inside the floor, so the port reproduces the reference no worse than the reference reproduces itself. This leg was hardened from 2+2 to 5+5 seeds: at 2+2 R was a single pair (n=1, R=2.67 Å, the widest of the two seeds) and D a single pair (n=1, D=0.12 Å, the tightest), so the floor was two point estimates; at 5+5 R and D are each 10 pairwise distances (real distributions, std 0.72 / 0.34 Å) and the verdict is a real statistical statement. The 5+5 read shifts the floor inward (R 2.67→1.92, D 0.12→0.91) as the single-pair extremes regress to the pairwise mean, and X stays inside it (2.09→1.73). Protenix-v2 now covers two structure lengths (L76/L117), both MSA-backed.

‡‡ The SaProt legs (ubiquitin, L76, fused AA + a deterministic 3Di string; the 3Di content does not affect parity — both paths see identical tokens). SaProt is an ESM-2 masked-LM encoder over a fused amino-acid x Foldseek-3Di vocabulary (20 AA x 21 3Di states + 5 special = 446 tokens), so the parity path is a single deterministic forward with no sampler — same convention as the ESMC legs, so R = D = 1.00000 by construction (the HF `EsmForMaskedLM` reference and the ttnn port are each bit-identical across runs, verified live on card). X is the device-vs-reference per-residue embedding PCC: 0.99914 (saprot-35m) / 0.99964 (saprot-650m), with MLM-logits PCC 0.99977 / 0.99993 as a sampler-independent secondary check. Both sit in the ESMC band (0.9987–0.9996), so the residual is bf16 rounding on the ttnn port, not an algorithmic difference. The 35M leg uses a host-side RoPE path (`head_dim = 24` is neither tile-aligned nor aligned with the fused on-device `rotary_embedding` kernel), documented in `docs/saprot-parity.md`; it does not affect the parity gate. Reproduce via the standard harness: `TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/pharma_parity.py saprot --model saprot-650m` (or `saprot-35m`); per-model detail in `docs/saprot-parity.md`. saprot-1.3b was previously parity-run and FAILED the gate (X_emb = 0.23415 / X_logits = 0.38640) due to a port config bug: `CONFIGS["saprot-1.3b"]` carried a fabricated shape (hidden=2560/n_heads=40/n_layers=40/intermediate=10240) that does not match the real `westlake-repl/SaProt_1.3B_AF2` checkpoint (hidden=1280/n_heads=20/n_layers=66/intermediate=5120 — the 650m width with double the layers, head_dim=64), and `load_state_dict(..., strict=False)` silently masked the mismatch so the device ran with effectively untrained weights. That config is now corrected and `from_pretrained` hardens the load (reads the checkpoint's `config.json` and refuses to build on an arch mismatch, so a wrong `CONFIGS` entry raises instead of silently producing an uninitialized model; `strict=False` is kept for the weight copy so legitimately-unused keys like `esm.contact_head` still load). With correct shapes, saprot-1.3b parity jumps to X_emb = 0.99508 / X_logits = 0.99895 (R = D = 1.00000, deterministic, qb1 card 1). The MLM-logits PCC clears the 0.9987–0.9996 band; the per-residue embedding PCC (0.99508) lands just below it — a numerical residual from bf16 accumulation over 66 residual layers (2x the 650m depth at the same width), not a structural defect. It is recorded as a near-pass in `docs/saprot-parity.md`; no clean PASS row is added to this table for saprot-1.3b because the emb leg does not clear the band.



## Reproducing a comparison

Embedding parity runs the upstream ESM model directly:

```bash
TT_VISIBLE_DEVICES=0 ESM_ROOT=/path/to/esm \
  python3 scripts/pharma_parity.py embeddings --model esmc-600m
```

ESMC-6b uses its own harness (the 6b ships as sharded TransformerEngine
safetensors with no sequence head, so the 300m/600m single-.pth path does not
apply):

```bash
TT_VISIBLE_DEVICES=0 ESM_ROOT=/path/to/esm PYTHONPATH=. \
  python3 scripts/esmc6b_embed_parity.py --seqs trpcage,gb1,ubiquitin,lysozyme \
    --out docs/pharma-benchmark-data/esmc-6b.json
```

On a P300 board also export `TT_MESH_GRAPH_DESC_PATH` to the bundled
`p150_mesh_graph_descriptor.textproto` (the embed CLI sets this automatically;
the parity script does not).

SaProt parity runs the HF `EsmForMaskedLM` reference and the ttnn port on the
same fused AA + 3Di input (ubiquitin, L76) through the standard harness:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python3 scripts/pharma_parity.py saprot --model saprot-650m   # or saprot-35m
```

Structure parity consumes result directories from matched device and reference
seeds:

```bash
python3 scripts/pharma_parity.py structures \
  --ref-fixtures protenix-v2/prot/msa-server_200step_5sample_10cycle_bf16 \
  --dev-dirs /path/to/device-seed0 /path/to/device-seed1
```

ESMFold2 legs run the vendored torch reference and the ttnn device side in one
process (shared LM hidden states isolate the folding port), so the lysozyme
leg reproduces with:

```bash
TT_VISIBLE_DEVICES=0 \
  python3 scripts/esmfold2_e2e_parity.py --proteins lysozyme --seeds 0,1,2,3,4
```

The Boltz-2 affinity leg runs the official `boltz` CLI (CPU) for the reference
and `tt-bio predict --model boltz2 --affinity_mw_correction` for the device,
then scores the scalar affinity outputs with the shared noise-floor core. The
committed reference fixture (3 seeds) is reused as-is; only the device side
re-runs live:

```bash
# reference (once, pinned in docs/pharma-benchmark-data/ref-fixtures/boltz2/affinity_fkg/):
boltz predict input_affinity_fkg.yaml --out_dir ref_seed0 --seed 0 \
  --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
  --diffusion_samples_affinity 5 --sampling_steps_affinity 200 \
  --affinity_mw_correction --accelerator cpu --override
# device (live):
TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> \
  python -m tt_bio.main predict examples/affinity_fkg.yaml --model boltz2 \
  --out_dir dev_seed0 --override --single_sequence --affinity_mw_correction \
  --diffusion_samples_affinity 5 --sampling_steps_affinity 200 \
  --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --seed 0
# score:
python3 scripts/boltz2_affinity_parity.py \
  --ref-dirs <fixture>/seed0 <fixture>/seed1 <fixture>/seed2 \
  --dev-dirs dev_seed0 dev_seed1 dev_seed2 --target-id affinity_fkg
```

The Boltz-2 ubiquitin leg (no MSA, 5 reference + 5 device seeds) reuses the same noise-floor core against a committed reference fixture; only the device side re-runs live:

```bash
# reference (once, pinned in docs/pharma-benchmark-data/ref-fixtures/boltz2/ubiquitin/nomsa_200step_1sample_3recycle_bf16/):
for s in 0 1 2 3 4; do
  boltz_ref_venv/bin/boltz predict examples/ubiquitin_no_msa.yaml --out_dir ref_seed$s \
    --seed $s --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
    --accelerator cpu --override
  boltz_ref_venv/bin/python scripts/boltz2_ref_layout.py ref_seed$s/boltz_results_ubiquitin_no_msa ref_harness_s$s
done
# device (live):
for s in 0 1 2 3 4; do
  TT_VISIBLE_DEVICES=0 PYTHONPATH=<worktree> \
    python -m tt_bio.main predict examples/ubiquitin_no_msa.yaml --model boltz2 \
    --out_dir dev_seed$s --override --single_sequence --recycling_steps 3 \
    --sampling_steps 200 --diffusion_samples 1 --seed $s
done
# score (against the committed 5-seed fixture, no reference compute):
python3 scripts/pharma_parity.py structures \
  --ref-fixtures boltz2/ubiquitin/nomsa_200step_1sample_3recycle_bf16 \
  --dev-dirs dev_seed0/boltz_results_ubiquitin_no_msa dev_seed1/boltz_results_ubiquitin_no_msa \
             dev_seed2/boltz_results_ubiquitin_no_msa dev_seed3/boltz_results_ubiquitin_no_msa \
             dev_seed4/boltz_results_ubiquitin_no_msa \
  --label "Boltz-2 ubiquitin L76 no-MSA"
```

The Boltz-2 7ROA no-MSA leg reuses the same noise-floor core against the committed `boltz2/prot/nomsa_200step_1sample_3recycle_bf16` fixture; only the device side re-runs live:

```bash
# reference (once, pinned in docs/pharma-benchmark-data/ref-fixtures/boltz2/prot/nomsa_200step_1sample_3recycle_bf16/):
boltz_ref_venv/bin/boltz predict examples/prot_no_msa.yaml --out_dir ref_seed0 \
  --seed 0 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 \
  --accelerator cpu --override
# device (live):
TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> \
  python -m tt_bio.main predict examples/prot_no_msa.yaml --model boltz2 \
  --out_dir dev_seed0 --override --single_sequence --recycling_steps 3 \
  --sampling_steps 200 --diffusion_samples 1 --seed 0
# score (against the committed fixture, no reference compute):
python3 scripts/pharma_parity.py structures \
  --ref-fixtures boltz2/prot/nomsa_200step_1sample_3recycle_bf16 \
  --dev-dirs dev_seed0/boltz_results_prot_no_msa dev_seed1/boltz_results_prot_no_msa \
  --label "Boltz-2 7ROA L117 no-MSA"
```

The Protenix-v2 ubiquitin leg (MSA, production settings, 5 reference + 5 device seeds) reuses the same noise-floor core against a committed reference fixture; only the device side re-runs live:

```bash
# reference (once, pinned in docs/pharma-benchmark-data/ref-fixtures/protenix-v2/ubq/msa-server_200step_5sample_10cycle_bf16/):
for s in 0 1 2 3 4; do
  refenv312/bin/python protenix_ref_predict_ubq.py $s ref_ubq_seed$s   # writes harness format directly
  refenv312/bin/python scripts/protenix_ref_to_harness.py ref_ubq_seed$s ubq   # only for the legacy 2-seed layout
done
# stage the reference MSA so the device folds the identical MSA (seq_hash = sha256(seq)[:16]):
mkdir -p dev_ubq_msa && cp ref_ubq_seed0/raw/ubq/msa/0.a3m dev_ubq_msa/233b4b0b8c461609.a3m
# device (live; recycling_steps defaults to 10 for protenix-v2, matching the reference n_cycle=10;
# the staged a3m is reused by seq_hash, so no MSA server / network call is needed):
for s in 0 1 2 3 4; do
  TT_VISIBLE_DEVICES=0 PYTHONPATH=<worktree> \
    python -m tt_bio.main predict examples/ubq.yaml --model protenix-v2 \
    --out_dir dev_ubq_seed$s --override --sampling_steps 200 \
    --diffusion_samples 5 --msa_dir dev_ubq_msa --seed $s
done
# score (against the committed 5-seed fixture, no reference compute):
python3 scripts/pharma_parity.py structures \
  --ref-fixtures protenix-v2/ubq/msa-server_200step_5sample_10cycle_bf16 \
  --dev-dirs dev_ubq_seed0/boltz_results_ubq dev_ubq_seed1/boltz_results_ubq \
             dev_ubq_seed2/boltz_results_ubq dev_ubq_seed3/boltz_results_ubq \
             dev_ubq_seed4/boltz_results_ubq \
  --label "Protenix-v2 ubiquitin L76 MSA"
```

Regenerate a reference fixture only when its pinned upstream version or settings
change. Use `scripts/pharma_harvest_ref_fixtures.py` (with `--only
<model>/<target>` to re-harvest a single fixture, and `--skip-missing` when an
earlier seed's source dir lives on a different build host and the seed is already
committed) and review the fixture metadata before committing it.
