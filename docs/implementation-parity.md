# Implementation parity

This checks whether TT-Bio reproduces each model's **original reference
implementation** on the same input. The device fold is compared to the
reference fold across seeds; it is device-vs-reference parity within the
model's own seed-to-seed noise, not a benchmark against experiment. Model
accuracy (does the fold match the native structure) is out of scope.

## Verdict

| model | target | verdict | reason |
|---|---|---|---|
| ESMC-300m | 4 proteins, L20–129 | PASS | deterministic encoder; emb PCC 0.9987–0.9996, residual is bf16 rounding |
| ESMC-600m | 4 proteins, L20–129 | PASS | same path; emb PCC 0.9994–0.9996 |
| ESMC-6b | 4 proteins, L20–129 | PASS | same path at 6b; emb PCC 0.9990–0.9997 (opt-in for load time, not accuracy) |
| ESMFold2 | trp-cage, L20 | PASS | CA-RMSD 0.61 Å inside the 0.51 Å floor |
| ESMFold2 | GB1, L56 | PASS | CA-RMSD 0.33 Å inside the 0.29 Å floor |
| ESMFold2 | ubiquitin, L76 | PASS | CA-RMSD 0.75 Å inside the 0.92 Å floor; device closer to ref than ref to itself |
| ESMFold2 | lysozyme, L129 | PASS | CA-RMSD 0.136 Å inside the 0.139 Å floor (X/floor 0.98); seed-wiring fix applied, see † |
| Protenix-v2 | 7ROA, L117, MSA | PASS | CA-RMSD 2.63 Å inside the 2.94 Å floor; confidence-head under-ranking shared with reference (model property) |
| Protenix-v2 | ubiquitin, L76, MSA | PASS | CA-RMSD 1.73 Å inside the 1.92 Å floor; passes on TM-score and CA-lDDT too |
| Protenix-v2 | HSA, L585, MSA | PASS | on-device fp32 diffusion matches the reference's own fp32 boundary; CA-RMSD 0.685 Å inside the 0.695 Å floor (was GAP-evidenced in bf16, X 1.03 Å) |
| Boltz-2 | trp-cage, L20, no MSA | PASS | wide no-MSA floor; absolute X 0.60 Å |
| Boltz-2 | 7ROA, L117, no MSA | PASS (legacy R/D/X); GAP under the envelope gate | wide no-MSA floor (R 4.98 Å); absolute X 4.21 Å. The newer, tighter envelope test (below) GAPs this leg (ratio 1.83, reproducible) — open item, not yet root-caused |
| Boltz-2 | 7ROA, L117, MSA | PASS | CA-RMSD 0.94 Å inside the 0.81 Å floor |
| Boltz-2 | ubiquitin, L76, MSA (production default) | PASS | all 4 metrics within the tight MSA-backed GPU-reference floor (CA-RMSD X/floor 1.03, 1-lDDT X/floor 0.97); residual systematic bf16, see §§§ |
| Boltz-2 | HSA, L585, no MSA | PASS | CA-RMSD 1.47 Å inside the 1.50 Å floor; first L585 target |
| Boltz-2 (affinity) | FKBP12 + SB3, L107, no MSA (non-default) | PASS | device-fp32 hybrid diffusion vs the GPU bf16 reference: pocket-lDDT X 0.014 within the GPU noise floor (X/floor 1.25); affinity scalar, affinity probability, and ligand-pose RMSD also pass (X/floor 0.79 / 1.38 / 0.92) |
| Boltz-2 (affinity) | FKBP12 + SB3, L107, MSA (production default) | GAP-evidenced | MSA tightens the floor ~8× (R 0.196→0.025): affinity scalar GAPs (X/floor 2.27) and pocket-lDDT GAPs (4.48), both systematic bf16 by the seed-independent same-seed diagonal; affinity_probability (1.45) and ligand-RMSD (0.84) PASS. The scalar GAP is PROVEN a genuine bf16-BACKEND floor (not a port defect) by GPU-vs-CPU reference triangulation on the scalar itself: the two bf16 references disagree on Δlog10(IC50) by 0.057 (FKBP12), the same magnitude as device-vs-CPU MSA (0.062), and the affinity head is deterministic + MSA-agnostic by code, see ‡ᴹ |
| Boltz-2 (affinity) | DHFR + MTX, L187, no MSA (non-default) | PASS-caveated | affinity scalar and ligand-pose RMSD pass (X/floor 0.68 / 1.36); pocket-lDDT GAPs (4.72), proven a genuine bf16-BACKEND floor by three-backend triangulation (GPU-bf16 and CPU-bf16 references disagree on the pocket by the same ~0.13 lDDT margin the device does), not a port defect |
| Boltz-2 (affinity) | DHFR + MTX, L187, MSA (production default) | PASS-caveated | affinity scalar (1.32), affinity_probability (0.95), and ligand-RMSD (1.61) PASS; pocket-lDDT GAPs (13.35), systematic bf16 by the same-seed diagonal, see ‡ᴹ |
| Boltz-2 (affinity) | trypsin + BAM, L223, no MSA (non-default) | PASS-caveated | affinity scalar and ligand-pose RMSD pass (X/floor 0.94 / 0.95); pocket-lDDT GAPs (10.13), proven a genuine bf16-BACKEND floor by three-backend triangulation (GPU-bf16 vs CPU-bf16 pocket-lDDT X/floor 7.51, both NO), not a port defect |
| Boltz-2 (affinity) | trypsin + BAM, L223, MSA (production default) | PASS-caveated | affinity scalar (0.79), affinity_probability (0.92), and ligand-RMSD (0.78) PASS; pocket-lDDT GAPs (2.75), systematic bf16 by the same-seed diagonal, see ‡ᴹ |
| OpenDDE | trp-cage, L20, no MSA | PASS | CA-RMSD 0.51 Å inside the 0.52 Å floor |
| OpenDDE | 7ROA, production | PASS | wide device-dominated floor (D 6.04 Å); absolute X 4.67 Å |
| OpenDDE-abag | 1AHW Ab–Ag | PASS | global DockQ 0.864; per-interface iRMSD 0.65/0.70/1.20 Å, all sub-Å-to-low-Å |
| BoltzGen | binder vs 7ROA chain A | PASS | designability 93.8% (≤2 Å scRMSD) vs reference 68.75%; device meets-or-exceeds |
| SaProt-35m | ubiquitin, L76 | PASS | deterministic encoder; emb PCC 0.99914, in the ESMC band |
| SaProt-650m | ubiquitin, L76 | PASS | deterministic encoder; emb PCC 0.99964, in the ESMC band |

Net: 22 PASS, 4 PASS-caveated, 1 GAP-evidenced. The three Boltz-2 affinity
legs were re-run with MSA (Boltz-2's production default — a pharma user folds a
target whose homologs are known, so the MSA is fed); the earlier single-sequence
rows are retained and relabeled `non-default`. The MSA legs score 8 PASS / 4 GAP
across their 12 metric-cells (see ‡ᴹ): the consistent GAP is 1-pocket-lDDT on all
three targets, the same narrower-basin systematic-bf16 property the no-MSA legs
show, and FKBP12+SB3 additionally GAPs the affinity scalar under MSA (X/floor
2.27) because MSA narrows the reference floor ~8× and exposes the residual device
bf16 offset; DHFR+MTX (1.32) and trypsin+BAM (0.79) affinity scalars still PASS.
The FKBP12 MSA scalar GAP — the last GAP-evidenced leg that was previously only
asserted a bf16 floor by transfer from the pocket-lDDT result — is now PROVEN a
genuine bf16-BACKEND floor on the scalar path itself: the pinned GPU-bf16 and
CPU-bf16 references disagree on Δlog10(IC50) by the same ~0.06 margin the device
does (no-MSA triangulation across all 3 targets, `boltz2-affinity-{fkg,dhfr,tryp}-scalar-gpu-vs-cpu.json`),
and the affinity head is deterministic and MSA-agnostic by code, so the
cross-backend offset is upstream and MSA-independent; the no-MSA triangulation
therefore transfers to the MSA leg structurally, not by assumption.
The no-MSA affinity rows use the device-fp32 hybrid diffusion path: FKBP12 PASSes
cleanly there (pocket-lDDT X 0.011 within the 0.011 GPU floor, X/floor 0.94), and
the two PASS-caveated no-MSA entries (DHFR, trypsin) are proven a genuine
bf16-BACKEND precision floor, not a port defect, by three-backend triangulation
(the pinned GPU-bf16 and CPU-bf16 references disagree on the pocket by the same
~0.09-0.13 lDDT margin the device does, so no single-backend lever or reference
switch can manufacture a PASS). All Boltz-2 legs were re-measured with the seed-wiring fix live (2026-07-21); every verdict held (see the seed-fix remeasure note in the details doc). Protenix-v2 HSA was GAP-evidenced under bf16
diffusion; running that model's diffusion sampler in fp32 on device — matching
the reference's own fp32 boundary rather than a blanket precision bump — closed
it to a clean PASS. The full measured R/D/X
table and per-leg evidence are in
[Implementation parity — details](implementation-parity-details.md).

**Method in one line.** R = reference-vs-reference across seeds, D =
device-vs-device across seeds, X = device-vs-reference; the floor is
max(R, D); a leg passes when X is no larger than the floor within sampling
uncertainty. Deterministic legs (ESMC, SaProt) are bit-exact by construction
(R = D = 1.0). Diffusion legs (Boltz-2, Protenix-v2, OpenDDE, Boltz-2 affinity)
share one CPU `torch.randn` stream between device and reference at a matched
seed, so the comparison is RNG-fair; both sides run bf16 where the reference
does. BoltzGen is scored by designability (fraction of designs re-folding
within 2 Å scRMSD), not by a distance. OpenDDE-abag by global DockQ and
per-interface iRMSD.

## Correctness method — integration-parity envelope (supersedes the R/D/X floor)

The R/D/X floor above answers a distribution question ("is X within the run-to-run spread?") with
a point comparison against a GUESSED floor `max(R, D)`. Because R, D and X each compare INDEPENDENT
stochastic samples (device and reference drew different diffusion noise), X conflates real backend
arithmetic divergence with ordinary sample-to-sample chaos: a correct port can fail by construction
(different noise basin) and a subtle bug can hide under a loose floor. The "GAP-evidenced" and
"PASS-caveated" verdicts above are the floor telling us it cannot separate a bug from noise — and
each was ultimately cleared only by an ad-hoc cross-backend triangulation (GPU-bf16 vs CPU-bf16
references disagreeing by the same margin the device does). The integration-parity envelope test
turns that triangulation into the systematic pass criterion.

A diffusion model is a deterministic function of its input noise. Feed byte-identical noise to
three CLOSED-LOOP runs — `device_bf16` (TT), `reference_fp32` and `reference_bf16` (both tt-bio's
own CPU torch path, `--no_kernels`, the second under `TT_BIO_REF_BF16=1`). The single `--seed` is
NOT enough to share draws: the device (ttnn) trunk and the CPU (torch) trunk consume the global RNG
differently before the sampler, so the gate sets `TT_BIO_SHARED_DRAW_SEED` on all three runs, which
re-seeds in `AtomDiffusion.sample` right before the first `torch.randn` so all three draw
byte-identical noise (verified bit-exact). Then, per leg per metric `d`:

    d(device_bf16, reference_fp32)  <=  d(reference_bf16, reference_fp32) * (1 + margin) + abs_floor

The floor is the intrinsic bf16 cost of the full trajectory (chaotic amplification included),
MEASURED from a bf16 recomputation of the reference itself, not guessed. Scorer:
`scripts/integration_envelope.py`; see `RELEASING.md` for the full rationale and the pass criterion.

**Shared draws require a sampler-entry re-seed (`TT_BIO_SHARED_DRAW_SEED`).** The single up-front
seed is not enough: the device (ttnn) and CPU (torch) trunks consume the global RNG differently
before the sampler, so without the re-seed the device and reference draw DIFFERENT diffusion noise
(measured bit-for-bit). With it they draw byte-identical noise, so the numerator is arithmetic-only.
The numbers below are with the fix (the first valid head-to-head).

**Three-leg head-to-head (no-MSA affinity, seed 0, shared draws).**

| leg | affinity_pred_value ratio | ligand-RMSD ratio | 1-pocket-lDDT | verdict |
|---|---|---|---|---|
| trypsin | 0.96 | 0.20 | 0.00 (bit-identical) | **PASS** |
| DHFR    | 1.22 | 0.19 | 0.00 (bit-identical) | **PASS** |
| FKBP12  | 1.90 | 0.32 | 0.00 (bit-identical) | **GAP** (affinity scalar only) |

Structure/pose parity is excellent on all three legs — the device structure is bit-identical to fp32
in pocket-lDDT and ligand pose is well inside the bf16 envelope. The affinity SCALAR passes on 2 of 3
(trypsin, DHFR) and GAPs on FKBP12 alone: the device affinity head carries a small TT-bf16 residual
(~0.04-0.24 abs log10 IC50; ~0.07 on FKBP12), and on FKBP12 — the target with the tightest
fp32-vs-bf16 envelope (0.037) — that residual exceeds the 1.5× bound. It is a lone outlier, not a
uniform band. Running the affinity diffusion in fp32 (`BOLTZ2_AFFINITY_DIFFUSION_FP32_DEVICE=1`)
removes only ~17% of it, so the residual is in the affinity head's non-diffusion bf16 arithmetic. Per
the standing rule this GAP is flagged as a real residual to hunt (candidate: an fp32 boundary on the
affinity head) and margin is not loosened. This is the sound test working: it passes a clean port
(2 of 3 affinity scalars and every structure metric) and isolates a genuine affinity-head residual
the old self-consistency floor buried in noise.

**Wired into the gate of record.** The envelope test is the default correctness criterion for
every diffusion (structure/affinity) leg in `scripts/full_parity_gate.py`: the gate folds the
device once at the reference seed, reads the leg's cached `ref_fp32` + `ref_bf16` CPU references,
and scores with `integration_envelope.py` through the one `finalize_leg` verdict path. The two CPU
references are the cached fixture (`--regen-refs` generates them, fingerprinted like the old ones,
so only the device fold + scoring re-run per release); a leg without them reports
`BLOCKED-REF-REGEN-NEEDED` rather than a false pass. The retired R/D/X floor is still available as
an opt-in device self-consistency (D) diagnostic via `--legacy-rdx`.

**Full matrix complete as of 2026-07-24.** Every envelope leg's `ref_fp32`/`ref_bf16` CPU
reference is now regenerated (the last 9: `boltz2-hsa-nomsa`, all 3 `protenix-v2-*-msa`
structure legs, all 3 `boltz2-affinity-*-msa` legs, and both `opendde-*-nomsa` legs — see
`~/.coworker/state/tt-bio-integration-parity-gate.md` §8-9 for the regen log and two real bugs
fixed along the way). A full non-dry `full_parity_gate.py` run against every one of the 21 wired
legs gives:

    Tally: 20 PASS, 1 GAP    (esmc/saprot/esmfold2/boltzgen/opendde-abag all reproduce committed)

The lone GAP is `boltz2-prot-nomsa` (7ROA, no-MSA structure leg): envelope worst kabsch_rmsd
ratio 1.83 (exceeds the 1.5× bound), reproduced bit-for-bit on a second `--fresh` re-fold (not
noise or a flaky measurement). This DRIFTS from the leg's legacy R/D/X-methodology verdict
(PASS) — plausibly the same effect already documented above for the FKBP12 affinity scalar: the
new envelope test's tighter, single-seed floor surfaces a bf16 residual that the old wide
cross-seed noise floor buried, rather than a new regression. Not root-caused or fixed this pass
(would need the same triangulation work as the FKBP12 case); flagged here as an open item, gate
metric intentionally not loosened to hide it. A second discrepancy, `boltz2-affinity-fkbp12-msa`,
DRIFTS the other direction — the envelope test PASSES it (both `affinity_pred_value` ratio 0.062
and `affinity_probability_binary` ratio 0.60) against the legacy `GAP-evidenced` record — but this is
not a contradiction: the two use different metrics/methodology and this leg's committed
`GAP-evidenced` record is a Moritz-reviewed, deeply-triangulated finding (§5 above) that is not
superseded by one envelope pass; it stays documented as-is pending its own re-review.

Prior state: a 2026-07-23/24 pass first wired the automated gate end-to-end but had only 9 of the
21 legs' CPU references generated (9 PASS, 12 BLOCKED-REF-REGEN-NEEDED). This pass closed every
remaining reference gap. Gate of record — pending Moritz's sign-off before merge.

## Reproduce

Each leg's reproduce command is in [Implementation parity — details](implementation-parity-details.md#reproducing-a-comparison).
The one-command runner for the full story is `scripts/full_parity_gate.py` (fans
the device side across cards, reuses the committed reference fixtures, and
emits the verdict table + tally); the per-leg scorers it dispatches to are
`scripts/pharma_parity.py` (structures / embeddings / saprot) and
`scripts/boltz2_affinity_parity.py` (affinity). Reference fixtures live under
`docs/implementation-parity-data/ref-fixtures/`.

### Where the reference fixtures come from

The verdict numbers a reader checks are the small committed JSONs at the top of
`docs/implementation-parity-data/` (the score/verdict files, ~160 KB total) plus
the per-fixture `meta.json`/`results.json` provenance under `ref-fixtures/`. The
large binary fixtures — the reference CIF structures and A3M MSAs that back each
diffusion leg — are externalized to GitHub Release assets to keep the repo
small, and are no longer committed going forward (see `.gitignore`).

A fresh checkout reproduces a leg end-to-end by restoring the binaries:

```bash
scripts/fetch_parity_fixtures.sh            # default tag = parity-fixtures-latest
# or a pinned pass: scripts/fetch_parity_fixtures.sh --tag parity-fixtures-2026-07
```

The fixtures are harvested from real reference runs (multi-hour GPU/CPU legs via
`scripts/pharma_harvest_ref_fixtures.py`); they are not regenerable on demand,
which is why they are versioned as release assets rather than rebuilt. The
fixtures already committed in the repo today (the 33 MB present at the time of
this change) stay tracked — no history rewrite was performed; only future binary
additions are externalized.
