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
| ESMFold2 | lysozyme, L129 | PASS | CA-RMSD 0.136 Å inside the 0.139 Å floor (X/floor 0.98); seed-wiring fix applied ([per-leg evidence](implementation-parity-details.md#per-leg-evidence)) |
| ESMFold2-Fast | trp-cage, L20 | PASS | 24-block checkpoint through the same harness; CA-RMSD 0.66 Å inside the 0.76 Å floor |
| ESMFold2-Fast | GB1, L56 | PASS | CA-RMSD 0.44 Å inside the 0.46 Å floor |
| ESMFold2-Fast | ubiquitin, L76 | PASS | CA-RMSD 1.46 Å against a 1.25 Å reference floor, inside that floor's 1.76 Å +1sd band; plddt_pcc 0.9976, distogram_pcc 0.9995 |
| ESMFold2-Fast | lysozyme, L129 | PASS | CA-RMSD 0.27 Å inside the 0.31 Å floor (X/floor 0.86); plddt_pcc 0.9997 |
| ESMFold2-Fast | cdk2x2, L256 | PASS | same CDK2 construct as the L512 row, so the two differ only in length; plddt_pcc 0.9939, distogram_rel_l2 0.045, ptm 0.9673 device against 0.9683 reference, same-seed plDDT gap 0.0033 against a 0.0024 reference seed spread |
| ESMFold2-Fast | cdk2x2, L512 | GAP-evidenced: structure PASS, confidence FAIL | the size the perf page publishes. Coordinates inside the floor (Kabsch 0.92 Å against 1.19 Å, distance-matrix 1-pcc 0.00080 against 0.00108, distogram_rel_l2 0.125 against a 0.25 bound), but mean plDDT 0.8987 device against 0.9197 reference on the same seed, a 0.0210 gap at 4.9x the reference's own 0.0043 seed spread, and ptm 0.7679 against 0.7980. Device-side and specific to the 24-block trunk: the 48-block row on the same fixture and stack agrees with its reference to 0.0005. Onset is between L256 and L512. `site/data/perf-512aa.json` keeps `parity_pending` on this row ([detail](esmfold2-e2e-parity.md)) |
| Protenix-v1 | 7ROA, L117, MSA | PASS | external CPU reference (upstream Protenix v0.5.0, fp32, the checkpoint's own 4 recycles); all-atom Kabsch RMSD X 1.714 Å inside the 1.840 Å reference noise floor (X/floor 0.93). Same target and the same MSA bytes as the Protenix-v2 row below, so the two differ only in checkpoint |
| Protenix-v2 | 7ROA, L117, MSA | PASS (legacy R/D/X); GAP-evidenced under the envelope gate | CA-RMSD 2.63 Å inside the 2.94 Å floor; confidence-head under-ranking shared with reference (model property). The envelope test GAPs this leg since v0.6.2 (numerator 1.168 Å vs a collapsed 0.042 Å envelope): the in-range AttentionPairBias unfusing shifts device trajectories at bf16 scale; root-caused in [the details](implementation-parity-details.md#envelope-gate-history-and-root-causes) |
| Protenix-v2 | ubiquitin, L76, MSA | PASS (legacy R/D/X); GAP-evidenced under the envelope gate | CA-RMSD 1.73 Å inside the 1.92 Å floor; passes on TM-score and CA-lDDT too. The envelope test GAPs this leg since v0.6.1 (numerator 2.015 Å vs a collapsed zero envelope): the in-range MSA row-chunking puts this target's 20826-sequence MSA on the chunked trunk path, measured non-bit-exact by design; root-caused as the path change, not a regression (see [the details](implementation-parity-details.md#envelope-gate-history-and-root-causes)) |
| Protenix-v2 | HSA, L585, MSA | PASS | on-device fp32 diffusion matches the reference's own fp32 boundary; CA-RMSD 0.685 Å inside the 0.695 Å floor (was GAP-evidenced in bf16, X 1.03 Å) |
| OpenFold3 | ubiquitin, L76, MSA | PASS | external CPU reference (official openfold3 0.4.4, fp32); all-atom RMSD X 1.46 Å inside the 1.64 Å reference noise floor (X/floor 0.89) |
| OpenFold3 | 7ROA, L117, MSA | PASS | external CPU reference; all-atom RMSD X 2.02 Å inside the 1.97 Å floor |
| OpenFold3 | 7XI5, L133, MSA + templates ON | PASS | external CPU reference; all-atom RMSD X 4.38 Å inside the 4.16 Å floor (X/floor 1.05); templates verified active (on/off structures differ up to 6.1 Å); 1-lDDT above its tighter floor ([per-leg evidence](implementation-parity-details.md#per-leg-evidence)) |
| OpenFold3 | 7XI5, L133, MSA, templates OFF | PASS at commit; GAP-evidenced since the fp32 diffusion boundary | external CPU reference; all-atom RMSD X 5.40 Å vs the 2.87 Å reference floor (was X 4.64 within a device-noise-inflated floor, D 3.76 → 0.70 under the intended fp32 boundary). All five device seeds land 0.59–0.61 Å vs the experimental structure, better than the CPU reference's own 0.42–0.90 Å spread, root-caused in [the details](implementation-parity-details.md#envelope-gate-history-and-root-causes) |
| OpenFold3 | 8HEL construct, L77, MSA, no templates | PASS | external CPU reference; all-atom RMSD X 5.52 Å inside the 7.59 Å floor (X/floor 0.73); 1-lDDT above its tighter floor ([per-leg evidence](implementation-parity-details.md#per-leg-evidence)) |
| OpenFold3 | 8HEL construct, L77, single-sequence | PASS | external CPU reference; all-atom RMSD X 11.31 Å within the floor (R 11.57, D 8.14; single-sequence de-novo helix is intrinsically ill-determined) |
| OpenFold3 | 9BK6 heterodimer (2xL~104/60), per-chain MSA | PASS | external CPU reference; all-atom RMSD X 1.663 Å within the 5-seed noise floor under the on-device fp32 diffusion boundary (`OF3_DIFFUSION_FP32_DEVICE`, default on, the Protenix HSA lever); bf16 diffusion missed this floor (X 1.889 Å); root cause and A/B in [openfold3-port.md](openfold3-port.md#precision); 1-TM and 1-lDDT above their tighter floors ([per-leg evidence](implementation-parity-details.md#per-leg-evidence)) |
| OpenBind-0 | ubiquitin, L76, MSA | PASS | external CPU reference (upstream aqlaboratory/openfold-3 v0.5.0 clone, fp32, OpenBind checkpoint); all-atom RMSD X 0.969 Å inside the 1.033 Å reference noise floor (X/floor 0.94), and 1-PCC, 1-TM and 1-lDDT inside theirs. Same sequence, same MSA bytes and same 200-step / 5-sample / 4-recycle settings as the OpenFold3 ubiquitin row above, so the two checkpoints are comparable leg to leg |
| OpenBind-0 | FKBP12 + SB3 (1FKG), L107 + 33 ligand atoms, MSA | PASS-caveated | external CPU reference; all-atom RMSD X 0.602 Å inside the 0.551 Å floor (X/floor 1.09), so `full_parity_gate.py` records it PASS on the gate metric. The residual is the ligand pose, not the fold: split under the same global superposition the protein half is X 0.582 Å against a 0.551 Å floor and the ligand half X 0.969 Å against the reference's own 0.524 Å ligand spread. 1-TM (1.21) and 1-lDDT (1.29) sit above their tighter floors. The device is 2.9× TIGHTER than the reference on the ligand (D 0.183 Å vs R 0.524 Å) since the CCD stereo fix; before it, our own reference conformers drew a random handedness per centre and the device's ligand spread was 0.630 Å, which inflated the floor and flattered the ratio. All 864 atoms pair between device and reference, the 33 SB3 atoms included ([per-leg evidence](implementation-parity-details.md#per-leg-evidence)) |
| Boltz-2 | trp-cage, L20, no MSA | PASS | wide no-MSA floor; absolute X 0.60 Å |
| Boltz-2 | 7ROA, L117, no MSA | PASS (legacy R/D/X); GAP-evidenced under the envelope gate | wide no-MSA floor (R 4.98 Å); absolute X 4.21 Å. The tighter envelope test GAPs this leg at the pinned seed (ratio 2.04); root-caused as seed-0 chaotic-trajectory amplification, not a precision bug (see [the details](implementation-parity-details.md#envelope-gate-history-and-root-causes)) |
| Boltz-2 | 7ROA, L117, MSA | PASS | CA-RMSD 0.94 Å inside the 0.81 Å floor |
| Boltz-2 | ubiquitin, L76, MSA (production default) | PASS | all 4 metrics within the tight MSA-backed GPU-reference floor (CA-RMSD X/floor 1.03, 1-lDDT X/floor 0.97); residual systematic bf16 ([same-seed diagonal](implementation-parity-details.md#1-same-seed-diagonal-shared-rng-proof)) |
| Boltz-2 | HSA, L585, no MSA | PASS | CA-RMSD 1.47 Å inside the 1.50 Å floor; first L585 target |
| Boltz-2 | 9NCY AbAg complex (3 chains, 505 tokens), no MSA | GAP-evidenced (envelope gate) | shared-draws envelope: CA-RMSD numerator 6.71 Å vs bound 2.539 Å on qb1 (ttnn 0.67.4); the 0.760 Å PASS numerator was recorded on pc (ttnn 0.68.0) at fixture birth and the whole commit range since is byte-inert on qb1. Root-caused 2026-08-16 as stack-sensitive basin selection on a chaotic no-MSA target, not a code regression (device, bf16 ref, fp32 ref land in equally accurate 6.4-6.6 Å basins vs ground truth, TM 0.92-0.94) (see [the details](implementation-parity-details.md#envelope-gate-history-and-root-causes)). First leg inside the former [385,506]aa L1 crash band (fixed by c06bd76cf); campaign-relevant size (AbAg-XM median 509) |
| Boltz-2 (affinity) | FKBP12 + SB3, L107, no MSA (non-default) | PASS (legacy R/D/X); PASS under the envelope gate | device-fp32 hybrid diffusion vs the GPU bf16 reference: pocket-lDDT X 0.014 within the GPU noise floor (X/floor 1.25); affinity scalar, affinity probability, and ligand-pose RMSD also pass (X/floor 0.79 / 1.38 / 0.92). Under the envelope gate (gate of record) the affinity scalar passes at ratio 1.35 (numerator 0.0505 vs bound 0.0660) and affinity_probability at 3.65 ([bf16-floor evidence](implementation-parity-details.md#why-every-non-pass-is-a-bf16-backend-floor-not-a-port-defect)) |
| Boltz-2 (affinity) | FKBP12 + SB3, L107, MSA (production default) | PASS-caveated | Under the envelope gate (gate of record) the affinity scalar passes at ratio 0.32 (numerator 0.0456 vs bound 0.226) and affinity_probability at 0.18; pocket-lDDT GAPs under legacy R/D/X (X/floor 4.48), the same narrower-basin systematic-bf16 property as the other affinity legs (seed-independent [same-seed diagonal](implementation-parity-details.md#1-same-seed-diagonal-shared-rng-proof)). The legacy R/D/X scalar "GAP" (X/floor 2.27) was a stale-fixture artifact, measured against pre-shared-draws refs before the fixture regen, and is superseded by the envelope pass |
| Boltz-2 (affinity) | DHFR + MTX, L187, no MSA (non-default) | PASS-caveated | affinity scalar and ligand-pose RMSD pass (X/floor 0.68 / 1.36); pocket-lDDT GAPs (4.72), proven a genuine bf16-BACKEND floor by three-backend triangulation (GPU-bf16 and CPU-bf16 references disagree on the pocket by the same ~0.13 lDDT margin the device does), not a port defect |
| Boltz-2 (affinity) | DHFR + MTX, L187, MSA (production default) | PASS-caveated | affinity scalar (1.32), affinity_probability (0.95), and ligand-RMSD (1.61) PASS; pocket-lDDT GAPs (13.35), systematic bf16 by the [same-seed diagonal](implementation-parity-details.md#1-same-seed-diagonal-shared-rng-proof) |
| Boltz-2 (affinity) | trypsin + BAM, L223, no MSA (non-default) | PASS-caveated (legacy R/D/X); GAP under the envelope gate | affinity scalar and ligand-pose RMSD pass under legacy R/D/X (X/floor 0.94 / 0.95); pocket-lDDT GAPs (10.13), proven a genuine bf16-BACKEND floor by three-backend triangulation (GPU-bf16 vs CPU-bf16 pocket-lDDT X/floor 7.51, both NO), not a port defect. Under the envelope gate (gate of record) the affinity scalar GAPs at ratio 2.81 (numerator 0.0537 vs bound 0.0387): the device predicts 2.552 where the CPU reference gives 2.606, about 2%. Pose is unaffected (ligand-RMSD ratio 0.11, pocket-lDDT bit-identical). Pre-existing, un-masked in v0.6.5 when the reference was regenerated on the conformer-seeding fix; see [the bar move](implementation-parity-details.md#the-trypsin-affinity-bar-moved-in-v065) |
| Boltz-2 (affinity) | trypsin + BAM, L223, MSA (production default) | PASS-caveated | affinity scalar (0.79), affinity_probability (0.92), and ligand-RMSD (0.78) PASS; pocket-lDDT GAPs (2.75), systematic bf16 by the [same-seed diagonal](implementation-parity-details.md#1-same-seed-diagonal-shared-rng-proof) |
| OpenDDE | trp-cage, L20, no MSA | PASS (legacy R/D/X); GAP-evidenced under the envelope gate | CA-RMSD 0.51 Å inside the 0.52 Å floor. The envelope test GAPs this leg since v0.6.2 (numerator 0.198 Å vs a collapsed zero envelope), same AttentionPairBias-unfusing root cause as the other v0.6.2 GAPs, root-caused in [the details](implementation-parity-details.md#envelope-gate-history-and-root-causes) |
| OpenDDE | 7ROA, production | PASS (legacy R/D/X); GAP-evidenced under the envelope gate | wide device-dominated floor (D 6.04 Å); absolute X 4.67 Å. The envelope test GAPs this leg since v0.6.2 (numerator 2.149 Å vs a collapsed zero envelope; still inside the 6.04 Å floor and below the committed cross term), same root cause, root-caused in [the details](implementation-parity-details.md#envelope-gate-history-and-root-causes) |
| OpenDDE-abag | 1AHW Ab–Ag | PASS | global DockQ 0.864; per-interface iRMSD 0.65/0.70/1.20 Å, all sub-Å-to-low-Å |
| BoltzGen | binder vs 7ROA chain A | PASS | designability 93.8% (≤2 Å scRMSD) vs reference 68.75%; device meets-or-exceeds |
| SaProt-35m | ubiquitin, L76 | PASS | deterministic encoder; emb PCC 0.99914, in the ESMC band |
| SaProt-650m | ubiquitin, L76 | PASS | deterministic encoder; emb PCC 0.99964, in the ESMC band |
| RFdiffusion3 | IAI protein motif-scaffold, I40/L419 | PASS | host featurizer 43/43 `f` keys bit-exact vs the committed upstream foundry reference capture; card-free, in-process (`scripts/rfd3_port/parity_gate.py`) |
| RoseTTAFold3 | ubiquitin, L76, MSA | PASS | vendored torch CPU reference, bf16 both sides, shared draws; CA-RMSD X 0.221 Å inside the 0.418 Å reference noise floor (X/floor 0.53) and all-atom X 0.219 Å inside a 0.467 Å floor (X/floor 0.47). Four of the five seeds reproduce the reference trajectory to 0.087-0.097 Å; seed 4 alone reads 0.736 Å and it is the reference that moved, its own sample sitting 0.86-0.91 Å from its siblings while the device’s stays inside the normal device spread: a bifurcation, like `boltz2-prot-nomsa` at seed 0. `scripts/rf3_port/accuracy_cell.py`, five seeds, shipped arm (10 recycles, 50 sampling steps, one diffusion sample). The shipped arm is now the fused SDPA with the ragged key tail masked (`fp32_softmax=False`), which reads 0.221 Å against the materialised route’s 0.229 Å, and 0.0920 Å against 0.0955 Å on the four non-bifurcating seeds |
| RoseTTAFold3 | 7ROA, L117, MSA | PASS | same harness and convention as the row above; CA-RMSD X 0.178 Å inside the 0.382 Å noise floor (X/floor 0.47) and all-atom X 0.229 Å inside a 0.632 Å floor (X/floor 0.36). No outlier seed here: all five X land 0.128-0.254 Å, so the ubiquitin leg’s seed-4 bifurcation is a per-seed trajectory property and not a length- or target-independent one. This is the rung the shipped-arm flip moves most: the materialised route read 0.203 Å against a 0.361 Å floor, so masking the ragged tail cut the error 1.14x while the floor widened to the device’s own spread. The unmasked fused route reads 1.64 Å here, outside the floor, which is why the pad and the arm ship together and never separately |
| RoseTTAFold3 | 7EIP, L997, MSA | PASS | the 1024 aa rung's absolute cell, on a real target instead of the tiled-CDK2 chimera every other ~1000-residue artifact in this repo is built from: chondroitin sulfate ABC endolyase I, 1.88 Å X-ray, R/R-free 0.1711/0.2128, one chain, monomeric assembly, ligand-free, folded as the mature enzyme (SEQRES 25-1021). Same harness and convention as the two rows above; the reference ran on a rented H200 (`--ref-device cuda`; the sampler and the RNG stay on the host, so the recorded draws are the same stream the device half re-harvests). CA-RMSD X 0.641 Å inside a 2.357 Å floor (X/floor 0.27) and all-atom X 0.638 Å inside a 2.260 Å floor (0.28). Read the ratio carefully: X grew 3.6x from the 7ROA row while the floor grew 6.2x, because the floor is the model's own five-seed spread and that widens with length. The cell is not tighter here, the model is looser. The absolute reading is the one that settles it: against the deposited coordinates over the 966 modelled residues the device sits 1.890 ± 0.153 Å and the H200 reference 1.951 ± 0.174 Å, so the port is no further from the experiment than the reference it replaces. 997 mod 32 = 5, so this is also the first RF3 anchor where the shipped arm's ragged key-tail masking does any work: 968 padded calls, all at `tri_att`, counted in the run and not argued from the flag, on a rollout an independent re-run reproduced bit-for-bit. The `rf3-1024aa` leg of `scripts/full_parity_gate.py` re-folds seed 0 of this row every release and fails the gate if the device drifts past 4.0 Å from the crystal. |
| RoseTTAFold3 | CDK2 ladder 128/256/512/768/1024 aa, plus 5vht (real homodimer, paired MSA, atomized NCAA) | PASS | supporting evidence for the row above, not a second absolute number: scored against the vendored torch reference, ceiling-relative with the bf16-vs-fp32 ceiling measured in the same run; no trend with size. Confidence reductions (pTM / ipTM / ranking) 12/12 against upstream's own code. Bit-exact run-to-run AND cross-process at 128 / 256 / 512. The pairformer s-track defect is fixed: `ttnn.softmax` returns rows summing to 0.9769 rather than 1, and an RF3-scoped accurate softmax takes the s-track from 0.021263 to 0.003290 against a 0.001869 reference (11.4x -> 1.76x); opt-in, not flipped for shared sites. The 1024 aa accuracy cell is measured, on 7EIP and not on this construction; see the row above. The shipped arm is the fused SDPA with the ragged key tail masked; at 128 aa the five-seed result is bit-identical across a `main` merge that touched the fused-SDPA route and across two different cards, and at 512 aa the CIF digest reproduces across two processes. `TT_BIO_TRIATT_FUSED_HIFI` is not additive with this arm and is not used: it lives on the materialised-softmax route, which is the other side of the same switch. |
| Nesso-1 | tyr48 + tyrosine, 61 tokens | PASS | host featurizer bit-exact vs the committed upstream capture; torch model 22/22 activations bit-identical and 11/11 output scalars exact. On device the worst of the 11 scalars sits at 3.43x upstream's own 0.058 featurization-draw spread, inside the 5.0x floor, and the device is deterministic (spread 0.0 across 3 repeats). On DAVIS the within-target Pearson is 0.662 against the upstream/H200 arm's 0.636 on the same 30 compounds per target |
| PXDesign | PD-L1 quick-start target, 196 tokens, design featurizer | PASS | 25/25 arms bit-exact vs a committed capture of the upstream featurizer, over the conditioning template, the `xpb` binder-placeholder exclusion, the 36-way restype and the whole 18-key model input built from the target file. One arm scores the capture rather than the port, so a fixture whose conditioning coordinates are wrong fails the leg instead of agreeing with it; card-free, in-process (`scripts/pxdesign_port/parity_gate.py`) |
| PXDesign | LacZ-C 768aa target + 80-mer binder, 848 tokens, generated coordinates | PASS | design coordinates bit-exact across all six warm repetitions of three independent device legs, `structure_sha16 cd80f8e274306706`, including across the fp32-softmax L1-padded plan arm, because the lever changes the shard's extent and not the arithmetic |
| AF2-IG (PXDesign filter) | LacZ-C 128aa + 80-mer binder, host featurizer | PASS | 60/60 featurizer keys bit-exact vs a captured ColabDesign forward pass, over both the complex and monomer stages; card-free, in-process (`scripts/af2_port/parity_gate.py`) |
| AF2-IG (PXDesign filter) | same fixture, torch trunk, complex stage (templates on) | PASS | 94/94 activation taps and all 6 filter scalars inside their bars vs the captured JAX taps; card-free, needs `params_model_1_ptm.npz` |
| AF2-IG (PXDesign filter) | same fixture, torch trunk, monomer stage (`model_3_ptm`) | PASS-caveated | 90 taps, 0 failed, 32 adjudicated inside the reference own float32-vs-bfloat16 envelope: on those taps the reference sits closer to our bfloat16 arm than our float32 arm does |
| AF2-IG (PXDesign filter) | same fixture, ttnn trunk on card, complex stage | GAP-evidenced | 9 of 94 taps and 3 of 6 scalars miss (worst envelope ratio 12.3; pLDDT 0.0028, pTM 0.0020, pAE 0.0021). A bfloat16 realisation floor in the trunk residual chain, amplified about 3x by the structure module; three trunk-precision levers were measured dead against it. The leg gates that floor instead of a PASS bar (`scripts/af2_port/device_floor.py`): GAP only while the failing taps, their envelope ratios and the scalar deltas all reproduce the committed record within 1.10x, and both `tap_gate.py --mutate` controls break every one of those conditions. Bit-identical on qb1 cards 3 and 0 |

**Tally: 37 PASS, 7 PASS-caveated, 2 GAP-evidenced.** The two GAP-evidenced legs are
`boltz2-9ncy-nomsa`, root-caused in [the details](implementation-parity-details.md#envelope-gate-history-and-root-causes), and `af2ig-trunk-device`, a
measured bfloat16 floor the leg gates as such. Rows that read "GAP-evidenced under the envelope
gate" pass the legacy R/D/X test and fail the tighter envelope test described below; each has its
root cause in [the details](implementation-parity-details.md#envelope-gate-history-and-root-causes).

The three Boltz-2 affinity legs run with an MSA, which is Boltz-2's production default. The
earlier single-sequence rows are kept and labelled `non-default`. Across their 12 metric cells the
MSA legs score 9 PASS and 3 GAP. The GAP is 1-pocket-lDDT on all three targets, the same
bf16 property the no-MSA legs show ([bf16-floor evidence](implementation-parity-details.md#why-every-non-pass-is-a-bf16-backend-floor-not-a-port-defect)),
and all three MSA affinity scalars pass the envelope gate (FKBP12 0.32, DHFR 1.32, trypsin 0.79).
The no-MSA affinity rows use the device-fp32 hybrid diffusion path. FKBP12 passes there
(pocket-lDDT X 0.011 within the 0.011 GPU floor, X/floor 0.94). DHFR and trypsin are
PASS-caveated because three-backend triangulation shows a bf16 backend floor: the pinned GPU-bf16
and CPU-bf16 references disagree on the pocket by the same ~0.09-0.13 lDDT the device does
([affinity-scalar triangulation](implementation-parity-details.md#2b-three-backend-triangulation-on-the-affinity-scalar-cross-backend-divergence-absorbed-by-the-envelope-gate)).
Every Boltz-2 leg was re-measured with the seed-wiring fix live on 2026-07-21 and every verdict
held. Protenix-v2 HSA was GAP-evidenced under bf16 diffusion; running its diffusion sampler in
fp32 on device, the reference's own precision boundary, closed it to a PASS.

The measured R/D/X table and per-leg evidence are in
[Implementation parity: details](implementation-parity-details.md).

## How a leg is scored

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

### The envelope test, which is the gate of record

The R/D/X floor compares independent stochastic samples against a guessed floor `max(R, D)`, so a
difference in X mixes real arithmetic divergence with ordinary sample-to-sample chaos: a correct
port can fail because it landed in a different noise basin, and a subtle bug can hide under a loose
floor. The PASS-caveated and GAP-evidenced verdicts are where that floor could not separate the
two. The envelope test replaces it for every diffusion (structure and affinity) leg in
`scripts/full_parity_gate.py`.

A diffusion model is a deterministic function of its input noise, so the gate feeds byte-identical
noise to three closed-loop runs: `device_bf16` (the card), `reference_fp32` and `reference_bf16`
(both tt-bio's own CPU torch path with `--no_kernels`, the second under `TT_BIO_REF_BF16=1`). One
`--seed` is not enough to share draws, because the ttnn trunk and the torch trunk consume the
global RNG differently before the sampler. `TT_BIO_SHARED_DRAW_SEED`, set on all three runs,
re-seeds in `AtomDiffusion.sample` right before the first `torch.randn`, and the three then draw
bit-identical noise. Per leg and per metric `d`:

    d(device_bf16, reference_fp32)  <=  d(reference_bf16, reference_fp32) * (1 + margin) + abs_floor

The right-hand side is the bf16 cost of the whole trajectory, chaotic amplification included,
measured from a bf16 recompute of the reference rather than guessed. The scorer is
`scripts/integration_envelope.py`; `RELEASING.md` has the full rationale and pass criterion.

The gate folds the device once at the reference seed and scores it against the leg's cached
`ref_fp32` and `ref_bf16` references (`--regen-refs` generates them). A leg without them reports
`BLOCKED-REF-REGEN-NEEDED` rather than passing, and a run where every leg is blocked prints
`GATE INCONCLUSIVE` and exits nonzero. The retired R/D/X floor is still available as a device
self-consistency diagnostic through `--legacy-rdx`.

The envelope needs a torch path that can be recomputed in fp32 and in bf16, which ttnn-only ports
do not have. Their legs (Protenix-v2, OpenDDE and OpenFold3 structure legs) carry `legacy_rdx` and
are scored R/D/X against their official upstream implementations. `predict` refuses
`--accelerator cpu/gpu` for a model with no torch path, `--regen-refs` refuses to write a
reference pair whose two arms are byte-identical, and a report whose every metric has a zero
envelope reads `NO-DATA`, never `PASS`.

Nesso-1 is the one model scored on neither a distance nor an envelope. It predicts no
coordinates, so there is nothing to align and no diffusion draw to share; its leg compares the
eleven output scalars against the torch reference and normalises by upstream's own
featurization-draw spread, because upstream roto-translates every conformer off the global RNG and
so differs from itself run to run by up to 0.058.

The dated history of the gate (fixture regenerations, collapsed envelopes, the 2026-08-11
`--accelerator cpu` fix) and the root cause of every open GAP are in
[Envelope gate history and root causes](implementation-parity-details.md#envelope-gate-history-and-root-causes).

## Measurement bounds and non-gated variants

Two facts a skeptical reader should have that do not appear as a verdict row
above. Both are recorded here because this doc, not the public JapanFold
accuracy page, is where the full detail belongs.

**BoltzGen designability carries a sampling bound.** The leg is n=16 per side
(two batches of 8, `docs/implementation-parity-data/boltzgen.json`), and the
reference's own two batches are 12.5 points apart on the ≤2 Å bar (batch_a 75%,
batch_b 62.5%). So part of the 93.75%-vs-68.75% margin is sampling noise, not
port quality. What the leg establishes is the direction, and the direction holds
across both batch pairings: device median scRMSD 0.78 Å vs reference 1.05 Å, and
device pass-rate on the favorable side of the reference's spread in every
pairing. BoltzGen designs new sequences, so there is no 1:1 correspondence to
score, so parity is in the designability distribution, not a per-design RMSD.

**SaProt-1.3B is served but is not a clean PASS.** On ubiquitin (L76) it measures
per-residue embedding PCC 0.995076 and MLM-logits PCC 0.998952 (R = D = 1.00000,
deterministic), which lands just below the 0.9987–0.9996 band the 35M and 650M
variants hit, so `docs/saprot-parity.md` records it as a near-pass and claims no
PASS row. The residual tracks depth rather than a port defect: 1.3B is the 650M
width at twice the layers (66 vs 33), so it accumulates about twice the bf16
rounding. It has no leg in `full_parity_gate.py` and is therefore absent from
the tally above; the 650M leg is the gated SaProt path.

**Boltz-2 coordinates move under the fused bias stacks, by design.**
`TT_BIO_FUSE_BIAS_STACKS` is on by default. It folds each LayerNorm affine into
the projection behind it in the diffusion conditioning, so the pair tensor is
normalised once per fold instead of once per layer. The algebra is exact in real
arithmetic and not in bfloat16, so the structure shifts: 0.2198 A all-atom on the
cdk2x2 298 aa control against that control's 0.35 A bar, measured identically on
Blackhole and on Wormhole, with plDDT moving 0.0026. At 512 aa it costs 0.000997
plDDT. Both are far inside the envelope every Boltz-2 leg above is scored against,
and the gate's structure and affinity legs are run with the flag on.
`TT_BIO_FUSE_BIAS_STACKS=0` restores the per-layer stack. The two granularity
levers that shipped with it, `TT_BIO_SDPA_ADD_GRANULARITY` and
`TT_BIO_GATE_GRANULARITY`, are bit-exact at every granularity and cannot move a
coordinate.

## Reproduce

Each leg's reproduce command is in [Implementation parity: details](implementation-parity-details.md#reproducing-a-comparison).
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
large binary fixtures (the reference CIF structures and A3M MSAs that back each
diffusion leg) are externalized to GitHub Release assets to keep the repo
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
this change) stay tracked; no history rewrite was performed; only future binary
additions are externalized.
