# ESMFold2 end-to-end real-weight parity (on-hardware)

The ttnn on-device ESMFold2 pipeline vs the **vendored** torch reference
(`tt_bio._vendor.esmfold2_hf` — no external repo/clone needed), on the real
`biohub/ESMFold2` weights, with the ttnn ESMC-6B producing the language-model
hidden states fed to **both** paths (so this isolates the ESMFold2 neural port
from the separately-validated ESMC port and from featurization).

Reproduce:

```bash
TT_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU \
  python scripts/esmfold2_e2e_parity.py --proteins trpcage --steps 20 --loops 3 \
    --out /tmp/ef2_parity.json
```

## Result — trpcage (L=20), Blackhole (pc card 0), normal (non-fast) path

| metric | value | reading |
|---|---|---|
| `plddt_pcc` | **0.9989** | per-residue confidence — the metric ESMFold ranks on |
| `plddt_mae` | 0.0031 | mean pLDDT 0.824 (tt) vs 0.821 (ref) |
| `distogram_pcc` | **0.9997** | pairwise distance-bin logits |
| `distogram_rel_l2` | **0.053** | gated: the harness fails the leg above 0.25. PCC is scale-blind, so this relative-L2 bound is the actual distogram anchor |
| `coord_dm_pcc` | 0.928 | alignment-free atom-atom distance matrix |
| `kabsch_rmsd` | 2.15 Å | tt-vs-ref, after weighted rigid alignment |
| `ref_selfvar_rmsd` | 1.98 Å | reference's **own** two-seed sample-to-sample spread |
| `ptm` | 0.251 (tt) / 0.247 (ref) | predicted TM-score |

## Verdict

**Pass.** pLDDT and distogram match (PCC ~0.999, rel L2 0.05) and pTM matches.
The tt-vs-ref
coordinate RMSD (2.15 Å) is within the reference's own sample-to-sample
variance (1.98 Å, two torch seeds) — the spread is intrinsic diffusion
stochasticity (independent RNG streams), not a port error, so coords are
compared alignment-free (distance matrix) and against that variance baseline
rather than element-wise. No accuracy regression from the ttnn port.

Diffusion is not bit-identical across the torch and ttnn samplers by design;
that is why the coordinate comparison is variance-relative, mirroring the
Boltz-2 `--fast` parity methodology (`docs/boltz2-fast-parity.md`).

## Result — ESMFold2-Fast (L=20 to L=129), Blackhole (qb2 card 1), non-fast path

Same harness, same targets, same metrics, on the 24-block `biohub/ESMFold2-Fast`
checkpoint (sha256 `60ca19f2…`, no MSA encoder). Trunk depth is read from the
checkpoint config on both sides, so one script covers both released depths:

```bash
TT_VISIBLE_DEVICES=1 MKL_THREADING_LAYER=GNU PYTHONPATH=$PWD \
  python scripts/esmfold2_e2e_parity.py --checkpoint esmfold2-fast \
    --proteins trpcage,gb1,ubiquitin,lysozyme --seeds 0,1,2,3,4 \
    --out docs/implementation-parity-data/esmfold2-fast.json
```

| target | `plddt_pcc` | `plddt_mae` | `distogram_pcc` | `distogram_rel_l2` | `kabsch_rmsd` | noise floor | ratio |
|---|---|---|---|---|---|---|---|
| trp-cage, L20 | 0.9989 | 0.0061 | 0.9997 | 0.041 | 0.66 Å | 0.76 Å (device) | 0.88 |
| GB1, L56 | 0.9995 | 0.0017 | 0.9997 | 0.044 | 0.44 Å | 0.46 Å (device) | 0.96 |
| ubiquitin, L76 | 0.9976 | 0.0034 | 0.9995 | 0.048 | 1.46 Å | 1.25 Å (reference) | 1.17 |
| lysozyme, L129 | 0.9997 | 0.0008 | 0.9997 | 0.039 | 0.27 Å | 0.31 Å (device) | 0.86 |

Five sampler seeds per target on both backends. `kabsch_rmsd` is the mean over the 25
device-vs-reference pairs. The noise floor is the larger of the two self-consistency
means, and the bracket says which side set it: on three of four targets the device
samples spread wider than the reference's, so the floor is the device's own. The pass
band is that floor plus one standard deviation of the wider side, the same
`noise_floor_verdict` criterion every parity leg in this repo uses
(`scripts/pharma_parity.py`). Ubiquitin is the one ratio above 1.0, and it sits well
inside its band: 1.46 Å against 1.76 Å, because the reference's own seed-to-seed spread
on this target has a standard deviation of 0.51 Å.

Mean-plDDT agreement between the two backends on the same seed is 0.0045 (L20), 0.0024
(L56), 0.0029 (L76) and 0.0005 (L129). It does not grow with length over this range.

**Pass.** Every metric is at or better than the 48-block leg measured the same way
(`docs/implementation-parity-data/esmfold2.json`: `plddt_pcc` 0.995-0.999,
`distogram_pcc` 0.999, Kabsch inside the floor on 3 of 4 targets against 4 of 4 here).
The alignment-free distance-matrix metric leaves the floor on ubiquitin, the same
target and the same way it does on the 48-block leg, which is why the leg's verdict
reads the Kabsch floor and reports the distance-matrix count rather than gating on it.

## Result — ESMFold2-Fast at L=512, the size the perf page publishes

The L20-L129 leg above passes, so the leg was re-run at the fixture the public perf page
measures its ESMFold2-Fast cell on. Same harness, same checkpoint, same protocol the page's
own cell uses: 10 recycles, 100 requested sampler steps (68 executed after the sigma_max=256
clip), qb2 card 1, non-fast path.

```
  python scripts/esmfold2_e2e_parity.py --checkpoint esmfold2-fast \
    --proteins cdk2x2_512 --loops 10 --steps 100 --seeds 0,1
```

The length ladder below is the same line with `--proteins cdk2x2_128` and `cdk2x2_256`.
Records: `docs/implementation-parity-data/esmfold2-fast-{512aa,cdk2x2_256,cdk2x2_128}.json`.

| metric | value | floor / band | verdict |
|---|---|---|---|
| `kabsch_rmsd` | 0.92 Å | 1.19 Å (device) | inside |
| `coord_dm_pcc` (1-pcc) | 0.00080 | 0.00108 (device) | inside |
| `distogram_pcc` | 0.9942 | — | — |
| `distogram_rel_l2` | 0.125 | 0.25 (gate bound) | inside |
| `plddt_pcc` | 0.9900 | — | — |
| `plddt_mae` | 0.0234 | — | — |
| mean plDDT, same seed | 0.8987 device / 0.9197 reference | 0.0043 (reference seed spread) | **outside, 4.9x** |

Geometry passes, confidence does not. The two backends place the same atoms: Kabsch and the
alignment-free distance-matrix metric both sit inside the device's own seed-to-seed floor, and
`distogram_rel_l2` is half the gate bound. But the device reads mean plDDT 0.8987 where the
reference reads 0.9197 on the same seed, a 0.0210 gap against a reference seed spread of 0.0043
and a device seed spread of 0.0003. That is 4.9x the wider of the two noise floors, so it is not
sampler variance. `ptm` moves the same way, 0.7679 device against 0.7980 reference.

**Floor, not a pass.** The 24-block port loses confidence, not structure, and only at length.

### What this settles about the perf page's 0.017

The page's ESMFold2-Fast cell carried an open note: its plDDT 0.8987 sat 0.017 below the three
NVIDIA folds' 0.9148-0.9155, where the ESMFold2 row's two sides agree to 0.0005. Two readings
were open, a difference in the NVIDIA stack (those cells ran `esm` 3.4.0 where the ESMFold2
row's ran 3.3.0) or a real port gap on the 24-block trunk.

It is the port. The vendored CPU fp32 torch reference, run here on the same fixture with the
same featurization and the same shared ttnn ESMC-6B hidden states, reads 0.9197. It lands 0.004
above the NVIDIA band where the device lands 0.016 below it, so an independent implementation on
our own box, with no `esm` release anywhere in the path, agrees with NVIDIA to within a quarter
of the disputed gap. Two independent references agreeing rules out the NVIDIA stack; the device
side is the odd one out.

The gap turns on with length, and the onset sits between L=256 and L=512. The L20-L129 leg
above uses four different monomers, so on its own it confounds length with target. The perf
page's own `cdk2x2_N` fixtures remove that: they are one CDK2 tandem construct truncated to a
length ladder, so N moves and the sequence family does not. Run at the same protocol on the
same box:

| `cdk2x2_N` | L=128 | L=256 | L=512 |
|---|---|---|---|
| same-seed mean-plDDT gap | target unusable | 0.0033 | **0.0210** |
| reference seed spread | 6.94 Å Kabsch | 0.0024 | 0.0043 |
| `plddt_pcc` | — | 0.9939 | 0.9900 |
| `distogram_rel_l2` | 0.365 | 0.045 | 0.125 |
| `ptm`, device vs reference | — | 0.9673 / 0.9683 | 0.7679 / 0.7980 |

L=128 is excluded on the reference's own evidence, not the device's: truncating CDK2 to 128
residues cuts the domain in half, and the reference folds it to a different structure on every
seed, 6.94 Å Kabsch between its own two seeds. Nothing can be measured against a reference that
does not converge. These fixtures were built to time folds at a given length, where sequence
content does not matter, so the short rungs are not automatically valid accuracy targets.

L=256 is a real target and the port tracks it: plDDT gap 0.0033 against a 0.0024 reference seed
spread, `ptm` agreeing to 0.001, `distogram_rel_l2` 0.045. Between there and L=512 the plDDT gap
grows 6x to 0.0210 and `ptm` opens from 0.001 to 0.030, while the coordinates stay inside the
noise floor.

The 48-block ESMFold2 row is the control that confines this to one checkpoint: on the same
512aa fixture, the same box and the same shared stack (ttnn ESMC-6B, featurization, diffusion
sampler), it reads 0.9285 against NVIDIA's 0.929, agreeing to 0.0005. So the shared
infrastructure is not what loses 0.021, and depth-wise error accumulation is not the mechanism
either: the checkpoint that diverges is the one with half as many trunk blocks.

`site/data/perf-512aa.json` therefore keeps `parity_pending: true` on the ESMFold2-Fast row.
The structural claim the row makes is sound and now measured; the confidence number it prints
is 0.021 low against this repo's own reference, and that is a real open defect, not a
measurement artefact.

### Where the 0.021 comes from

The gap is the folding trunk's pair state. Not the confidence head, not the sampled structure,
not the featurization.

The confidence head is host fp32 in the port except its own 4-block pair trunk, so the gap has
to enter through one of the head's inputs. Each input can be substituted with the reference's
value, and `scripts/esmfold2_e2e_parity.py` does exactly that on one device fold at cdk2x2_512,
seed 0 (`--conf_ab` plus `--trunk_probe`, `--z_sens`, `--x_swap_pdb`, `--inputs_swap`,
`--ref_z`). Device baseline 0.897708 on qb2 card 0:

| substituted input | probe | mean plDDT | move |
|---|---|---|---|
| nothing (baseline) | — | 0.897708 | — |
| the head's own pair trunk | ttnn -> reference fp32 | 0.897731 | +2.2e-5 |
| `s_inputs` + relpos + token bonds | reference values | 0.897647 | -6.1e-5 |
| `x_pred` | NVIDIA reference structure, 0.65 Å away | 0.897070 | -6.4e-4 |
| `z` | **reference trunk's own pair state** | **0.914742** | **+0.017034** |

Substituting `z` alone recovers 0.017 of the 0.024 gap and lands the device pipeline inside the
0.9148-0.9171 the three NVIDIA folds report. `ptm` moves 0.7560 -> 0.7822 against the
reference's 0.7980. The head itself is exact: on identical inputs the device's head and the
reference's head agree to 8e-6 (0.9147418 against 0.9147503).

The pair state differs by 12.1 % relative L2, and that is what ordinary bf16 arithmetic
accumulates to: one 24-block trunk pass on device sits 3.4 % from the same pass in host fp32,
and the recurrence runs four of them (`total_steps = num_loops + 1`), so the error adds
coherently rather than in quadrature. Trunk matmuls already run HiFi4 with
`fp32_dest_acc_en=True`, so the L-length triangle contraction is fp32-accumulated; what is left
is bf16 operand precision inside each block.

Two things make this specific to this checkpoint rather than to bf16 in general.

**The direction of the error matters far more than its size.** Perturbing `z` by the same 12.1 %
in a random direction costs 3.3e-4, 1/52 of what the port's own error costs. The port's error is
structured, not noise: the device's pair state is also uniformly ~5 % smaller than the
reference's across its whole distribution (rms 29.00 against 30.37, median 10.75 against 11.42,
p99 91.0 against 95.6).

**The 48-block checkpoint carries more pair error and does not lose confidence.** Its trunk sits
5.6 % from host fp32 per pass against the 24-block's 3.4 %, at both L=256 and L=512, yet its
plDDT agrees with its reference to 0.0005. Pair-state precision alone therefore cannot predict
the deficit; the two checkpoints' confidence heads read the same pair error differently. Their
coordinate sensitivities differ the same way: 0.5 Å of displacement costs the 24-block head
0.127 of plDDT and the 48-block head 0.024.

So there is no wrong formula here to correct. Closing the gap means running the 24-block trunk's
pair state at higher precision than bf16, which is a performance and memory change, not a bug
fix, and the 48-block checkpoint does not need it. `site/data/perf-512aa.json` keeps
`parity_pending: true` on the ESMFold2-Fast row.

Two measurement notes for anyone repeating this. ESMFold2's representative atom is CB, with CA
only as the glycine fallback (`compute_representative_atoms`); substituting CA for every residue
feeds the head a CA-CA distance map where it expects CB-CB and reads mean plDDT 0.65 on
reference structures that are fine. And a plDDT-versus-displacement slope measured on random
noise overstates the coordinate channel by more than an order of magnitude: it reads -0.127 at
0.5 Å, while a coherent alternative fold 2.8 Å away costs 0.018, because noise destroys local
geometry that a genuine refold preserves.
