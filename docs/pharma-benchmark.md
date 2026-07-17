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

These are the committed benchmark measurements for TT-Bio 0.3.0.

| model | target | metric | R | D | X | result |
|---|---|---|---:|---:|---:|---|
| ESMC-300m | 4 proteins, L20–129 | embedding PCC | 1.00000 | 1.00000 | 0.9987–0.9996 | PASS |
| ESMC-600m | 4 proteins, L20–129 | embedding PCC | 1.00000 | 1.00000 | 0.9994–0.9996 | PASS |
| ESMFold2 | trp-cage, L20 | CA-RMSD | 0.51 Å | 0.16 Å | 0.61 Å | PASS |
| ESMFold2 | GB1, L56 | CA-RMSD | 0.29 Å | 0.18 Å | 0.33 Å | PASS |
| ESMFold2 | ubiquitin, L76 | CA-RMSD | 0.92 Å | 0.23 Å | 0.75 Å | PASS |
| Protenix-v2 | 7ROA, L117, MSA | CA-RMSD | 2.94 Å | 1.47 Å | 2.63 ± 0.42 Å | PASS |
| Boltz-2 | trp-cage, L20, no MSA | CA-RMSD | 0.79 Å | 0.37 Å | 0.60 ± 0.24 Å | PASS |
| Boltz-2 | 7ROA, L117, MSA | CA-RMSD | 0.81 Å | 0.98 Å | 0.94 ± 0.14 Å | PASS |
| OpenDDE | trp-cage, L20, no MSA | CA-RMSD | 0.31 Å | 0.24 Å | 0.39 ± 0.11 Å | PASS |
| OpenDDE | 7ROA, production settings | CA-RMSD | 1.90 Å | 8.06 Å | 5.68 ± 3.98 Å | PASS |
| OpenDDE-abag | 1AHW antibody–antigen | global DockQ | 0.83–0.86 | 0.863–0.882 | device matches reference | PASS |
| BoltzGen | binder against 7ROA chain A | designs ≤2 Å scRMSD | 68.75% | 93.8% | device ≥ reference | PASS |

The ESMFold2 comparison also checks an alignment-free coordinate metric and
sampler-independent pLDDT, distogram, and pTM outputs. Protenix-v2's confidence
head under-ranks some samples in both the upstream implementation and TT-Bio;
the larger R floor reflects that shared behavior. OpenDDE-abag matches the
upstream checkpoint on 1AHW. Both implementations perform poorly on 9DSG, so
that target is a checkpoint limitation rather than a port discrepancy.

## Reproducing a comparison

Embedding parity runs the upstream ESM model directly:

```bash
TT_VISIBLE_DEVICES=0 ESM_ROOT=/path/to/esm \
  python3 scripts/pharma_parity.py embeddings --model esmc-600m
```

Structure parity consumes result directories from matched device and reference
seeds:

```bash
python3 scripts/pharma_parity.py structures \
  --ref-fixtures protenix-v2/prot/msa-server_200step_5sample_10cycle_bf16 \
  --dev-dirs /path/to/device-seed0 /path/to/device-seed1
```

Regenerate a reference fixture only when its pinned upstream version or settings
change. Use `scripts/pharma_harvest_ref_fixtures.py` and review the fixture
metadata before committing it.
