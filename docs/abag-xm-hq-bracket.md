# AbAg-XM HQ bracket (spec 2.8)

Every sample re-scored against every scorable ARK interface row of its target; success recomputed on the same 161 scorable targets with the pinned subsample estimator (ranked@50 = the rank-0 pick, exact). Their anchors: acceptable ranked 66.4%, HQ ranked ~35-38%.

## opendde-abag ranked success per variant (%, 95% target-bootstrap CI)

| variant | acceptable DockQ>=0.23 | HQ DockQ>=0.8 | in their band? |
|---|---|---|---|
| (i) declared row | 67.1 [59.6, 74.5] | 27.3 [21.1, 34.8] | no |
| (ii) max over rows | 68.9 [62.1, 76.4] | 29.2 [22.4, 36.6] | no |
| (iii) mean over rows | 64.6 [57.1, 72.0] | 17.4 [11.8, 23.6] | no |
| (iv) Fab-level | 67.1 [59.6, 74.5] | 25.5 [18.6, 32.3] | no |

declared-row self-test: 483 folds recomputed, 0 disagree with the shipped labels beyond 1e-6.
Fab (variant iv) computed for 86 opendde targets; the rest fall back to the declared row (H-only / dual-VHH / scFv).

## variant (v): PXMeter-style per-row success (row denominator)

| gen | thr | rows | ranked | oracle |
|---|---|---|---|---|
| opendde-abag | 0.23 | 363 | 48.2% | 54.8% |
| opendde-abag | 0.8 | 363 | 19.8% | 28.4% |
| protenix-v2 | 0.23 | 363 | 36.1% | 50.7% |
| protenix-v2 | 0.8 | 363 | 10.2% | 18.7% |
| boltz2 | 0.23 | 363 | 23.4% | 38.0% |
| boltz2 | 0.8 | 363 | 8.3% | 11.8% |

## verdict

No variant lands in their HQ band (33-38%) while holding acceptable in 64-68%, so the 27.3%-vs-~35-38% gap is NOT a label-unit artifact: Fab-level grouping does not raise ranked-HQ (25.5%, CI overlaps the declared row's 27.3%), and even the most generous unit (max over rows) reaches only 29.2%. The residual is ranking-calibration, MSA depth, and protocol: their benchmark folds full assemblies while ours is minimal-unit by design (D11), their MSA is unpublished, and their x-axis is model seeds. No refolding (decided, spec section 4); the chain-pair `dockq` column stays the sole label unit.

