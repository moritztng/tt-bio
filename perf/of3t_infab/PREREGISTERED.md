# of3t-infab pre-registration

Written and pushed before the first fold. Question: does wk/of3t change what regular inference
computes or how long it takes? Moritz: "make sure regular inference is not changed to softmax fp64,
not made slower ... i dont want to see regression in inference."

## Trees

- BEFORE `c5b346679df0cda6b0aa511a41526b481e8da541`, the merge-base of origin/main and wk/of3t.
- AFTER `589504144e43a3d9dca644c7385f4663c72d4e39`, origin/wk/of3t (of3t-verbinstall, of3t-inproj
  and D266 included).

Both are `git archive` exports in `/tmp/of3t/of3t-infab/{before,after}`. Each fold records the
`tt_bio.__file__` its folding process loaded, so a fold that picked up the wrong tree shows.
Fixtures are read from this worktree for both arms, so both fold the same input bytes.

## Instruments

`infab.py` (copied from `perf/land_standing/d264/infab.py`, widened) and `optrace.py` (copied
unchanged from `perf/of3t_inproj/optrace.py`). Card: qb2 card 1 (p300c), `TT_VISIBLE_DEVICES=1`,
lease holder `worker:of3t-infab`. Per model the order is BEFORE, AFTER, BEFORE, AFTER.

Inside every process a fold starts, a sitecustomize records:

- census: every `tt_bio.*` module imported, `tt_bio.autograd` present or not, its exact softmax and
  layer norm counters read without importing it, `tt_bio.taped_ttnn` and `tt_bio.train.*`.
- op trace: every call through ttnn's `FastOperation.__call__`, which in fast-runtime mode is
  every ttnn op, operator overload, `Tensor.__getitem__` and `generic_op` custom kernel. One line
  per call: nesting depth, op name, each argument's signature (tensor shape, dtype, layout,
  storage, memory config; program descriptor kernel sources, compile-time args, defines, core
  ranges; the repr of everything else, which carries program configs and compute kernel configs),
  pointers and the tree root removed.
- digest: sha256 of every `.cif`/`.pdb`/`.npy` output, array contents for `.npz`, AF2's own
  `structure_sha16_all`.
- wall seconds, the fold's own `runtime_s` where results.json has one, and AICLK from tt-smi every
  3 s during the fold (min/median/max).

The folding process of a model is the process that imported its key module (`tt_bio.openfold3`,
`tt_bio.protenix`, ...). A fold whose census never saw that process fails the fold.

## MODELS

Folded, one small cell each, 4 folds per model:

| model | command | cell |
|---|---|---|
| openfold3, openbind, boltz2, protenix-v2, protenix-v1, esmfold2, opendde, rf3 | `predict --single_sequence --sampling_steps 6 --diffusion_samples 1 --seed 0` | `perf/size512/fixtures/cdk2x2_128.yaml` (128 aa) |
| af2ig (what af2.py serves) | `scripts/af2_port/fold_timing.py --reps 2`, params_model_1_ptm | `designpop_bg119/binder_complex.pdb` |
| boltzgen | `design --num_designs 1 --steps design` | `tests/fixtures/boltzgen/bg400.yaml` |
| rfd3 | `design --num_designs 1 --seed 0 --num_timesteps 20` | `iai_protein/iai_inputs.yaml` (40 tokens) |
| pxdesign | `design --num_designs 1 --seed 0 --n_step 50` | `tests/fixtures/pxdesign/PDL1.yaml` |
| nesso1 | `affinity --trunk bf16 --recycling_steps 5 --tokens_budget 256` | `perf/nesso1/inputs/ladder/aa128/cdk2_128.yaml` |
| esmc-300m | `embed` | `fixtures/cdk2_128.fasta` (first 128 aa of the cdk2 fixture) |
| saprot-35m | `saprot` (no structure) | same fasta |

The reduced step counts (rfd3 20, pxdesign 50, 6 sampling steps) are the same in both arms; this
row compares trees, it does not time a production run.

In scope = the census shows the folding process imported at least one module the diff touches
(`git diff --name-only BEFORE AFTER -- tt_bio`, `.py` files). A model the census puts out of scope
is reported with its census and not held to the bars.

Excluded, not folded, to keep the matrix inside one pass: esmfold2-fast, opendde-abag, esmc-600m,
esmc-6b, saprot-650m, saprot-1.3b. Each is a second checkpoint of a model folded above and runs the
same Python modules with different weights. Its import set is taken from its sibling's census,
which is an inference, not a measurement, and is stated as such in the result.

## ACCURACY bar

Per in-scope model: every digest identical across all four folds (A/A BEFORE, A/A AFTER, A/B).
Any digest change is a finding: it is attributed to a named commit or it is a FAIL. A model whose
A/A pair already differs has an unreadable digest bar; it is named, the source of the
nondeterminism is stated, and it cannot count toward GO.

`tt_bio.autograd`, `tt_bio.taped_ttnn` and every `tt_bio.train.*` module absent from every process
of every fold, exact counters zero. Positive control for the same counters:
`perf/of3t_stackship/STACK_SHIP_SHIP2.json` (commit 7a325213c, qb2 card 1, training arm): softmax
verb 5901 / raw 1742, layer_norm verb 2160 / bw 1296.

## TIME bar

Per in-scope model: the folding process's op trace identical BEFORE vs AFTER, in both pairs
(fold 0 vs 1 and 2 vs 3), with A/A trace identity (0 vs 2, 1 vs 3) reported first. Identical
trace means identical programs dispatched, so identical device time by construction. If A/A
traces differ the trace is not an instrument for that model and it is named.

A B/A trace difference is priced op by op: which ops, how many, at what shapes. An explicit
`kwarg=None` against an absent kwarg is resolved against the op's own signature default. Any other
added or changed device op fails the bar unless its device time is measured at or below the A/A
floor.

`optrace.py` (TriangleMultiplication under `ttnn.graph`, 32 configs) runs BEFORE, AFTER, BEFORE,
AFTER after the matrix: op names, argument hash and output bytes must match 32/32 in every pair.

Walls: recorded per fold with DURING-sampled AICLK. They are a timing claim only if
`host_quiet.py` exits 0 both before and after the matrix, and then only against the A/A floor
|B0 − B2| and |A1 − A3| stated first (B0 and A1 also pay kernel compile for their tree, so the
floor is conservative). qb2 loadavg was 68 at 17:41Z today, so the expected outcome is that the
walls are not a timing claim.

## Prediction

Every in-scope model: digests identical 4/4, traces identical, autograd absent. The inference
edits in the diff are site selectors that resolve to the shipped op when no site flag is set, and
guards on `ops.taping()`, which is False with no tape open. The one place I expect the trace might
move without device work moving is an explicit `compute_kernel_config=None` passed where BEFORE
passed nothing (protenix AtomTransformer's `site_softmax`).

## VERDICT rule

GO = both bars hold on every in-scope model. NO-GO = a digest or trace change on inference that no
commit accounts for as intended, or an added device op with a cost. PARTIAL = a model's bar is
unreadable (A/A fails, fold fails), named.

Budget: one pass of this matrix. No models or repetitions beyond this file.

## Addendum, 2026-09-25, AFTER2 (written and pushed before its first fold)

The first matrix (INFAB.json, trees above) found two inference ops that wk/of3t added: finding 2,
16 `from_torch` per openfold3/openbind fold (0802a53f0), and finding 4, 484 `ttnn.clamp` per
protenix-v2 fold (f84e232af). Both were removed in 83da296bc, now in AFTER2 = `f01fa0813`.
`git diff 589504144 f01fa0813 -- tt_bio` is `openfold3_confidence.py` and `tenstorrent.py`
(`_accurate_softmax` only) and nothing else.

Folded B, A2, B, A2 with the same infab.py into `INFAB2.json` (workdir `work2/`):

- openfold3, openbind (both carried finding 2), protenix-v2 (the only folding process in the first
  matrix whose trace reached `_accurate_softmax`), opendde, rf3 (both import tenstorrent.py).
- rfd3 again, now with `--from_pdb`. Its first-matrix folds failed in both arms before any device
  work: the default featurization path needs a golden bridge that is a dev fixture. A harness
  fix, not a new model.

Bars on AFTER2:

- protenix-v2, opendde, rf3, rfd3: digest identical 4/4. Trace B vs A2 identical after removing an
  explicit `compute_kernel_config:None` kwarg (finding 5, the signature default).
- openfold3, openbind: digest A2 equal to the first matrix's AFTER digest (D1 alone moved it off
  BEFORE). Trace B vs A2 differs only by the D1 `multiply_` scalar (finding 1), 8 fewer
  `deallocate` (finding 3) and finding 5; `from_torch` count delta 0.
- autograd, taped_ttnn and train.* absent from every process, exact counters zero, as before.

Training side, `walkcheck.py`, card 1: build `OpenFold3Forward(...).model` in the AFTER and the
AFTER2 tree. The walked device tensor count must be equal, the 16 confidence-head weights must be in
the AFTER2 walk, and `confidence_head.late_device_weights()` must be empty. 3948 was the D127 tree;
later commits added walk tensors, so the reference is AFTER, not that number.

Card 1's board-pair sibling dev0 may be co-tenanted; walls are recorded and are no timing claim.

Addendum 2, 2026-09-25 02:05Z, before its folds: protenix-v1 is added to the AFTER2 set. The first
matrix shows it ran the clamp too (196 per fold), as did opendde (488) and rf3 (484), so "protenix-v2
is the only process that reaches `_accurate_softmax`" above was wrong. Same bars as protenix-v2.

Addendum 3, 2026-09-25 ~02:45Z, before its folds: boltzgen's digest bar was unreadable in the first
matrix. `bg400.cif` and `bg400.npz` were identical 4/4, but `intermediate_designs/bg400.cif`
differed A/A in both trees: the designed sequence changes run to run. Source: `design --model
boltzgen` ignores `--seed`, nothing seeds torch, and `data_from_yaml.py:266` /
`data_from_generated.py:463` call `np.random.default_rng(None)`, which reads OS entropy. Its trace
was identical A/A and A/B (999053 ops), so the programs dispatched did not change.

Harness fix, same class as rfd3's `--from_pdb`, no tt_bio change: with `INFAB_SEED=0` the folding
process's sitecustomize seeds `random`, numpy and torch at start and maps `default_rng(None)` to
`default_rng(0)`, identically in both arms. boltzgen is folded B, A2, B, A2 into `INFAB3.json`
(workdir `work3/`). Bar: every digest identical 4/4, trace B vs A2 identical, census as before. If
A/A still differs, boltzgen stays unreadable, it is named, and it does not count toward GO.
