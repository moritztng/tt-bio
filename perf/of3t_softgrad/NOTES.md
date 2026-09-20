# of3t-softgrad — the two shippable softmax levers, on the gradient, at full scope

Pre-registration: `perf/of3t_softgrad/PREREGISTERED.md`, pushed as commit `c69129695` before
the first arm ran. Bands and bars are that file's, unchanged.

## The 0.4.3 boundary had to be rebuilt first

`/home/ttuser/of3t_rebase/` was pruned with its row's worktree, and it held `diffcap043` — the
0.4.3 diffusion boundary every arm in this row scores against. Five live scripts still point at
it and each fails with a bare `FileNotFoundError`: `perf/of3t_trajectory/devgrad_traj.sh`,
`perf/of3t_residual/devgrad_res.sh`, `perf/of3t_residual/square_check.py`,
`perf/of3t_conditioning/capture_cond_boundary.py`, `perf/of3t_rebase/devgrad043.sh`.

The bundle survives. `/home/ttuser/of3t_refprec/bundle_ref` IS `bundle_min_043`: its own
`MANIFEST.json` names `/home/ttuser/of3t_rebase/bundle_min_043` as its canonical location, and
the five declared hashes are the five the original capture verified. `recap043.sh` rebuilds the
capture from it into `/home/ttuser/of3t_softgrad/diffcap043`, under `$HOME` rather than inside a
worktree that can be pruned the same way.

**The rebuild is checked, not asserted.** Four numbers, three from the original capture report
and one from a row that never touched this rebuild:

| | recorded | rebuilt |
|---|---|---|
| forward loss | 1.2675874205688995 | 1.2675874205688995 |
| cotangent norm | 0.019426651390714835 | 1.942665e-02 |
| `vs_bundle` worst rel | 0.0 | 0.0 |
| `vs_bundle` scope | 89.2106 % | 89.2106 % |
| `of3t-conditioning`'s structure-0 forward rel | 1.104135e-02 | 1.104135e-02 |

`OMP_NUM_THREADS` is pinned at the 8 the original used, because a float64 CPU reduction is a
function of its thread count.

## The controls reproduce, so the harness is the harness

547 tensors, 51.1358 % of the model's squared gradient norm, upstream bf16 reference verified by
digest `ff78d7bc…` before it was loaded (A24). Perfect-fix threshold vs their step 5.750945e-02.

    arm                 v their step    v float64        r   cos v them   ERRCOS   x thresh
    shipped             7.426217e+00  7.568637e+00   7.7814     0.411379  +0.2071   129.1304
    softmax_precise     7.426217e+00  7.568637e+00   7.7814     0.411379  +0.2071   129.1304
    softmax_host_f64    7.777580e-02  5.930664e-02   0.9787     0.997141  +0.0977     1.3524

`shipped` reads the 7.426217 it must read and `softmax_host_f64` reads the 0.0777758 it must
read, to every published digit, on a boundary rebuilt from scratch. `perf/of3t_softgrad/bars.py`
recomputes both headlines independently of `a25.py` and lands on the same values.

**A28, beside every row**: upstream's own bf16 step on this scope is 5.852018e-02 from float64
with r = 1.017575, and 441 of its own 547 tensors sit over the 5.0e-02 per-tensor bar, carrying
41.6989 % of the mass. That is what a faithful bf16 implementation reaches. It does not widen any
bar.

## softmax_precise is a no-op at scope, bit-for-bit

1,440 softmax calls intercepted over the 48 structures, a non-zero count, and the rule fired at
every one. And all **547 of 547 gradient tensors are bit-identical to the shipped arm** —
`rel_l2(precise, shipped)` is exactly 0.0 and the largest absolute difference is exactly 0.0.
The forward is identical too, `forward_rel_median` 8.474800850934073e-03 on both.

The mechanism is one line. `of3t-adaln`'s rule installs the lever with

    kwargs.setdefault("compute_kernel_config", precise_config())

and every softmax call site in this path passes that argument **explicitly**:

    tt_bio/openfold3_diffusion_transformer.py:209   compute_kernel_config=self._softmax_ckc
    tt_bio/openfold3_atom_transformer.py:184        compute_kernel_config=self._softmax_ckc

`softmax_ckc(token)` returns `_SOFTMAX_PRECISE_CKC` or `None` depending on the site flag
(`tt_bio/tenstorrent.py:3535`). Either way the key is **present** in `kwargs`, so `setdefault`
cannot write it. The arm labelled `softmax_precise` was the shipped arm under another name, and
its intercept count could not catch that: the count proves the rule ran, not that the rule
changed anything.

The repair is assignment rather than `setdefault`, and it has to be paired with a reading of
what the site flag already resolves to, because those are two different arms: a lever that is
already on by default is not a lever. Until that is re-run, `of3t-adaln`'s block-8 row
`softmax_precise 0.349441` is scored against the same `setdefault` and should not be quoted.

## Cost on the real arm, with the clock sampled during

`cost.py` reads the per-structure seconds the instrument prints and pairs each arm with the
AICLK sampled from `qbcard/cardtel.tsv` during its own window. Card 0, 48 structures per arm.

    arm                 s/struct   median   x shipped   wall s   AICLK mean   min    max    n
    shipped                 1.23     1.20       1.000       66         1300   800   1350   33
    softmax_precise         1.23     1.20       1.000       64         1316   800   1350   32
    softmax_host_f64        1.77     1.70       1.441       90         1325   800   1350   45

`of3t-softmax`'s micro-bench factors, precise 1.46x and accurate 4.71x per op, are one softmax
shape. At scope the softmax is a small enough share that even moving it to the host in float64
costs 1.441x the whole arm.

## Per-tensor, against both references (A27)

Bar 5.0e-02, out of 547. `their_step` is upstream's own bf16 training step, whose denominator is
THEIR gradient; `float64` is the bundle's float64 gradient. Different measurements.

    arm                 ref              massw       median        worst   >bar   mass>bar %
    shipped             their_step  7.426217e+00 1.398141e-01 1.803328e+01    519     97.5871
    shipped             float64     7.568637e+00 1.250047e-01 1.850397e+01    459     94.4376
    softmax_precise     their_step  7.426217e+00 1.398141e-01 1.803328e+01    519     97.5871
    softmax_precise     float64     7.568637e+00 1.250047e-01 1.850397e+01    459     94.4376
    softmax_host_f64    their_step  7.777580e-02 8.215627e-02 4.463192e-01    472     85.4292
    softmax_host_f64    float64     5.930664e-02 4.885781e-02 2.618132e-01    254     52.0828

## The forward is blind, at scope as it was on block 8

`forward_rel_median` over the 48 structures: shipped 8.474800850934073e-03, softmax_precise the
same to every digit, softmax_host_f64 6.462246e-03. The forward spans **1.31x** across the arms
that are on the record so far while the gradient spans **95.5x** (7.426217e+00 over
7.777580e-02). Block 8 read 1.56x against 52.75x. The blindness is confirmed at scope and it is
wider here than it was on one block.

## softmax_accurate: the backward goes non-finite and is being localised

The `accurate` rule runs `tenstorrent._accurate_softmax` in the forward and the shipped backward
with `precise_config()` on its reduction. On one structure its **forward is healthy and better
than shipped** — forward rel 7.877502e-03 against shipped's 1.104135e-02 on the same structure —
and 30 softmax calls were intercepted, the expected per-structure count. The gradient comes back
non-finite on all 547 tensors, so the fault is in the backward pair rather than in
`_accurate_softmax` itself. The one-structure shipped control on the same rebuilt capture reads a
finite median 9.762531e-01, which is what makes that a localisation rather than a guess.

## Re-running

    perf/of3t_softgrad/recap043.sh                       # ~17 min CPU, rebuilds diffcap043
    perf/of3t_softgrad/chain.sh shipped sm64 precise     # ~4 min on card 0
    perf/of3t_softgrad/chain.sh accurate accpermcot

then, with no card:

    perf/of3t_residual/a25.py --f64 … --bf16 … --bf16-sha ff78d7bc… --arms label=…pt
    perf/of3t_softgrad/bars.py  --f64 … --bf16 … --arms label=…pt
    perf/of3t_softgrad/cost.py  --logs label=…log

## Gated

Nothing ships. No default moved. Every arm is on `wk/of3t-softgrad` and nothing is merged.
`softmax_host_f64` is a host computation on a tape verb, a diagnostic bound, and is not a product
candidate.
