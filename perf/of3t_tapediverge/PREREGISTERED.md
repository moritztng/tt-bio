# of3t-tapediverge — pre-registration, written BEFORE any number of this row exists

Branch `wk/of3t-tapediverge`, cut from `origin/wk/of3t` at `7f626e7bf`. Artifacts
`perf/of3t_tapediverge/`, scratch `/tmp/of3t/tapediverge/`. Nothing under `tt_bio/` moves: every
lever this row prices is installed by monkeypatch from `tdrun.py`, so `git diff origin/wk/of3t --
tt_bio/` stays empty and no shipped default can change.

## Scope and harness, fixed here

One harness for the forward and the gradient of the same scope, because D30's warning is that its
forward and gradient came from two different ones. That harness is
`perf/of3t_diffusion/device_gradient.py`, which reports `forward_rel_median` over 48 structures
and the parameter-gradient statistics from the same process, at the 0.4.3 boundary
(`/home/ttuser/of3t_softgrad/diffcap043`), reference `grads_f64_043.pt`. Card 0 on qb2.

The msa_module pair comes from `perf/of3t_auxheads/msa_instrument.py`, which likewise reports
`forward` and `gradient` from one process.

## Arms, fixed here

    A   shipped                                     control
    A2  shipped, rerun                              A/A floor
    B   renorm            TT_BIO_SOFTMAX_BW_RENORM=1
    C   renorm + sdpa_selfvalue                     D31 discriminator
    D   renorm + sm216_precise                      AMENDMENT 1 narrow arm
    E   renorm + sum_precise_all                    AMENDMENT 1 wide arm
    X   renorm + permute-cot                        break control

`sdpa_selfvalue` forces `value=None` into `autograd.triangle_attention`, so the tape forward value
becomes the function its own backward differentiates. `sm216_precise` gives `precise_config()` to
the numerator reduction at `taped_ttnn.py:216`. `sum_precise_all` gives it to every `ttnn.sum`
call that passes none, which reaches `autograd.py:949` as well and also reaches the forward — so
that arm is expected to move `forward_rel_median` and the narrow arm is not.

## What each arm must show to count

Every arm reports a REACH counter out of the loaded modules. A lever that asked to fire and served
0 calls fails the run rather than reporting the shipped number under another name (D121). A lever
that fired and left the gradient bit-identical is a result and is reported as one
(`a-lever-can-fire-and-be-inert`).

## Thresholds, fixed before the numbers

* **D30/D58 amplification.** Report forward and gradient as a pair per arm, never a ratio alone.
  The factor **collapses** if the renorm arm's gradient/forward ratio is under 3x; it **survives**
  if it is over 10x; anything between is a partial move and is reported as one.
* **D31.** The shipped-vs-selfvalue gradient difference is judged against the **5.0e-02
  per-tensor bar** and against the distance the repaired arm already carries. Under 5.0e-03
  mass-weighted refutes D31 as a material contributor for this model; over 5.0e-02 confirms it.
* **AMENDMENT 1.** Bit-identity against arm B is the test. Any move is reported with its size.

## What this row does not claim

One step is not a training run. Nothing here speaks to stability over 100k steps, to drift, or to
the long-run behaviour of the update rule.
