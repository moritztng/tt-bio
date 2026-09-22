# The diffusion gradient's 0.9778 is a 1-of-48 scope artefact, not a ceiling

Pass 114. `of3t-rebase` took instrument A at diffusion scope against the rebuilt 0.4.3 reference
and reported median **9.778e-01**, worst **1.291e+00**, **547 of 547** tensors over the 5.0e-02
bar, over a set holding **51.14 %** of the model's squared gradient norm, against a zero-model
baseline of **1.0**. Read as written that refutes half the proof mass. It does not.

## What the artifact says

`device_gradient_043.json`: `structures_asked [0]`, `structures_done [0]`, **`n_struct_total 48`**.
The arm differentiated **one** noised structure. The reference is BUNDLE-MIN's gradient from a
full training step, which accumulates **all 48**.

## The generator's own comment names this failure mode

`perf/of3t_diffusion/device_gradient.py:326`:

> The 48 structures are summed by running 48 tapes and letting the leaves accumulate, which is
> exact only if `backward` adds into an existing `.grad` across tape contexts rather than
> replacing it. A norm that grows structure over structure is the evidence; **one that stays flat
> means the run is measuring the LAST structure alone and the total is wrong.**

With one structure there is **one** probe value, `1.2510e-04`, and nothing to compare it against.
The check built for exactly this cannot fire.

## The asymmetry is explicit in the comparison code

Stress-tested rather than assumed, because this desk has been wrong four times this session. The
two references are selected differently in the same function:

```python
fwd_ref = S["xl_out"][0, k].double()      # forward: indexed BY STRUCTURE k
...
r = ref_grad.get(nm)                       # gradient: keyed by parameter NAME only
d = ||gt - r|| / ||r||                     # -> the full 48-structure accumulation
```

So the **forward** is compared per-structure, which is why `forward_rel_median` reads a clean
**1.104e-02**. The **gradient** is compared against the accumulation while our side ran one
structure. The asymmetry is in the code, not inferred from the numbers.

**And note what this does NOT say.** It does not say our diffusion backward is correct. A wrong
`g_0` compared 1-of-48 also reads ~0.99. The measurement cannot distinguish the two, which is
precisely why it must not be published as either a pass or a ceiling.

## Running all 48 is necessary and not sufficient

The probe's own comment gives the second condition: accumulation across 48 tapes "is exact only if
`backward` adds into an existing `.grad` across tape contexts rather than **replacing** it." A run
that replaces would measure the **last** structure alone — and would also read ~0.99. So the
48-structure arm is only valid if the probe is shown **growing** structure over structure. Two
failure modes, one number.

## The arithmetic predicts the observed number

With reference `G = Σ_{k<48} g_k` and the arm computing `g_0`, the `g_k` roughly independent and
of similar magnitude:

```
||g_0 − G||²  =  ||g_0||² − 2⟨g_0,G⟩ + ||G||²  ≈  (1 − 2 + 48)·||g||²  =  47·||g||²
ratio         =  √47 / √48  =  0.9895
```

| | |
|---|---|
| predicted for a 1-of-48 arm | **0.9895** |
| **observed** | **0.9778** |
| zero model | 1.0000 |

A single-sample gradient compared against a 48-sample sum lands at ~0.99 **by construction**,
whether the port is right or wrong. That also explains why all 547 tensors are over bar uniformly
and why the median sits a hair *under* the zero-model answer rather than at it.

## What is genuinely good here and must not be discarded with it

- **A18's discriminator reads 1.104e-02**, inside the 5.0e-02 bar PROTOCOL **A19** fixed earlier
  the same night, *before* the number existed. Admitting the gradient was correct.
- **Coverage is 547 of 761** reference tensors with `weights_without_grad = 0`, against D21's 283.
- **The reference floor is bit-exact**, 0.000e+00 over all 761 tensors, so nothing on the
  reference side contributes.

The instrument is in far better shape than at D21. Only its **scope** is wrong.

## The two ways out

1. Run all 48 structures, letting the leaves accumulate, and show the probe **growing** structure
   over structure as its own comment requires. That is the real measurement.
2. Or emit the **per-structure** reference gradient for `k = 0` from `bundle_min.py`, which makes
   the arm already taken valid immediately at no extra device cost.

`GRADIENT:` should read **scope-limited**, with the probe's growth curve beside it. A18 cleared;
the gradient was not yet asked the right question.
