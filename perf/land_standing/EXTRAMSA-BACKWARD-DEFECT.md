# `predictor(extra_msa=True)` raises in the backward: the checkpoint lambda re-uses a buffer the forward freed

Found 2026-09-26 by `land-standing` on `origin/main` `8ed707b93`, the tree it landed the
`extra_msa` constructor argument on. **Default off, so no user is exposed** — but the switch does
not work when flipped, and the row that owns `tt_bio/bindcraft2.py` should have this before it
spends a card on the A/B.

## What happens

`perf/bcx_extrawire/round_ab.py --rounds 4 --assert-fires --card 3` on qb2 p300c. Round 1, the
OFF arm, completes and banks. Round 2, the first ON round, raises:

    RuntimeError: TT_THROW @ ttnn/core/tensor/storage.cpp:60 — Buffer is not allocated

wrapped by JAX as `JaxRuntimeError: INTERNAL: CpuCallback error calling callback`, because the
backward runs inside a `jax.pure_callback`. The Python frames, innermost last:

    tt_bio/bindcraft2.py:638   ExtraMsaOnDevice._backward
    tt_bio/autograd.py:626     backward
    tt_bio/autograd.py:655     _backward
    tt_bio/autograd.py:2342    <lambda>
    tt_bio/autograd.py:2325    _recompute          <- gradient checkpointing re-runs the forward
    tt_bio/bindcraft2.py:142   <lambda>            <- the checkpointed callable
    tt_bio/af2.py:350          _residual
    tt_bio/taped_ttnn.py:1168  call

## Root cause

`ExtraMsaOnDevice.extra_msa` checkpoints a lambda that calls `_residual` on a value captured from
OUTSIDE the checkpoint (`tt_bio/bindcraft2.py:140-142`):

    if recompute:
        z = self.ag.checkpoint(
            lambda t, blk=block, c=const: blk(blk._residual(t, c), *pair_masks), z)

and `af2._residual` **deallocates both its inputs** (`tt_bio/af2.py:351-352`):

    ttnn.deallocate(update)
    ttnn.deallocate(x)

So the forward frees `const`, and the backward's recompute calls the same lambda again with
`c=const` already deallocated. It is deterministic, not a race and not load-dependent.

The Evoformer arm twenty lines above does not have the bug, and the contrast is the whole fix
(`bindcraft2.py:116-121`):

    m, z = self.ag.checkpoint(
        lambda a, b, blk=block: blk(a, b, msa_mask, *pair_masks), m, z)

It passes every tensor the callable consumes AS a checkpointed argument, and calls no
deallocating helper on a captured one. `device_evoformer_s` was 524.9 s in the same round-1
record, so that arm ran its own recompute fine.

## What the OFF round measured, since it is banked and properly clocked

    round 1, extra_msa off:  round_wall 554.319 s   sequence_gradients 553.796 s
                             device_evoformer 524.909 s   device_extra_msa 0
                             device share of sequence_gradients 0.9478
                             AICLK n=531  min 1343  median 1350  max 1350   load1 17.0

**That is the COMPILE round** — the harness's own default is `--rounds 16` precisely so the first
round's JAX compile amortises — so it is not comparable to the 36.369 s warm median in
`state/ask-bcx-extramsa-default-decision.md`. It is recorded because the clock is real, not
because the seconds are.

## Not fixed here, deliberately

`tt_bio/bindcraft2.py` belongs to the extra-MSA row; two rows in that file is what the charter
forbids. Everything needed to fix it is above, and the obvious shape of the fix is to hoist
`_residual` out of the checkpointed callable, or to pass `const` in as a checkpointed argument so
the recompute gets its own copy.

Evidence: `perf/land_standing/out/extramsa_fire/rounds.jsonl` and `traceback.txt`.

## Confirmed by a model-free reproduction, and it is the mechanism rather than the model

`perf/land_standing/checkpoint_capture_repro.py`, qb2 card 3, **seconds**, no AF2 weights, no
BindCraft 2, no JAX. Two arms, same shapes, one variable — whether the checkpointed callable
frees the tensor it closed over:

    control, keeps it                          -> OK, grad arrived=True
    frees its capture (the extra-MSA shape)    -> RAISED RuntimeError: TT_THROW @
                                                  /project/ttnn/core/tensor/storage.cpp:60

**The same throw site as the BC2 crash**, reached with `autograd.matmul` and a two-line stand-in
for `_residual`. So the defect is `checkpoint`'s contract meeting a consuming callable, not
anything about the extra-MSA blocks, the pool, the pair masks or the model.

It also matters for the fix's cost: a BC2 round is ~10 minutes a try because round 1 is the JAX
compile, and this is a seconds-long red/green the owning row can iterate against.

## The contract, stated exactly

`autograd.checkpoint`'s own docstring says parameters `fn` closes over "need no duplication: they
are LEAVES". `_recompute` (`autograd.py:2322-2325`) duplicates only the declared `inputs`:

    inner = [Tensor(t.value, requires_grad=t.requires_grad) ... for t in inputs]
    with recompute_scope():
        y = fn(*inner)

so every captured value is reused as-is on the second call. That is correct for a leaf that
survives the forward and wrong for one the forward frees. `af2._residual` frees both its
arguments, and `ExtraMsaOnDevice.extra_msa` builds `const` per block with `model._up(...)` and
hands it to `_residual` from OUTSIDE the checkpoint.

**So either end can be fixed**: pass `const` in as a checkpointed input so the recompute gets its
own duplicate, or keep `_residual` out of the checkpointed callable the way the Evoformer arm
does. The repro's control arm shows the second shape already works.
