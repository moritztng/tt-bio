# Finding — `predictor(extra_msa=True)` raises in the backward; a checkpoint callable frees its own capture

Found by `land-standing` 2026-09-26 on `origin/main`. **Still unfixed at `origin/main`
`d8070a1d0`**: `tt_bio/bindcraft2.py:159` and `:162` carry the pattern unchanged.

**Whose:** the extra-MSA row owns `tt_bio/bindcraft2.py`. `land-standing` deliberately did not
fix it — two rows in that file is what the charter forbids — so this note exists so the owner
does not have to rediscover any of it.

## Symptom

`perf/bcx_extrawire/round_ab.py --assert-fires` on qb2 card 3: round 1 (OFF) completes, the first
ON round raises

    RuntimeError: TT_THROW @ ttnn/core/tensor/storage.cpp:60 — Buffer is not allocated

wrapped by JAX as `CpuCallback error`, because the backward runs inside a `jax.pure_callback`.
Default-off, so no user is exposed — but the switch does not work when flipped, and the A/B the
campaign is waiting on cannot run until it does.

## Root cause

`ExtraMsaOnDevice.extra_msa` checkpoints a callable that consumes a tensor it closed over:

    const = model._up(model.opm_constant[index].reshape(1, 1, -1))     # bindcraft2.py:159
    z = self.ag.checkpoint(
        lambda t, blk=block, c=const: blk(blk._residual(t, c), *pair_masks), z)   # :162

`af2._residual` deallocates **both** its arguments (`af2.py`, `ttnn.deallocate(update)` /
`ttnn.deallocate(x)`). `autograd._recompute` duplicates only the declared `inputs` and reuses
every capture — its docstring says captured parameters "are LEAVES" and need no duplication,
which holds only while the forward leaves them allocated. The forward frees `const`; the
recompute calls the same lambda again with it freed. Deterministic, not a race.

## Why the Evoformer arm twenty lines up is fine

It passes every tensor the callable consumes AS a checkpointed argument and calls no
deallocating helper on a captured one. Its recompute ran fine in the same round
(`device_evoformer_s` 524.909 s).

**The discriminator, which generalises:** a capture hoisted outside the block loop is proven to
survive the forward *by the forward itself* — block 2 would fail if block 1 freed it. `const` is
built fresh inside the loop and used once, so nothing in the forward notices.

## It is a known class here with a house remedy

`tt_bio/tenstorrent.py:10008` already records it:

    if not ops.taping():
        # Under a tape the multiply's backward reads its operands (freeing `out` was a
        # storage.cpp:60 TT_THROW); inference keeps the free, 48 MB at 384 aa.
        ttnn.deallocate(out)

Same throw site, already hit once, already fixed this way. `not ops.taping()` appears **6 times**
in that file. `af2._residual` frees unconditionally.

## Blast radius: one of five call sites

`ops.set_checkpoint_hook` wires the contract; five callers use it. `bindcraft2.py`'s Evoformer
arm, `openfold3_msa_embedder.py:185`, `openfold3_template.py:144` and the Pairformer trunk at
`tenstorrent.py:10721` all hoist their captures outside the block loop, and the trunk also gates
its frees on `not ops.taping()`. **The extra-MSA arm is the only one affected.**

## A seconds-long red/green to fix against

A BC2 round is ~10 minutes a try because round 1 is the JAX compile.
`perf/land_standing/checkpoint_capture_repro.py` on `wk/land-standing` reproduces the identical
throw site with `autograd.matmul` and a two-line stand-in, no model and no BindCraft 2:

    control, keeps its capture              -> OK, grad arrived=True
    frees its capture (extra-MSA shape)     -> RAISED RuntimeError: TT_THROW storage.cpp:60

## Two shapes of fix, either works

- pass `const` in as a checkpointed input so the recompute gets its own duplicate, or
- keep `_residual` out of the checkpointed callable, which is what the Evoformer arm does and
  what the repro's green control arm demonstrates.

Full write-up and artifacts: `perf/land_standing/EXTRAMSA-BACKWARD-DEFECT.md`,
`perf/land_standing/out/extramsa_fire/` on `origin/wk/land-standing` (0 behind main, verified).

## Do not confuse this with `d8070a1d0`

`origin/main` `d8070a1d0` is titled *"bindcraft2: a test for the checkpoint family, which nothing
covered"*, and it lands in the same file. **It is unrelated.** That commit is about a *weights*
checkpoint — `_is_multimer` reading a family off the npz array names, following `17ec09261`. The
defect above is in a *gradient* checkpoint, `autograd.checkpoint`. Two senses of the word in one
file, and scanning main's log makes it easy to close this report as already handled. It is not:
the pattern at `bindcraft2.py:159/162` is unchanged as of `d8070a1d0`, verified by reading it.

Also verified on `d8070a1d0`, so the harness is not the blocker: `perf/bcx_extrawire/wirecheck.py`
is **25 checked / 25 passed / 0 failed**, and `perf/bcx_predictor/run_arm.py --help` resolves its
whole import chain. The A/B is blocked on the backward alone.
