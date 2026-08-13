# ttnn: a re-tilized copy of a provably identical tensor changes a fold's output

Filed from tt-bio (https://github.com/moritztng/tt-bio). Nothing here is a tt-bio bug; tt-bio is
where it was found and where the reproducer lives.

Versions. The Blackhole leg ran on qb2 against a source-built tt-metal at
`4760ad176fb3e714bcc5fc001bc6e99778691ac2` (`v0.73.0-dev20260610-56-g4760ad176fb`), ttnn
`0.65.1rc17.dev6519+geba89dcec7e.d20260614`. The Wormhole Galaxy legs predate that check and their
build was not recorded, so treat the Wormhole version as unknown rather than assuming it matches.

## Summary

Substituting OpenDDE's `z_struct` pair tensor with a re-tilized copy of itself
(`to_layout(ROW_MAJOR)` then `to_layout(TILE)`, or equivalently `to_torch` then `from_torch`)
changes every structure the fold produces. A `ttnn.clone` of the same tensor changes nothing. The
two tensors agree exactly under every op we probed, including the two production ops that consume
them.

## Reproducer, ~140 s per leg

One tree, one env flag or one-line patch per leg. Target 9ncy (505 tokens), config
`--recycling_steps 1 --diffusion_samples 1 --max_parallel_samples 1 --seed 42`. Wormhole Galaxy
card; the fold-level effect also reproduces on Blackhole (below). Substitution point is
`OpenDDE.expand_and_refine` in `tt_bio/opendde.py`, immediately after

    z4 = ttnn.reshape(z_st, (1, Ns, Ns, self.expander.c_z))

Six legs, five of them neutral controls, CIF md5 as the readout:

    no touch                                    e6227922e3869293575511f9a69ee618
    discarded ttnn.to_torch(z4)                  e6227922   the read is innocent
    ttnn.clone(z4)          device -> device     e6227922   a device copy is innocent
    bare 733 MB alloc + free before the refiner  e6227922   allocator churn is innocent
    to_layout(ROW_MAJOR) -> to_layout(TILE)      b8999b32   <-- moves the fold
    to_torch -> from_torch                       b8999b32   <-- same value

The host is not involved: a device-only re-tilize reproduces the host round trip exactly, and
`clone`, which copies the tile buffer without re-tilizing, does not.

## The rule that accounts for every observation

    z_struct assembled by ttnn.concat of tiled chunks -> 9ncy e6227922, 9i3p 2a5c3583
    z_struct produced by a tilize (from_torch, or an
      explicit to_layout(RM) -> to_layout(TILE))      -> 9ncy b8999b32, 9i3p f786d83e

`from_torch` tilizes; `ttnn.concat` of already-tiled chunks does not. Nine rounds of A/Bs, every
result consistent with this. The two 9i3p values differ only in which assembly path produced the
byte-identical tensor.

Reproduced on Blackhole (P300, qb2) on 2026-08-12 at 9i3p: concat path
`2ded34e38b4e6522961240875ab3552dd29d82e2812e74baa5d7660cfd78b2f0`, tilize path
`9c20a40581aa5bb2c859821fcfec039265be6bb363a15cd58b23156bf3b7a3db`, both within-arm deterministic
across repeats. So it is not Wormhole-specific.

## The tensors agree under every op we probed

Measured on the real tensor inside a real fold, both assemblies built in one process from the same
blocks (`scripts/probe_zstruct_assembly.py`):

    [ZCMP] logical equal=True
    [ZCMP] minimal_matmul equal=True max=0 mism=0/122179712
    [ZCMP] linear         equal=True max=0 mism=0/122179712
    [ZCMP] layer_norm     equal=True max=0 mism=0/366539136
    [ZCMP] sum_dim-2      equal=True max=0 mism=0/375168

`minimal_matmul` is the trimul in-projection and `linear` its output projection, so these are the
production ops, not gentler probes. At 1902 tokens the same comparison is 0 mismatch in 115.8 M
elements. Shape, padded shape, dtype, layout and memory config all match.

**This does not establish that the two tensors are interchangeable under every op.** It establishes
it for four. The refiner runs more than four ops on `z_struct`, and since the fold demonstrably
moves, the divergence has to come from one that was not probed.

## Eliminated mechanisms, each measured, single variable, one tree

- tile-padding content: a contraction over the padded token axis is 0 mismatch in 30.5 M elements
- dtype, layout, padded shape, memory config: all match
- placement: a 256 MB global address shift and a targeted `ttnn.reallocate(z_struct)` are both
  byte-identical
- freeing order and synchronisation in the assembly loop: both neutral
- the host read itself (a discarded `to_torch`): neutral
- the refiner's cold first call / lazy `_gp_cache` build (96 tensors, 9.4 MB): prewarming it is
  numerically inert. An earlier discarded-pass warm-up did move the fold, but only because it
  round-tripped z to the host and back, i.e. it was another re-tilize

## The next experiment

Bisect the refiner: hash every intermediate that consumes `z_struct`, under both assemblies in one
process, and find the first op whose output differs. That op, not the tilize, is the defect. A
plausible shape for it: a downstream kernel selecting on the tensor spec rather than on its logical
contents, which is why `probe_zstruct_assembly.py` now dumps shape / padded_shape / dtype / layout /
memory_config for both assemblies before comparing any data.

## Why it matters downstream

tt-bio's large-target capacity fix uploads `z_struct` as one `from_torch` allocation, so above a
1.5 GiB threshold OpenDDE outputs land on the tilize value instead of the concat value. Both values
come from provably identical inputs and neither is more correct. A number that moves when an
unrelated upstream allocation changes is not defensible as a reference, whatever it is a reference
for.
