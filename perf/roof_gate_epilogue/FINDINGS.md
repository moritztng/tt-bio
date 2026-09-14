# The triangle-attention gate multiply: 2.412 ms a block, and no way to fuse it into the producer

`perf/roof_orchestrator/FUSION_PAIRS.md` ranks
`generic_minimal_matmul (fused TriAtt in-projection) -> gate_and_project's multiply_` at 268.4 MB a
Pairformer block, 3.62 % of the block's 7416.9 MB, and calls it the cheapest structural item on the
list because the producer is already a custom fused `generic_op` with its own writer.

It is not available, and the reason has nothing to do with the kernel.

## Why the producer cannot host the epilogue

`gate_and_project` computes `o * sigmoid(g)`. `g` is the buffer to delete. `o` is
`attend(q, k, v, bias)`, and q, k and v come out of the **same** `generic_minimal_matmul` call as
`g`: `TRIATT_FUSED_QKVG` and `TRIATT_FUSED_QKVGB` (`tt_bio/triatt_qkv.py`) are both on by default,
so one matmul writes q, k, v, the gate and the pair bias as N chunks of one contraction. An epilogue
in that kernel's pack stage would multiply by a tensor its own output is an input to.

Measured on whglx card 2 over 18 TriangleAttention calls: `qkvg` 18 served / 0 declined,
`qkv_heads` 18/0, `tail` 18/0, `sdpa_fused` 27/0, zero rejects.

## Where it can go instead

Both operands exist together only after the SDPA.
`tt_bio/kernels/triatt_sdpa/compute/compute_common.hpp:2092` ends every q chunk with the 1/rowsum
normalisation packing straight to `cb_out`. `vDHt == 1` and the gate is head-major `[B, H, S, 32]`,
so gate tile *i* is output tile *i*: one extra CB, one extra reader stream, a `sigmoid` and a `mul`
before the pack. The out-projection's in0 reader is the other candidate and is worse — it needs an
extra L1 round trip per tile and competes with the matmul's DST.

Both need a descriptor change and a new eligibility condition in a kernel five models share.

## What the bytes are worth

whglx card 2, Wormhole B0, 512 aa, grid 8x9, HiFi4 / fp32 dest acc, arms interleaved per rep,
2 warm + 7 timed, medians. `block_ablate.py` elides the two triangle-attention gate multiplies and
leaves everything else alone; that is the ceiling a correct fusion could reach.

    base                77.770 ms
    gate elided         75.358 ms
    ceiling              1.03200x     2.412 ms/block
    A/A floor            0.034 %
    realistic            1.0211x      charging the host kernel its new 67.1 MB read

    per gate multiply    1.206 ms in the block, 201.3 MB, 166.9 GB/s
                         = 73.4 % of the 227.5 GB/s measured Wormhole DRAM roof
    isolated (screen.py) 1.367 ms, 147.3 GB/s

The shim's reach is censused, not assumed: two sigmoid-gated `(512, 4, 512, 32)` multiplies a block
elided, two `(1, 512, 512, 128)` trimul input gates left alone. Eliding all four instead reads
1.0681x / 4.957 ms, which is `roof-fuse-trimul-out`'s row on top of this one.

KIND arithmetic, k = 0.623 +/- 0.090 -> predicted Blackhole 1.0132x on the block. Predicted, not
measured.

## The harness trap this hit

`tt_bio.reference.PairformerLayer` zero-initialises every output projection — 13 of its 2-D weights
are exactly zero, including both triangle attentions' `linear_o.weight` and `linear_g.weight` — so
the block it builds is the identity on z and any digest read off its output is a constant.

Verified directly (`gate_effect.py`, whglx card 2): eliding **both** gate multiplies leaves
`torch.equal(out_z_gated, out_z_elided)` True, `max|dz| = 0.0`.

`perf/b2z2_byte_round2/ab_block.py` refills the zeros and documents why.
**`perf/b2z2_byte_floor/ab_block.py` does not**, and that is the harness behind the trace this
ranking is built on: its "same digest both arms" leg cannot discriminate any triangle-attention or
trimul-tail change, and the lever it measured — the fused qkv+gate projection — routes its entire
effect through the zero `linear_o`. `block_ablate.py` here refills them and prints which it refilled;
with live weights the arms' digests separate and the A/A leg's stay identical.
