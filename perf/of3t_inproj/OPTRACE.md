# Inference trimul op trace, base vs fix

The inference A/B owed for this row's `tenstorrent.py` edit asks one question: does an inference
fold get slower. The host has not been quiet (loadavg1 20 to 29 from the bgref BoltzGen fp32 CPU
references and of3t-denoise device arms, host_quiet.py red), and the 1536 reference has no finish
inside 24 h. So the wall-clock half was replaced by a check that load cannot move: device time
is set by the programs dispatched, so identical programs and identical outputs mean identical
device work.

`optrace.py` builds one `TriangleMultiplication` (c=128, random weights in tt-bio's fused
layout), runs it once to warm, then captures one call with `ttnn.graph`. Grid: L 64/256/384/512,
outgoing and incoming, `gated_move` off and on, pair mask off and on, 32 configs. Per config it
records the op-name sequence, a hash of every node's arguments (shapes, dtypes, memory configs,
program descriptors, allocated buffers) with pointers, buffer addresses, tensor ids and the tree
root removed, and the sha of the output bytes.

Trees: base = HEAD with `tt_bio/tenstorrent.py` from dce82cc17 (sha 93d648c28fee), fix = HEAD
(96e3e241caa1). Each run records the `tenstorrent.py` it imported. Order base, fix, base2, fix2 on
qb2 p300c card 1, AICLK during min 800 / median 1350 / max 1350 (69 samples).

| pair        | op names | arguments | output bytes |
|-------------|----------|-----------|--------------|
| base/base2  | 32/32    | 32/32     | 32/32        |
| fix/fix2    | 32/32    | 32/32     | 32/32        |
| base/fix    | 32/32    | 32/32     | 32/32        |
| base2/fix2  | 32/32    | 32/32     | 32/32        |

Before the kernel path was normalized the argument hash matched base/fix on only 16/32 (every
config at L 384 and 512). The one difference was `kernel_source=<tree>/tt_bio/kernels/...` in the
custom mm_split and reblock_permute programs; with both roots mapped to one token the L=384
graphs match node for node (244 nodes).

Host side: the guards the edit adds to inference are an `is None` test in `_gp_in_chunks` and
`_gp_in_gout`, an `ops.taping()` test in `_gout_eligible`, and a torch/ttnn branch in
`_gp_fused_order` that only runs on a cache miss. A cache-hit `_gp_in_chunks` call costs a median
0.154 to 0.179 us across the four arms, base and fix alike (`host_us` in OT_*.json).

The fold-level digests were already identical (ab.log: OpenFold3 8b9caa931d4ef2ee x4, Protenix-v2
f09c55d8e80bf3ce x4). Out of scope: a quiet-host wall-clock figure. It was not taken.
