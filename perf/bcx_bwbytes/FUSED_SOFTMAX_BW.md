# The fused softmax backward: what it is worth, and whether it can be written

The trunk backward's largest single bucket is one expression on the triangle attention's score
tensor: **10.122 GB of an AF2 Evoformer block's 39.394 GB at n=256, in thirty ttnn calls**
(`perf/bcx_bwbytes/reach.json`). It is the only bucket in the backward that scales as n-cubed --
exponent 2.92 on Evoformer and 2.94 on extra-MSA, against 1.88-2.10 for everything on the pair
track -- so its share grows with the model's size: 23.7 % of the block at n=224, 25.7 % at 256,
37.6 % at 512 and **58.5 % at MGX's 1536**, where it alone moves more bytes than the whole rest of
the block.

## What it costs today, in passes of the score tensor

With `SOFTMAX_BW_RENORM` on, which it has been since 2026-09-21:

    multiply(g, y)            2 reads + 1 write
    sum(., -1)                1 read
    sum(y, -1)                1 read
    divide                    O(n^2), negligible
    subtract(g, inner)        1 read + 1 write
    multiply(y, .)            2 reads + 1 write
                              ---------------------
                              10 passes, all fp32

A single kernel reads y, reads g and writes dx: **3 passes**, or 1.5 fp32-equivalents at bf16, and
it pays no narrowing cast because the narrowing happens inside. So the bucket falls to about 15 %
of itself, which is 5.8 GB per Evoformer block at n=224 and the difference between 1.071x and
1.198x on a real BC2 round.

The zero-build floor under that job already exists and is `SOFTMAX_BW_ROUTE="moreh"`
(`tt_bio/autograd.py`): `ttnn.moreh_softmax_backward` is bound in the `ttnn==0.68.0` wheel, and
`dx = S * moreh(y/S, g)` with `S = rowsum(y)` keeps the renorm exactly. That is 8 passes against
10, so it is worth about a fifth of what a real kernel is worth. Take it as the baseline to beat,
not as the answer.

## Can it be written in tt-lang? Yes, and the one limit it hits is not the expensive one

Checked against the repo at `/home/ttuser/tt-lang` this pass, card-free.

**The simulator is genuinely card-free.** `bin/ttlang-sim` is a shell script whose last line is
`exec "$PYTHON" -m sim.ttlang_sim "$@"`, and `python/sim/ttlang_sim.py` knows only about
*simulated* devices (`--num-devices`, `set_num_devices`, `GetNumAvailableDevices()`). It opens no
hardware. `examples/eltwise_add.py` runs and prints PASSED with every card on the box held by
another row, and the four holders were byte-for-byte the same pids before and after.

**Correction worth carrying:** the `tt-lang-fused-kernel` skill says "the sim binary dispatches to
device when present" and points at `TTLANG_HAS_DEVICE` to confirm it. On this checkout that is
wrong twice over. `TTLANG_HAS_DEVICE` is exported by `build/env/activate` as a CMake BUILD-time
fact -- the machine that built it had a device -- and it is read by `test/ttlang_test_utils.py`,
not by the launcher. A row that believes the skill either avoids a safe card-free tool or reports
a hardware run it never did.

**The two headline compiler limits do not apply.** No 3D batched matmul and no
`ttl.math.transpose`: this expression has no matmul at all and no transpose. It is elementwise
plus two reductions over the last dimension.

**The limit it DOES hit is elementwise feeding a reduce**, issue #474, and the compiler says so
with a clear message rather than failing to legalise
(`test/python/invalid/invalid_reduce_fused_elementwise.py`). `rowsum(g*y)` is exactly that shape.
The documented workaround is to store the elementwise result to a dataflow buffer first -- and a
dataflow buffer is **L1**, so the workaround costs L1 traffic and not a DRAM round trip. The
3-DRAM-pass target survives it intact, which is the whole point: this bucket is DRAM bytes.

## What a row picking this up still owes

* the reduce idiom is `ttl.math.reduce_sum(block, scaler, dims=[-1])` with its own scaler block
  (`test/sim/test_math.py`); confirm whether it reduces WITHIN a tile as well as across the tile
  grid, because the softmax backward needs the full row sum and that is the one semantic this
  note did not pin down
* grade against float64, not against another device arm. `perf/bcx_bwbytes/softmax_bw_probe.py`
  already builds the reference as the RENORMED expression, so an arm that drops the correction is
  graded against what dropping it costs
* the sibling caution: `moreh_layer_norm_backward` is WRONG on Blackhole (dx 2.741e+06 relative
  L2 in bf16, upstream #12349). Nothing in this family is believed on a speed number alone
* it is engine-level and serves Boltz-2, OpenFold 3, BC2 and RFdiffusion 3 alike. UNIFIED NOT
  PER-MODEL is standing, and a `bcx_softmax_bw` anywhere in it is a wrong turn
