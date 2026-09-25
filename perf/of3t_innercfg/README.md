# D55's last line: the config it withholds cannot change the answer

`tt_bio/autograd.py:softmax_bw_inner` threads its `config` to the row-sum denominator and not to
the numerator `inner = sum(g*y)` that `dx = y*(g - inner)` is built on. D55 has been filed
against that asymmetry twice. It is cosmetic, for two independent reasons, and this directory is
the measurement.

Host pc, card 0, Blackhole p150a, AICLK 1350 MHz during the device work. Accuracy does not move
with the clock; it is recorded because every device reading on this fleet carries one.

## 1. `ttnn.sum` already accumulates in fp32, so the config has nothing left to buy

`CONTROL2.json`, the numerator reduction at the trunk's own shape `[8,4,384,384]`, dim -1, against
a float64 reference on the operands the card held:

| config on `ttnn.sum` | rel L2 vs float64 | bit-identical to no config |
|---|---|---|
| none (as it ships) | 2.3961e-03 | — |
| `precise_config()` | 2.3961e-03 | **yes** |
| HiFi4, `fp32_dest_acc_en=False` | 7.2872e-03 | no |
| HiFi2, `fp32_dest_acc_en=False` | 6.8733e-03 | no |
| LoFi, `fp32_dest_acc_en=False`, approx on | 6.8733e-03 | no |

The kwarg is read, not dropped: taking `fp32_dest_acc_en` away moves the reduction 3.0x. The same
five configs on a matmul (`BREAK_CONTROL_matmul_same_sweep`) separate cleanly, 1.7369e-03 to
2.2969e-02, so the sweep can tell configs apart. Same answer on dim -2 and dim 0.

`CONTROL.json` shows why in one number: 4,096 addends of 1.0 in bfloat16 sum to exactly 4096.0
with no config. Sequential bf16 accumulation saturates near 256.

What *does* move the reduction is the dtype of the product, which no config sets: an fp32
`ttnn.multiply` takes `inner` from 2.0026e-03 to 3.4623e-04, 5.8x. It does not help `dx`
(3.4602e-03 -> 4.1692e-03), because the bf16 rounding of the `subtract` below dominates.

## 2. Under the shipped default the line does not execute at all

`MODELAB_route_on.json` — the real OF3 trunk, crop 128, 1 cycle, taped forward and backward,
`exact_training` ON, `ops_active ["softmax","layer_norm"]`, 2,531 declared weights, 1,648 with
gradients:

    softmax_bw_inner fires      0
    ag.triangle_attention       0
    _v_sdpa                     0
    host_f64_softmax          318

All 318 of the trunk's softmaxes go through the taped **verb**, which `_EXACT_OPS` answers with
`host_f64_softmax`, whose backward computes `inner` in host float64 at `autograd.py:1412-1426`.
The trunk never enters `ag.triangle_attention` under a tape — measured at crop 128, 256 and 384,
`triangle_attention` 0 and `_v_sdpa` 0 at all three. Inside `exact_training(False)` the line fires
111 times per backward, through `taped_ttnn._v_softmax`.

`REACH.json` is the same question per caller, each with the control arm beside it:

| caller | shipped default | `exact_training(False)` |
|---|---|---|
| `taped_ttnn._v_softmax` (the verb) | 0 fires | 1 |
| `autograd.softmax` (exported, no in-tree caller) | 1 | 1 |
| `_v_sdpa` -> `triangle_attention` | 1 | 1 |
| `triangle_attention` direct | 1 | 1 |

## 3. What the change moves: nothing, against a floor that is enormous

`VJP.json`. The float64 reference's rule is validated first by central finite differences in the
same run: rel L2 5.90e-11, cosine 1.0. Then the whole `autograd.triangle_attention` VJP, shipped
against fixed, one process, one device open:

    dq  rel L2 4.0437e-03   dk 4.0613e-03   dv 2.8407e-03   dbias 4.0565e-03

bit-identical between the arms to every digit. `TIGHT.json` drives the cancellation from 0.58 to
6.35 cancelled digits, where `dx` is 115.68x wrong: **bit-identical at every rung**, so the zero
is a property of the op and not of OpenFold3's magnitudes.

`MODELAB_*.json` at model scale, per parameter path:

* crop 128: ship vs fix **1,648 / 1,648 bit-identical**, five comparisons over two runs, and the
  ten-pair matrix of five arms is bit-identical throughout.
* crop 384: the backward does not reproduce **itself**. Ship against ship reads 199 / 1,648
  bit-identical, worst rel L2 **1.8758** at `trunk.msa_module.blocks.2.pwa.z_norm_bias`, min
  cosine -0.358. Ship against fix sits inside that band (199-487 / 1,648, worst 1.84-2.34), and
  fix against fix is no tighter (963 / 1,648, worst 1.2253). No arm comparison at this crop can
  resolve a change this small, in either direction.

`gradcompare.py` already documents `*.pwa.z_norm_bias` as not reproducible run to run. At crop 384
on pc that instability is the dominant term in the whole tensor set, which is worth knowing before
anyone grades a gradient lever at that crop here.

## The change that was made

`tt_bio/autograd.py` gains nine comment lines and no code. `ast_equal.py` proves that: the parsed
syntax tree is identical to `origin/main`'s, and its control — the one-line config change this row
was dispatched to evaluate — is correctly reported as different.

## Files

    vjp.py       FD validation, the reduction alone, the triangle-attention VJP  -> VJP.json
    control.py   does the config reach ttnn.sum at all                           -> CONTROL.json
    control2.py  is it honoured and already at its ceiling, or ignored           -> CONTROL2.json
    reach.py     which of the three callers fire, per exactness arm              -> REACH.json
    tight.py     six decades of cancellation tightness                           -> TIGHT.json
    modelab.py   the OF3 trunk, every arm pair, per parameter path               -> MODELAB_*.json
                 (MODELAB_exact_{on,off}.json predate the route counters the later runs carry)
    ast_equal.py the edit is comment-only, with a control
