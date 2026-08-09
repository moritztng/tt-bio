# Pairformer matmul dataflow — qb1 card 1, ttnn 0.67.4, 13x10 grid

`proj_ab.py` runs every Pairformer matmul class at 298 aa (N_tok padded to 320) through
`ttnn.linear(core_grid=)`, `ttnn.experimental.minimal_matmul`, and a sweep of explicit
`MatmulMultiCoreReuseMultiCast1DProgramConfig`s. Roofs are measured in the same process.

    TT_VISIBLE_DEVICES=1 PYTHONPATH=. python3 perf/pf_matmul/proj_ab.py --cz 256
    TT_VISIBLE_DEVICES=1 PYTHONPATH=. python3 perf/pf_matmul/proj_ab.py --cz 384

Roofs this card: compute square 122.56 TFLOP/s, DRAM read+write 391.9 GB/s, and 35.47 TFLOP/s at
K=256, which is the rate that actually bounds a trunk matmul.

The four classes that reach ttnn through `core_grid=` are 1.70-2.09x off. `core_grid=` resolves to
`in0_block_w=1, out_block=per_core`; `1d_bw1_obh25_obw8` reproduces it bit-exactly and to 0.01% in
time, which is how we know.

| class | shape | prod | tuned | x | n/blk | ms/fold |
|---|---|---:|---:|---:|---:|---:|
| trimul.out_proj (pv2) | 102400x256 @ 256x256 | 0.7521 | 0.3835 | 1.961 | 4 | 708 |
| triatt.out (pv2) | 102400x256 @ 256x256 | 0.7518 | 0.3855 | 1.950 | 2 | 352 |
| triatt.triangle_bias (pv2) | 102400x256 @ 256x32 | 0.4811 | 0.2302 | 2.090 | 2 | 241 |
| trimul.out_proj (opendde) | 102400x384 @ 384x384 | 1.1384 | 0.5854 | 1.945 | 4 | 1062 |
| triatt.out (opendde) | 102400x384 @ 384x384 | 1.1377 | 0.5853 | 1.944 | 2 | 530 |

Bit-exact subset (`out_block` only, `in0_block_w` left at 1): 534 ms/fold on protenix-v2, 945 on
opendde, `torch.equal` against production at the op.

Do not apply this globally. opendde's `transition.up` (mt=160) is *faster* on production
`core_grid=`, which correctly picks the 2D split there; a blanket swap costs 522 ms/fold.
`triatt.qkv`, `triatt.gate` and `trimul.in_proj` are already at their shape-achievable rate on
`minimal_matmul` and every tuned linear is slower.

Plan, mechanism and acceptance checks: `~/.coworker/state/perfwar-pairformer-matmul-dataflow.md`.
