# Pairformer matmul dataflow — qb1 card 1, ttnn 0.67.4, 13x10 grid

`ttnn.linear(core_grid=CORE_GRID_MAIN)` derives `in0_block_w = 1` and `out_block_h = per_core_M` on
the pair-track projections. An explicit 1D program config is 1.22x at `in0_block_w = 1` (bit-exact)
and 1.84x at `in0_block_w = 8` (not).

    TT_VISIBLE_DEVICES=1 PYTHONPATH=. python3 perf/pf_matmul/prodshape.py --c-z 256 --bias-pad 1 --out prodshape_cz256.json
    TT_VISIBLE_DEVICES=1 PYTHONPATH=. python3 perf/pf_matmul/infold_pp.py --model protenix-v2 --bw 1 --out x.json
    TT_VISIBLE_DEVICES=1 PYTHONPATH=. python3 perf/pf_matmul/fold_ab.py --model protenix-v2 --arm off  --out fold_off.json
    TT_VISIBLE_DEVICES=1 PYTHONPATH=. python3 perf/pf_matmul/fold_ab.py --model protenix-v2 --arm bw1  --out fold_bw1.json

Fold A/B, protenix-v2, 298 aa: **31848.3 -> 31333.0 ms, 515 ms/fold, identical CIF sha256.**

Roofs this card: square compute 122.59 TFLOP/s, DRAM copy 392.4 GB/s, and 35.5 TFLOP/s at K=256,
which is the rate that actually bounds a trunk matmul.

| class (production shape) | mt/kt/nt | prod | bw1 | bw8 | minimal_matmul | calls/fold |
|---|---|---:|---:|---:|---:|---:|
| `trimul.out_proj` `(1,298,298,256)@(256,256)` | 2980/8/8 | 0.7036 | 0.5759 | 0.3820 | 0.3559 | 2096 |
| `triatt.out` `(298,298,256)@(256,256)` | 2980/8/8 | 0.6989 | 0.5724 | 0.3693 | 0.3495 | 1048 |
| `triatt.triangle_bias` `(298,298,256)@(256,8)` | 2980/8/1 | 0.4439 | 0.3964 | 0.2181 | 0.2272 | 1048 |

`proj_ab.py` and its two JSON files measured a flattened `102400x256` stand-in. Production passes a
batched tensor, so `m_tiles` is 298x10 = 2980, not 3200. Use `prodshape.py`.

Do not apply the config globally. opendde's `transition.up` (mt=160) is *faster* on production
`core_grid=`, which correctly picks the 2D split there. `triatt.qkv`, `triatt.gate` and
`trimul.in_proj` are already at their shape-achievable rate on `minimal_matmul`.

Plan, mechanism and acceptance checks: `~/.coworker/state/perfwar-pairformer-matmul-dataflow.md`.
