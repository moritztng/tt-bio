# 35 of the engine's 90 matmul call sites pass no `core_grid`, and the hot path is among them

`c10-fold-census` returned **STOP** on the 10.0 s target and then named one lead it could not
resolve. Measured on 27 of 33 matmul keys as an interleaved same-session variant,
`core_grid=CORE_GRID_MAIN` moves the fold's matmul class from **11.88 to 21.38 TFLOP/s** — 11.048 s
to 6.1398 s, a **4.908 s / 6,626 Mcycle** difference. It was scrupulous about what that is not:

> That 4.908 s is the gap between two of MY replay configurations. It is NOT a fold saving and must
> not be entered on the ladder as one: whether each Boltz-2 call site currently passes `core_grid`
> is a code question I did not resolve.

That question needs no chip. This answers it.

## The answer

| | sites |
|---|---|
| `ttnn.linear` / `ttnn.matmul` call sites in `tt_bio/tenstorrent.py` | **90** |
| pass an explicit `core_grid` | 53 |
| **bare — no `core_grid`, no `**kw` splat** | **35** |
| no `core_grid` but a `**kw` splat that could carry one | 2 |

Of the 35 bare sites, **18 are on Boltz-2's default device path**:

`TriangleMultiplication.__call__`, `TriangleAttention.__call__`, `OuterProductMean.__call__` (×2)
and `_small_depth`, `DiffusionTransformer.__call__` (×2), `Diffusion.__call__` (×2),
`AdaLN.s_terms` (×2), `MiniTriangularUpdate._matmul_einsum`, and the shared helpers
`_pair_proj_linear` (×2), `_narrow_proj_linear` (×2), `attn_value_matmul` (×2).

The other 17 are on the `Fp32*` precision path and are counted **separately**, because whether a
default fold reaches it is not established here.

**So the census's 4.908 s is not purely an artifact of its own replay configuration — the engine
really does call these ops both ways.**

Done by AST, not grep: these calls span many lines and sit adjacent to each other, so a grep on the
call name cannot tell which keywords belong to which call. A control builds exactly that trap and
requires the parse to get it right.

## What this does *not* establish

- **It does not price a lever.** A bare site gets ttnn's default grid, which may already be right
  for that shape; the census measured a difference on *its* shapes, not on these sites.
- **It does not map a site to a census key.** The launch-key census records class, output shape and
  K but not the call site.
- It is **exactly the shape of a one-size-tuning defect**, a standing class on this fleet: a grid
  good for one shape can be wrong for another. Any change must be A/B'd **per site**, never applied
  across the board.

## The cheap next step

Read the five keys the census named — `1x140x32x128` K=128 and K=256, `1x512x512x16` K=128,
`1x16x512x256` K=64, and the top-FLOP `1x512x768` K=768 — back to their call sites, A/B only those
currently bare, and price the result as fold seconds in one interleaved session. A short session,
not a table.

    python3 core_grid_sites.py                      # writes core_grid_sites.json
    python3 -m pytest test_core_grid_sites.py -q    # 11 controls
