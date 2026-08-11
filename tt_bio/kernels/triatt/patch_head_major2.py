#!/usr/bin/env python3
"""Extend the head-major guard to the read side and the single-output writer (K1b).

K1a moved the split writer, which is what the three-way qkv projection needs. The other half of the
sub-block needs the same transform in two more places, and it is the same two lines each time:

  HEAD_MAJOR_OUT_MT   write_block_sync / write_block_sync_granular -- the gate `g` projection, one
                      output of 8 tile columns, written as (batch, head, row) so the elementwise
                      gate multiply sees the same layout the SDPA already produced.
  HEAD_MAJOR_IN0_MT   read_in0_block_sync -- the `out` projection reads that head-major activation
                      as if it were [S*S, 256], so nlp_concat_heads has nothing left to do.

Three separate guards, not one, so each half can be turned on and measured alone. Undefined, every
one of them is the wheel's own expression.
"""
import sys
from pathlib import Path

P = Path(sys.argv[1])
src = P.read_text()

GUARD = """
// Same transform on the read side and on a single-output write. `logical_d1` is the head count in
// both cases: K tiles for the in0 read of a [batch, head, row, 32] activation, N tiles for the gate
// projection's output.
#ifdef HEAD_MAJOR_IN0_MT
#define MM_IN0_TILE_ID(row, col, logical_d1) \\
    ((((row) / HEAD_MAJOR_IN0_MT) * (logical_d1) + (col)) * HEAD_MAJOR_IN0_MT \\
     + ((row) % HEAD_MAJOR_IN0_MT))
#else
#define MM_IN0_TILE_ID(row, col, logical_d1) ((row) * (logical_d1) + (col))
#endif

#ifdef HEAD_MAJOR_OUT_MT
#define MM_OUT_TILE_ID(row, col, logical_d1) \\
    ((((row) / HEAD_MAJOR_OUT_MT) * (logical_d1) + (col)) * HEAD_MAJOR_OUT_MT \\
     + ((row) % HEAD_MAJOR_OUT_MT))
#else
#define MM_OUT_TILE_ID(row, col, logical_d1) ((row) * (logical_d1) + (col))
#endif
"""

if "MM_IN0_TILE_ID" in src:
    print("already patched")
    raise SystemExit(0)

anchor = "#define MM_SPLIT_TILE_ID(row, tidx, logical_d1) ((row) * (logical_d1) + (tidx))\n#endif\n"
assert src.count(anchor) == 1, "K1a guard block not found"
src = src.replace(anchor, anchor + GUARD, 1)

subs = [
    # read_in0_block_sync -- the only in0 read that is not the local-all-gather slice
    ("""                    uint32_t tile_id = i * shape.logical_d1 + j;
                    noc_async_read_tile(tile_id, tensor_accessor, write_ptr);""",
     """                    uint32_t tile_id = MM_IN0_TILE_ID(i, j, shape.logical_d1);
                    noc_async_read_tile(tile_id, tensor_accessor, write_ptr);"""),
    # write_block_sync
    ("""            uint32_t tile_id = i * shape.logical_d1 + j;
            noc_async_write_tile(tile_id, tensor_accessor, read_ptr);""",
     """            uint32_t tile_id = MM_OUT_TILE_ID(i, j, shape.logical_d1);
            noc_async_write_tile(tile_id, tensor_accessor, read_ptr);"""),
    # write_block_sync_granular
    ("""                uint32_t tile_id = m_tile * shape.logical_d1 + n_tile_id;
                noc_async_write_tile(tile_id, tensor_accessor, out_read_ptr);""",
     """                uint32_t tile_id = MM_OUT_TILE_ID(m_tile, n_tile_id, shape.logical_d1);
                noc_async_write_tile(tile_id, tensor_accessor, out_read_ptr);"""),
]
for old, new in subs:
    assert src.count(old) == 1, f"expected exactly one of {old.splitlines()[0]!r}"
    src = src.replace(old, new, 1)

P.write_text(src)
print(f"patched {P}: {len(subs)} more tile-id sites + 2 guards")
