#!/usr/bin/env python3
"""Patch the copied matmul_dataflow_common.hpp so the split writer can place tiles head-major.

The qkv matmul's output tile (i, n) already IS tile (batch i/MT, head n%8, row i%MT) of q, k or v:
head_dim is 32, exactly one tile, so no element moves inside a tile. Only the destination index
changes, and the transaction count, transaction size and arithmetic are all unchanged, so the
result is bit-exact against nlp_create_qkv_heads(minimal_matmul(...)) by construction.

Two call sites compute that index -- the deferred writer and the granular one -- and both get the
same guard, keyed on HEAD_MAJOR_MT (tiles per batch along M). Undefined, the file is byte-identical
in behaviour to the wheel's.
"""
import re
import sys
from pathlib import Path

P = Path(sys.argv[1])
src = P.read_text()

GUARD = """
// --- head-major destination, tt-bio ---------------------------------------------------------
// With HEAD_MAJOR_MT defined, the M axis is read as (batch, row) with HEAD_MAJOR_MT row tiles per
// batch, and the chunk's tile grid as (batch, head, row) instead of (row, head). Same tile, same
// transaction, different address.
#ifdef HEAD_MAJOR_MT
#define MM_SPLIT_TILE_ID(row, tidx, logical_d1) \\
    ((((row) / HEAD_MAJOR_MT) * (logical_d1) + (tidx)) * HEAD_MAJOR_MT + ((row) % HEAD_MAJOR_MT))
#else
#define MM_SPLIT_TILE_ID(row, tidx, logical_d1) ((row) * (logical_d1) + (tidx))
#endif
"""

anchor = '#include "api/dataflow/dataflow_api.h"\n'
assert src.count(anchor) == 1, "anchor for the guard block not found exactly once"
src = src.replace(anchor, anchor + GUARD, 1)

subs = [
    ("uint32_t tile_id_in_chunk = i * chunk_shape.logical_d1 + tile_idx_in_chunk;",
     "uint32_t tile_id_in_chunk = MM_SPLIT_TILE_ID(i, tile_idx_in_chunk, chunk_shape.logical_d1);"),
    ("uint32_t tile_id = m_tile * chunk_shape.logical_d1 + tile_idx_in_chunk;",
     "uint32_t tile_id = MM_SPLIT_TILE_ID(m_tile, tile_idx_in_chunk, chunk_shape.logical_d1);"),
]
for old, new in subs:
    assert src.count(old) == 1, f"expected exactly one {old!r}"
    src = src.replace(old, new, 1)

P.write_text(src)
print(f"patched {P}: {len(subs)} tile-id sites + 1 guard block")
