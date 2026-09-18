#!/usr/bin/env python3
"""The head-major writer's tile map, checked on the CPU against the op it deletes.

`MM_SPLIT_TILE_ID` in `tt_bio/kernels/triatt/matmul_dataflow_common.hpp` is the whole lever: the
qkv matmul's writer sends output tile *(row, tidx)* of chunk *t* to the address tile
*(batch, head, row, channel)* of q, k or v, so `nlp_create_qkv_heads` never runs. Nothing moves
inside a tile, which is why the result is bit-exact -- but only if the address arithmetic is right.

Two checks, neither of which needs a device:

  MACRO   the header is compiled by g++ with the same defines the host passes and evaluated over
          every (row, tidx) of each shape. Catches a typo in the macro text itself.
  LAYOUT  a tiled buffer is built in torch, the writer's tile map is applied, the destination is
          de-tiled, and the result is compared with `x.reshape(B, S, H, D).permute(0, 2, 1, 3)` --
          what the stock `linear -> nlp_create_qkv_heads` pair produces -- by `torch.equal`.

Shapes are the four executed signatures from `perf/c12_tail_screen/leads.json` plus the triangle
attention's own shapes, which must not move: HEAD_MAJOR_DT defaults to 1 and the expression has to
collapse to what ships today.

`equals_plain_writer` is the atom block's q reading in one column: at MT = 1 and DT = 1 the
head-major id IS the plain writer's, so that projection is head-major by choosing its destination's
shape and compiles no define. It is True at `atom_q_512aa` and False everywhere else, which is the
control on the claim rather than a restatement of it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

TILE = 32
HDR = Path(__file__).resolve().parents[2] / "tt_bio" / "kernels" / "triatt"

# (name, batch, seq, n_heads, padded_head_dim, n_chunks) -- padded_head_dim in ELEMENTS.
# `seq` is the rows one batch contributes to the matmul's M axis.
SHAPES = [
    # --- the diffusion / trunk signatures this row converts -----------------------------------
    ("dit_token_512aa", 1, 512, 16, 64, 3),     # 4800 progs, 0.20470 s   DiffusionModule
    ("apb_trunk_512aa", 1, 512, 16, 32, 3),     #  264 progs, 0.00581 s   PairformerLayer
    ("atom_kv_512aa", 140, 128, 4, 32, 2),      # the kv half of the 1200-prog atom signature
    ("atom_q_512aa", 140, 32, 4, 32, 1),        # the q half -- one row tile per window, MT = 1
    # --- shipped triangle attention, must be unchanged ----------------------------------------
    ("triatt_512aa_qkv", 512, 512, 8, 32, 3),
    ("triatt_512aa_qkvg", 512, 512, 8, 32, 4),
    ("triatt_298aa_qkv", 298, 320, 8, 32, 3),   # padded seq 320
    # --- a second multi-tile head, to show DT is a parameter and not a special case -----------
    ("dt4_synthetic", 2, 96, 3, 128, 3),
]

# The harness compiles the header's OWN macro text, so a typo in the shipped file fails here.
# `@DEFINES@` is either both defines or HEAD_MAJOR_MT alone, which is how the triangle attention
# ships -- that case has to reach the `#ifndef HEAD_MAJOR_DT` default of 1.
_HARNESS = """
#include <cstdint>
#include <cstdio>
@DEFINES@
@MACRO@
int main() {
    for (uint32_t row = 0; row < @ROWS@; row++)
        for (uint32_t t = 0; t < @D1@; t++)
            printf("%u\\n", MM_SPLIT_TILE_ID(row, t, @D1@));
    return 0;
}
"""


#: Set by --negctrl: one mutation of the macro that must make every multi-tile-head shape fail
#: while leaving the one-tile shapes alone. A check that cannot fail is not a check.
NEGCTRL = False


def _macro_text() -> str:
    """The MM_SPLIT_TILE_ID block as it stands in the shipped header."""
    src = (HDR / "matmul_dataflow_common.hpp").read_text()
    start = src.index("#ifdef HEAD_MAJOR_MT")
    end = src.index("#endif", src.index("#define MM_SPLIT_TILE_ID(row, tidx, logical_d1) ((row)"))
    block = src[start:end + len("#endif")]
    if NEGCTRL:
        # Drop the row term's DT stride: correct at DT = 1, wrong at every DT > 1.
        was = "+ (((row) % HEAD_MAJOR_MT) * HEAD_MAJOR_DT)"
        assert was in block, "negative control lost its anchor in the header"
        block = block.replace(was, "+ (((row) % HEAD_MAJOR_MT))")
    return block


def macro_ids(mt: int, dt: int | None, rows: int, d1: int) -> list[int]:
    """Compile the header's own macro with g++ and evaluate it over every (row, tidx).

    `dt=None` leaves HEAD_MAJOR_DT undefined, which is how every shipped triangle-attention call
    compiles it.
    """
    defines = [f"#define HEAD_MAJOR_MT {mt}"]
    if dt is not None:
        defines.append(f"#define HEAD_MAJOR_DT {dt}")
    body = (_HARNESS.replace("@DEFINES@", "\n".join(defines))
                    .replace("@MACRO@", _macro_text())
                    .replace("@ROWS@", str(rows))
                    .replace("@D1@", str(d1)))
    with tempfile.TemporaryDirectory() as td:
        src, exe = Path(td) / "h.cpp", Path(td) / "h"
        src.write_text(body)
        subprocess.run(["g++", "-O2", "-o", str(exe), str(src)], check=True)
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def closed_form_ids(mt: int, dt: int, rows: int, d1: int) -> list[int]:
    """`((batch * H + head) * MT + row) * DT + channel`, the destination tile grid."""
    out = []
    for row in range(rows):
        for tidx in range(d1):
            b, r = divmod(row, mt)
            h, d = divmod(tidx, dt)
            out.append(((b * (d1 // dt) + h) * mt + r) * dt + d)
    return out


def tile_buffer(x: torch.Tensor) -> torch.Tensor:
    """`x` of shape [M, N] as a tile-linear buffer [M/32 * N/32, 32, 32], tile grid row-major."""
    m, n = x.shape
    return (x.reshape(m // TILE, TILE, n // TILE, TILE)
             .permute(0, 2, 1, 3)
             .reshape((m // TILE) * (n // TILE), TILE, TILE)
             .contiguous())


def untile(buf: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """The inverse of `tile_buffer`."""
    return (buf.reshape(m // TILE, n // TILE, TILE, TILE)
               .permute(0, 2, 1, 3)
               .reshape(m, n))


def check(name, batch, seq, heads, pdim, chunks):
    assert seq % TILE == 0 and pdim % TILE == 0
    mt, dt = seq // TILE, pdim // TILE
    d1 = heads * dt                      # N tiles in one chunk
    # A one-tile head is compiled with HEAD_MAJOR_DT undefined, exactly as it ships today.
    dt_define = None if dt == 1 else dt
    rows = batch * mt                    # M tiles of the whole matmul
    m, n = rows * TILE, chunks * d1 * TILE

    ids_macro = macro_ids(mt, dt_define, rows, d1)
    ids_closed = closed_form_ids(mt, dt, rows, d1)
    macro_ok = ids_macro == ids_closed
    onto = sorted(ids_macro) == list(range(rows * d1))
    # Whether the head-major id IS the plain writer's `row * logical_d1 + tidx`. True exactly when
    # MT = 1 and DT = 1, which is the atom block's q: a 32-row window is one row tile, so that
    # projection is head-major by choosing its destination's shape and needs no define at all.
    plain = ids_macro == [row * d1 + tidx for row in range(rows) for tidx in range(d1)]

    # LAYOUT: one chunk is enough -- the writer's chunk index only selects the destination tensor.
    torch.manual_seed(0)
    x = torch.randn(m, n, dtype=torch.float32)
    layout_ok = True
    for c in range(chunks):
        chunk = x[:, c * d1 * TILE:(c + 1) * d1 * TILE].contiguous()
        src = tile_buffer(chunk)
        dst = torch.empty_like(src)
        for row in range(rows):
            for tidx in range(d1):
                dst[ids_macro[row * d1 + tidx]] = src[row * d1 + tidx]
        got = untile(dst, batch * heads * seq, pdim).reshape(batch, heads, seq, pdim)
        ref = chunk.reshape(batch, seq, heads, pdim).permute(0, 2, 1, 3).contiguous()
        layout_ok &= torch.equal(got, ref)

    return {"shape": name, "batch": batch, "seq": seq, "n_heads": heads, "padded_head_dim": pdim,
            "HEAD_MAJOR_MT": mt, "HEAD_MAJOR_DT": dt,
            "DT_define_passed": dt_define, "n_chunks": chunks,
            "tiles_per_chunk": rows * d1, "macro_equals_closed_form": macro_ok,
            "map_is_a_permutation": onto, "torch_equal_vs_create_heads": layout_ok,
            "equals_plain_writer": plain}


def main() -> int:
    global NEGCTRL
    NEGCTRL = "--negctrl" in sys.argv[1:]
    rows = [check(*s) for s in SHAPES]
    for r in rows:
        print(f"{r['shape']:<20} MT={r['HEAD_MAJOR_MT']:<4} DT={r['HEAD_MAJOR_DT']} "
              f"tiles={r['tiles_per_chunk']:<7} macro={r['macro_equals_closed_form']} "
              f"perm={r['map_is_a_permutation']} torch.equal={r['torch_equal_vs_create_heads']} "
              f"plain={r['equals_plain_writer']}")
    ok = all(r["macro_equals_closed_form"] and r["map_is_a_permutation"]
             and r["torch_equal_vs_create_heads"] for r in rows)
    if NEGCTRL:
        broke = [r["shape"] for r in rows if not r["torch_equal_vs_create_heads"]]
        kept = [r["shape"] for r in rows if r["HEAD_MAJOR_DT"] == 1]
        expect = [r["shape"] for r in rows if r["HEAD_MAJOR_DT"] > 1]
        good = broke == expect
        print(f"\nNEGCTRL: broken {broke}\nNEGCTRL: DT=1 shapes unaffected {kept}")
        print("NEGCTRL PASS" if good else "NEGCTRL FAIL -- the check does not read the row stride")
        return 0 if good else 1
    out = Path(__file__).resolve().parent / "tile_map.json"
    out.write_text(json.dumps({"all_pass": ok, "shapes": rows}, indent=1) + "\n")
    print(f"\n{'PASS' if ok else 'FAIL'} -- {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
