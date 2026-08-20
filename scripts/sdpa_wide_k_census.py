#!/usr/bin/env python3
"""Which padded sequence lengths the wide-k SDPA ladder (TT_BIO_SDPA_WIDE_K) actually changes.

Reads the ladder off tt_bio.tenstorrent itself rather than restating the arithmetic, so a later
change to `_sdpa_chunks_shipped` / `_dividing_sdpa_chunk_size` / `SDPA_CHUNK_MAX` shows up here.
Device-free: nothing in this file opens a card.

Two token-padding regimes reach `_tri_att_sdpa_at`, and which one a model uses decides which
padded lengths it can land on:

  * 32-align only -- the model hands its raw token count to `Pairformer` and `_padded_sdpa_len`
    tile-aligns it (Protenix-v2, OpenFold3, OpenDDE).
  * PAIRFORMER_PAD_MULTIPLE = 64 -- the `TorchWrapper` modules pad the token dim first
    (Boltz-2 via `PairformerModule` / `TrunkModule` / `MSAModule`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TT_BIO_SDPA_WIDE_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tt_bio import tenstorrent as T  # noqa: E402


def ladder(padded: int, wide: bool) -> tuple:
    os.environ["TT_BIO_SDPA_WIDE_K"] = "1" if wide else "0"
    return T._tri_att_k_chunks(padded, padded)


def census(max_len: int) -> dict:
    rows = []
    for padded in range(T.SDPA_CHUNK_TILE, max_len + 1, T.SDPA_CHUNK_TILE):
        off = ladder(padded, False)
        on = ladder(padded, True)
        rows.append({
            "padded": padded,
            "shipped_k": off[0],
            "divides": padded % off[0] == 0,
            "wide_ladder": list(on),
            "affected": on != off,
        })
    return rows


def raw_window(padded: int, pad_multiple: int) -> tuple | None:
    """Raw token counts N whose own padding reaches this padded length, or None if unreachable."""
    lo = padded - pad_multiple + 1
    if lo < 1:
        lo = 1
    if padded % pad_multiple:
        return None
    return (lo, padded)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    rows = census(a.max_len)
    affected = [r for r in rows if r["affected"]]

    print(f"SDPA_CHUNK_MAX={T.SDPA_CHUNK_MAX}  SDPA_CHUNK_TILE={T.SDPA_CHUNK_TILE}  "
          f"PAIRFORMER_PAD_MULTIPLE={T.PAIRFORMER_PAD_MULTIPLE}")
    print(f"padded lengths swept: {T.SDPA_CHUNK_TILE}..{a.max_len} step {T.SDPA_CHUNK_TILE} "
          f"({len(rows)} of them), affected: {len(affected)}")
    print()
    print("padded  shipped_k  divides  wide ladder                 raw N (32-align)  raw N (64-pad)")
    for r in affected:
        w32 = raw_window(r["padded"], 32)
        w64 = raw_window(r["padded"], 64)
        f32 = f"{w32[0]}-{w32[1]}" if w32 else "--"
        f64 = f"{w64[0]}-{w64[1]}" if w64 else "--"
        print(f"{r['padded']:6d}  {r['shipped_k']:9d}  {str(r['divides']):7s}  "
              f"{str(r['wide_ladder']):26s}  {f32:16s}  {f64}")

    unaffected_sample = [r["padded"] for r in rows if not r["affected"]]
    print()
    print(f"unaffected (single-entry ladder, byte-for-byte today): {len(unaffected_sample)} lengths, "
          f"e.g. {unaffected_sample[:8]} ... {unaffected_sample[-4:]}")
    for r in rows:
        if not r["affected"]:
            assert len(r["wide_ladder"]) == 1 and r["wide_ladder"][0] == r["shipped_k"], r
    print("ASSERTED: every unaffected length returns exactly the shipped k_chunk with the flag on.")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"max_len": a.max_len, "sdpa_chunk_max": T.SDPA_CHUNK_MAX,
                       "pairformer_pad_multiple": T.PAIRFORMER_PAD_MULTIPLE,
                       "rows": rows}, fh, indent=1)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
