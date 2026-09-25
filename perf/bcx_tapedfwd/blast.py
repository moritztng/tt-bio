#!/usr/bin/env python3
"""Which lengths the `_dividing_k_chunks` fix changes, and which it provably cannot.

Host only, no device, no ttnn import: the k ladder is arithmetic on the padded length. A change
inside a route eleven models share needs its blast radius stated as a set of lengths rather than
as "inert elsewhere", so this enumerates it.

The fix is a no-op at every length whose shipped k_chunk already divides the padded length,
because `_dividing_k_chunks` returns `(shipped_k,)` there and the ladder is byte for byte the old
one. It bites only where the shipped k does NOT divide -- and at exactly those lengths the old
ladder served nothing at all, since every q rung inherits `use_padded_mask` from the k chunk.

    blast.py --max 1536
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TILE = 32


def shipped_k(padded: int, cap: int) -> int:
    """`_sdpa_chunks_shipped`'s k pick, reproduced from its own rule: the padded length when it
    fits under the cap, else the widest tile multiple at or below the cap."""
    return padded if padded <= cap else (cap // TILE) * TILE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1536)
    ap.add_argument("--out", default="out/blast.json")
    args = ap.parse_args()

    from tt_bio import tenstorrent as T                      # needs ttnn; import late so --help works
    changed, inert = [], []
    for n in range(TILE, args.max + 1, TILE):
        padded = T._padded_sdpa_len(n)
        sk = T._sdpa_chunks_shipped(n, n)[1]
        ladder = T._dividing_k_chunks(n, n)
        (changed if list(ladder) != [sk] else inert).append(
            {"n": n, "padded": padded, "shipped_k": sk, "ladder": list(ladder),
             "shipped_k_divides": padded % sk == 0})

    print(f"tile-aligned lengths {TILE}..{args.max}: {len(inert)} inert, {len(changed)} changed")
    for r in changed:
        print(f"  n={r['n']:5d} padded={r['padded']:5d} shipped_k={r['shipped_k']:4d} "
              f"-> ladder {r['ladder']}")
    assert all(r["shipped_k_divides"] for r in inert), "an inert length whose k does not divide"
    assert not any(r["shipped_k_divides"] for r in changed), "a changed length whose k divides"
    print("checked: changed <=> shipped k does not divide the padded length")

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"max": args.max, "changed": changed,
                               "inert_count": len(inert)}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
