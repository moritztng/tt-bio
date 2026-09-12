#!/usr/bin/env python3
"""The grid-aware q_chunk pick, checked off-device.

Two things have to hold. It must reproduce the optimum the sweep measured (q_chunk 128 at 512
tokens and 16 heads on a 72-core grid), and it must never hand the kernel a config the shipped
path would not: never wider than `_capped_sdpa_chunk_size`, never under one tile, and never a
chunk that does not divide the padded length -- a ragged q tail is what
`fused-sdpa-ragged-tile-tail` is about, and the shipped cap is allowed one only as a fallback.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.tenstorrent as T                                    # noqa: E402

CASES = [
    # name, q_len, batch*heads, cores, expected q_chunk, shipped q_chunk
    ("token DiT 512 aa, WH 8x9",        512,  16,  72, 128, 256),
    ("token DiT 512 aa, BH 11x10",      512,  16, 110, 128, 256),
    ("token DiT 512 aa, one core",      512,  16,   1, 256, 256),
    ("atom SDPA, 560 heads x 32 q",      32, 560,  72,  32,  32),
    ("token DiT 298 aa (320 padded)",   298,  16,  72, 160, 256),
    ("token DiT 1024 aa",              1024,  16,  72, 256, 256),
    ("8 heads, 512 q, 72 cores",        512,   8,  72,  64, 256),
]


def main() -> int:
    bad = 0
    for name, q, work, cores, want, shipped in CASES:
        assert T._capped_sdpa_chunk_size(q) == shipped, (name, T._capped_sdpa_chunk_size(q))
        got = T._grid_q_chunk(q, work, shipped, cores)
        units = work * (T._padded_sdpa_len(q) // got)
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {name:30s} q_chunk {got:4d} (shipped {shipped:4d}) "
              f"units {units:4d} of {cores:4d} cores")
    for q in range(32, 3000, 32):
        for work in (1, 4, 8, 16, 64, 560):
            for cores in (72, 110, 130):
                cap = T._capped_sdpa_chunk_size(q)
                g = T._grid_q_chunk(q, work, cap, cores)
                assert 32 <= g <= cap, (q, work, cores, g)
                # a narrowed pick always divides the padded length; only the untouched shipped
                # cap is allowed the ragged tail it already has today
                assert g == cap or T._padded_sdpa_len(q) % g == 0, (q, work, cores, g)
    print("  invariants hold over 93 lengths x 6 head counts x 3 grids")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
