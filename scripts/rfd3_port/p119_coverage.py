"""p119 -- how many atom counts does the block-sparse arm actually fire on?

The release-gate on-arm run came back `0 blocked, 1791 dense-fallback`: the arm was DARK on the
gate's own fixture and `gate=True` proved nothing about it. Cause is `block_sparse.plan()`'s
`if nb_rows % q_block: return None` -- Q must DIVIDE the tile-padded atom axis, which is a much
stronger constraint than the multiple-of-32 one the design notes state. Host-only, no device.
"""
import json
import pathlib
import sys

TILE = 32
Q = 1216
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p119/coverage.json")


def tile(n):
    return -(-n // TILE) * TILE


def live(atoms, q=Q):
    return tile(atoms) % q == 0


# R4, the fixture every measurement in this lineage used, and the gate's own binder.
KNOWN = {"rfd3_R4 (p103/p105/p106/p107 fixture)": 6051,
         "examples/rfd3_binder.json (release-gate Anchor 1)": 1350}

# The arm fires iff ceil(atoms/32) is a multiple of Q/32 tiles.
q_tiles = Q // TILE
span = range(256, 12001)
hits = [a for a in span if live(a)]
res = {
    "q_block": Q,
    "q_tiles": q_tiles,
    "rule": "arm is live iff ceil(atoms/32) %% %d == 0" % q_tiles,
    "known_fixtures": {k: {"atoms": v, "padded": tile(v), "padded_tiles": tile(v) // TILE,
                           "arm_live": live(v)} for k, v in KNOWN.items()},
    "census_span": [span.start, span.stop - 1],
    "n_atom_counts": len(span),
    "n_live": len(hits),
    "live_fraction": round(len(hits) / len(span), 5),
    "live_windows_atoms": [[tile(a) - TILE + 1, tile(a)]
                           for a in sorted({tile(h) for h in hits})],
}
# Which q_block values would cover a given atom count: divisors of the padded axis that are
# multiples of 32. This is the size of the fix, not a proposal to make it.
for name, v in KNOWN.items():
    n = tile(v)
    res["known_fixtures"][name]["q_candidates_mult32"] = [
        d for d in range(TILE, n + 1, TILE) if n % d == 0]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
