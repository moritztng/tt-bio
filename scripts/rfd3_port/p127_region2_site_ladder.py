#!/usr/bin/env python3
"""p127 -- region 2's site and addressable fraction, before any kernel work. No device needed.

`state/rfd3-fusion-programme.md` §10.5 and the p3 brief require p124's shape-of-analysis to run
on region 2 *first*: the largest single estimate left in the model (-7.0 s/design, `fc1`) is worth
nothing if the shapes the lever needs are size-keyed the way L5b's `blk == in0_block_w` was.

Region 2's site is `Transition._swiglu`'s `fc1` = `ttnn.linear(xn, W1, activation="silu")`. Two
things decide what a lever can reach there, and only one of them is a coincidence:

1. **The K axis is constant.** `fc1`'s K is `c_z = 128` = 4 tiles at every design size, so the set
   of possible fp32 accumulation groupings (`in0_block_w` in {1,2,4}) does not move with the token
   count. That is the opposite of L5b, whose `blk = find_max_divisor(Wt, 4)` walked with the axis.
   Whether a *bit-exact* pinned config exists is then a property of ttnn's heuristic at one shape,
   not of a divisibility test, and it is screened per cache key rather than computed here.

2. **The L1-resident branch has an engage gate.** `Transition.__call__` takes the chunked branch
   only for a 4D input with `shape[2] >= _PAIR_TRANSITION_MIN_W` (512). Below that the whole pair
   tensor goes through one `_swiglu` call, `fc1`'s output is a single 494 MB tensor at hidden=512,
   and there is no chunk small enough for an L1-resident output. So the *residency* half of region 2
   has no site under 512 tokens, exactly as L5b had none under ~3425 atoms.

Everything here is the arithmetic of the shipped functions, imported rather than transcribed
(`_pair_transition_chunk_h`, `_PAIR_TRANSITION_*`, `align_tile`). No measurement, so nothing here
is a screen result and nothing here needs a card.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tt_bio.rfd3.tiles import TILE, align_tile
from tt_bio.rfd3.model import (_PAIR_TRANSITION_L1_BYTES, _PAIR_TRANSITION_MIN_W,
                               _pair_transition_chunk_h)

# The eight pair-shaped Transition calls per diffusion step. c_z = 128 for both, n sets hidden:
# pairformer_stack.{0,1}.z_transition is n=4 (model.py:816) and transition_2.{0,1} is n=2
# (model.py:2452); each is called once per recycle and there are two recycles.
C_Z = 128
SITES = [("z_transition", 4 * C_Z, 4), ("transition_2", 2 * C_Z, 4)]
BF16 = 2
FIXTURES = {"R1": 296, "R2": 418, "R3": 514, "R4": 685}


def fc1_keys(tokens, hidden, residents=2):
    """Every distinct `fc1` call shape one `Transition.__call__` presents, with its row count.

    `residents` is the number of chunk-sized L1 tensors live at once inside `_swiglu`: 2 today
    (`b` and `m`), 3 if `fc1`'s output joins them. It only moves the chunk height through
    `_pair_transition_chunk_h`'s byte cap, which is what makes the cap the thing to check before
    proposing an extra resident.
    """
    w_pad = align_tile(tokens)
    chunked = tokens >= _PAIR_TRANSITION_MIN_W
    if not chunked:
        return w_pad, False, [(tokens, tokens)], None
    h = _pair_transition_chunk_h(1, w_pad * residents // 2, hidden, tokens)
    full, tail = divmod(tokens, h)
    rows = [(h, full * h)] + ([(tail, tail)] if tail else [])
    return w_pad, True, rows, h


def price(tokens, hidden, residents=2):
    w_pad, chunked, rows, h = fc1_keys(tokens, hidden, residents)
    gflop = 2 * tokens * w_pad * C_Z * hidden / 1e9
    # fc1's own DRAM traffic: it reads xn once and writes its output once.
    read_mb = tokens * w_pad * C_Z * BF16 / 1e6
    write_mb = tokens * w_pad * hidden * BF16 / 1e6
    return {
        "tokens": tokens, "hidden": hidden, "w_pad": w_pad, "chunked": chunked,
        "chunk_h": h, "n_chunk_calls": -(-tokens // (h or tokens)),
        "keys": [{"batch": b, "rows": r} for b, r in rows],
        "n_distinct_keys": len({b for b, _ in rows}),
        "kt": C_Z // TILE, "nt": hidden // TILE, "mt": w_pad // TILE,
        "body_row_share": round(rows[0][1] / tokens, 5),
        "gflop": round(gflop, 3), "read_mb": round(read_mb, 1), "write_mb": round(write_mb, 1),
        "l1_bytes_per_chunk_out": (h or tokens) * w_pad * hidden * BF16,
    }


def main():
    out = {
        "note": "arithmetic of the shipped functions; no device, not a screen result",
        "min_w_gate": _PAIR_TRANSITION_MIN_W,
        "l1_budget_bytes": _PAIR_TRANSITION_L1_BYTES,
        "kt_is_constant": True,
        "kt": C_Z // TILE,
        "fixtures": {}, "sweep": {},
    }
    for name, tok in FIXTURES.items():
        out["fixtures"][name] = {
            "tokens": tok,
            "residency_site": tok >= _PAIR_TRANSITION_MIN_W,
            "sites": {s: price(tok, hid, 2) for s, hid, _ in SITES},
            "sites_3_residents": {s: price(tok, hid, 3) for s, hid, _ in SITES},
        }

    # Over the token range this model is asked to serve, how much of fc1 does each half reach?
    lo, hi = 40, 1200
    axis = range(lo, hi + 1)
    with_site = [t for t in axis if t >= _PAIR_TRANSITION_MIN_W]
    shares = sorted(price(t, 512, 2)["body_row_share"] for t in with_site)
    no_tail = [t for t in with_site if t % _pair_transition_chunk_h(
        1, align_tile(t), 512, t) == 0]
    out["sweep"] = {
        "token_range": [lo, hi],
        "residency_site_fraction": round(len(with_site) / len(list(axis)), 4),
        "body_row_share_min": shares[0],
        "body_row_share_median": shares[len(shares) // 2],
        "designs_with_no_ragged_tail": len(no_tail),
        "designs_with_site": len(with_site),
        "chunk_h_values_2_residents": sorted({price(t, 512, 2)["chunk_h"] for t in with_site}),
        "chunk_h_values_3_residents": sorted({price(t, 512, 3)["chunk_h"] for t in with_site}),
        "kt_values": sorted({price(t, 512, 2)["kt"] for t in axis}),
        "nt_values": sorted({price(t, h, 2)["nt"] for t in axis for _, h, _ in SITES}),
    }

    dest = pathlib.Path("perf/p127")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "region2_site_ladder.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"Kt = {out['kt']} at every design size; Nt in {out['sweep']['nt_values']}")
    print(f"residency site needs tokens >= {_PAIR_TRANSITION_MIN_W}: "
          f"{out['sweep']['residency_site_fraction']:.1%} of {lo}-{hi} tokens")
    print(f"chunk height, 2 residents {out['sweep']['chunk_h_values_2_residents']}, "
          f"3 residents {out['sweep']['chunk_h_values_3_residents']}")
    print(f"body row share: min {out['sweep']['body_row_share_min']:.4f}, "
          f"median {out['sweep']['body_row_share_median']:.4f}; "
          f"{out['sweep']['designs_with_no_ragged_tail']} of "
          f"{out['sweep']['designs_with_site']} designs have no ragged tail")
    print()
    print(f"{'fixture':8} {'tok':>5} {'site':>5} {'hid':>5} {'w_pad':>6} "
          f"{'h/2res':>7} {'chunks':>7} {'h/3res':>7} {'chunks':>7} {'keys':>5} {'body':>7} "
          f"{'GFLOP':>7} {'wr MB':>7} {'L1/chunk MB':>12}")
    for name, f in out["fixtures"].items():
        for site, p in f["sites"].items():
            q = f["sites_3_residents"][site]
            print(f"{name:8} {p['tokens']:5} {str(f['residency_site']):>5} {p['hidden']:5} "
                  f"{p['w_pad']:6} {str(p['chunk_h']):>7} {p['n_chunk_calls']:7} "
                  f"{str(q['chunk_h']):>7} {q['n_chunk_calls']:7} "
                  f"{p['n_distinct_keys']:5} {p['body_row_share']:7.4f} {p['gflop']:7.1f} "
                  f"{p['write_mb']:7.1f} {p['l1_bytes_per_chunk_out']/1e6:12.1f}   ({site})")
    print("\nwrote perf/p127/region2_site_ladder.json")


if __name__ == "__main__":
    main()
