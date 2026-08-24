#!/usr/bin/env python3
"""p102 -- the per-step u_max distribution, which decides whether the key axis has to be
compiled at its worst case.

p101 sized the block-sparse key axis U by the widest union over the three schedule points p94
happened to save, and the prize fell from +6.223 to +2.826 s/design because U is a compile-time
constant. That is the right number for ONE fixed U. It is the wrong number if U is bucketed,
because only the widest steps need the widest axis and p94's three points cannot say how many
those are.

This runs a full 200-step design with p94's spy on `_create_attention_indices`, and for every
atom index it builds it records u_max per candidate block size -- the scalar only, so a whole
schedule costs a few kB instead of 18 MB a point. u_max comes off a [nb, 6080] bool mask's row
sums (p99's bitmap form), not a per-block torch.unique, so the probe adds well under a
millisecond to a step.

Then it prices the buckets against p100/p101's measured chain(Q, U), interpolated in U between
the two widths each Q was measured at, and reports what a bucketed axis is worth against the
single worst-case axis p101 priced.
"""
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p102/umax_schedule.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
QBLOCKS = (320, 608, 1216, 3040)
NK, K, CALLS_PER_STEP = 6080, 128, 9

# measured chain cost, ms/call, from perf/p100 (early width) and perf/p101 (worst-case width).
# Two widths per Q is enough for a slope; anything outside the bracket is extrapolation and is
# flagged as such in the output.
MEASURED = {
    320:  {2048: 7.5993, 3040: 14.0848},
    608:  {2496: 6.4628, 3648: 9.4108},
    1216: {3264: 6.6896, 4224: 8.3432},
    3040: {3936: 7.9982, 5280: 9.9363},
}
DENSE_MS = 9.9236          # mean of p100/p101's in-round dense controls

_orig = M._create_attention_indices
SEEN = {"n": 0}
ROWS = []


def tile(n):
    return ((n + 31) // 32) * 32


def umax_per_q(idx):
    """u_max for each candidate block size, from a bool mask's row sums."""
    L = idx.shape[0]
    nb_rows = tile(L)
    pad = torch.cat([idx, idx[-1:].expand(nb_rows - L, K)], 0).long()
    out = {}
    for Q in QBLOCKS:
        nb = nb_rows // Q
        if nb * Q != nb_rows:
            continue
        mask = torch.zeros(nb, NK, dtype=torch.bool)
        mask.scatter_(1, pad.reshape(nb, Q * K), True)
        out[Q] = int(mask.sum(1).max())
    return out


def _spy(f, X_L, tok_idx, n_keys, n_seq_neighbours):
    idx = _orig(f, X_L, tok_idx, n_keys, n_seq_neighbours)
    SEEN["n"] += 1
    if idx.shape[-1] == K and idx.shape[-2] > 2000:
        ROWS.append(dict(call_no=SEEN["n"], umax=umax_per_q(idx[0].detach().cpu())))
    return idx


M._create_attention_indices = _spy


def chain_ms(Q, U):
    """Measured chain cost at (Q, U), linear in U between the two measured widths."""
    pts = sorted(MEASURED[Q].items())
    (u0, c0), (u1, c1) = pts[0], pts[-1]
    slope = (c1 - c0) / (u1 - u0)
    return c0 + slope * (U - u0), not (u0 <= U <= u1)


def price(us, Q, buckets):
    """Mean chain cost over the schedule when U is the smallest bucket that fits the step."""
    tot, extrap, over = 0.0, 0, 0
    for u in us:
        fit = [b for b in buckets if b >= u]
        if not fit:
            over += 1
            fit = [max(buckets)]
        c, ex = chain_ms(Q, fit[0])
        tot += c
        extrap += int(ex)
    return tot / len(us), extrap, over


def main():
    specs = json.loads(FIXTURE.read_text())
    out_dir = "/tmp/rfd3_p102"
    os.system("rm -rf %s" % out_dir)
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=1,
                           batch_size=1, verbose=False)
    print("[p102] atom indices %d of %d total index builds, over %d timesteps"
          % (len(ROWS), SEEN["n"], STEPS), flush=True)
    if not ROWS:
        print("[p102] NO ATOM INDEX CAPTURED -- the spy did not match", flush=True)
        return

    report = {}
    for Q in QBLOCKS:
        us = [tile(r["umax"][Q]) for r in ROWS if Q in r["umax"]]
        if not us:
            continue
        us_s = sorted(us)
        n = len(us_s)
        q = lambda p: us_s[min(n - 1, int(p * n))]                       # noqa: E731
        worst = us_s[-1]
        single, _ = chain_ms(Q, worst)
        # bucket sets: the worst case alone, then the median/worst pair, then quartiles
        cands = {
            "single_worst": [worst],
            "p50_worst": sorted({q(0.50), worst}),
            "quartiles": sorted({q(0.25), q(0.50), q(0.75), worst}),
        }
        best_name, best = None, None
        rows = {}
        for name, bs in cands.items():
            mean_ms, extrap, over = price(us, Q, bs)
            prize = (DENSE_MS - mean_ms) * CALLS_PER_STEP * STEPS / 1000.0
            rows[name] = dict(buckets=bs, mean_chain_ms=round(mean_ms, 4),
                              prize_s_per_design=round(prize, 3),
                              extrapolated_steps=extrap, unfitted_steps=over)
            if best is None or mean_ms < best:
                best_name, best = name, mean_ms
        print("\nQ=%4d  u_max(tiled) over %d atom indices: min %d  p25 %d  p50 %d  p75 %d  max %d"
              % (Q, n, us_s[0], q(0.25), q(0.50), q(0.75), worst), flush=True)
        print("       single worst-case axis U=%d -> %.4f ms/call, %+.3f s/design"
              % (worst, single, (DENSE_MS - single) * CALLS_PER_STEP * STEPS / 1000.0), flush=True)
        for name, r in rows.items():
            print("       %-13s %-28s %7.4f ms/call  %+7.3f s/design%s"
                  % (name, str(r["buckets"]), r["mean_chain_ms"], r["prize_s_per_design"],
                     "  (%d extrapolated)" % r["extrapolated_steps"]
                     if r["extrapolated_steps"] else ""), flush=True)
        report["Q%d" % Q] = dict(n_indices=n, u_min=us_s[0], u_p25=q(0.25), u_p50=q(0.50),
                                 u_p75=q(0.75), u_max=worst, best=best_name, buckets=rows,
                                 umax_tiled=us)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(steps=STEPS, seed=SEED, fixture=str(FIXTURE),
                                   n_index_builds=SEEN["n"], n_atom_indices=len(ROWS),
                                   dense_ms=DENSE_MS, calls_per_step=CALLS_PER_STEP,
                                   measured_chain=MEASURED,
                                   card=os.environ.get("TT_VISIBLE_DEVICES"),
                                   host=os.uname().nodename, report=report), indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
