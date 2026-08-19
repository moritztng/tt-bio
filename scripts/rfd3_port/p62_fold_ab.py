"""Fold A/B for the two matched-batch-1 levers, with the digest that decides bit-exactness.

Arms alternate off/on/off/on in one process on one lease: the card warms across a session
(P3.19 saw 134.155 -> 115.714 s across four folds), so only adjacent pairs are an honest
comparison. One fold is run and discarded first so no arm pays a kernel compile.

    arm "off"  = shipped: unblocked neighbour graph, 18 per-block pair-bias projections
    arm "on"   = _ATTN_ROW_BLOCK rows at a time, one hoisted [128, 576] pair-bias matmul

argv: <out.json> [rung] [which]   which in {both, rowblock, pairbias} picks what "on" turns on,
so each lever can be landed against its own control before they are combined -- two levers'
deltas are never added arithmetically.

    ~/.coworker/scripts/benchlock.sh rfd3-matched-batch-denominator-reopen -- env \
      TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:rfd3-matched-batch-denominator-reopen \
      PYTHONPATH=$PWD /home/ttuser/tt-bio-dev/env/bin/python3 -u \
      scripts/rfd3_port/p62_fold_ab.py perf/p62/fold_ab_both.json R4 both
"""
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                          # noqa: E402
from tt_bio.rfd3 import model as M                                     # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                            # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p62/fold_ab_both.json")
RUNG = sys.argv[2] if len(sys.argv) > 2 else "R4"
WHICH = sys.argv[3] if len(sys.argv) > 3 else "both"
FIXTURE = (pathlib.Path(RUNG) if RUNG.endswith(".json")
           else pathlib.Path("perf/dsfix/fixtures/rfd3_%s.json" % RUNG))
CKPT = "/home/ttuser/.boltz/rfd3/weights"
STEPS, SEED, BATCH = 200, 42, 1
ROWBLOCK = int(os.environ.get("RFD3_ATTN_ROWBLOCK", "512"))
ARMS = [False, True, False, True]
# the digest four alternating folds agreed on for the published 114.134 cell
PAGE_DIGEST = "382e7b23ae8cfc75"

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def set_arm(on):
    M._ATTN_ROW_BLOCK = ROWBLOCK if (on and WHICH in ("both", "rowblock")) else 0
    M._PAIRBIAS_FUSED = bool(on and WHICH in ("both", "pairbias"))


def fold(tag):
    out_dir = "/tmp/rfd3_p62_%s" % tag
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    specs = json.loads(FIXTURE.read_text())
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=BATCH,
                           batch_size=BATCH, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    return WALLS[0], dig, len(cifs)


def main():
    set_arm(False)
    s, _, _ = fold("warm")
    print("[p62] warmup fold %.3f s, discarded" % s, flush=True)
    warm = round(s, 3)

    rows = []
    for i, on in enumerate(ARMS):
        set_arm(on)
        s, dig, n = fold(str(i))
        rows.append({"arm": "on" if on else "off", "rep": i, "sampler_s": round(s, 3),
                     "cif_sha256_16": dig, "n_cifs": n})
        print("[p62] %s %-3s rep%d %8.3f s  %d cif  %s"
              % (RUNG, rows[-1]["arm"], i, s, n, dig), flush=True)

    def med(name):
        v = sorted(r["sampler_s"] for r in rows if r["arm"] == name)
        return statistics.median(v), min(v), max(v)

    off, off_lo, off_hi = med("off")
    on, on_lo, on_hi = med("on")
    digs = {a: {r["cif_sha256_16"] for r in rows if r["arm"] == a} for a in ("off", "on")}
    exact = len(digs["off"] | digs["on"]) == 1
    matches_page = digs["on"] == {PAGE_DIGEST}
    print("\noff %8.3f s [%.3f, %.3f]   on %8.3f s [%.3f, %.3f]   %.4fx, %+.3f s"
          % (off, off_lo, off_hi, on, on_lo, on_hi, off / on, on - off))
    print("digests off %s  on %s -> %s; matches the published cell's %s: %s"
          % (sorted(digs["off"]), sorted(digs["on"]),
             "BIT-EXACT" if exact else "DIVERGES", PAGE_DIGEST, matches_page))
    print("bar 109.960 s/design: on arm is %s" % ("AT OR BELOW" if on <= 109.960 else "ABOVE"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "which": WHICH, "row_block": ROWBLOCK,
                               "num_timesteps": STEPS, "seed": SEED, "batch": BATCH,
                               "rung": RUNG, "discarded_warmup_s": warm,
                               "bit_exact": exact, "matches_page_digest": matches_page,
                               "off_s_median": round(off, 3), "on_s_median": round(on, 3),
                               "ratio": round(off / on, 4), "bar_s": 109.960,
                               "clears_bar": bool(on <= 109.960),
                               "host": "qb2", "card": 2, "ttnn": "0.68.0",
                               "torch": torch.__version__}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
