"""Fold A/B for the tile-aligned concat, with the digest that decides whether it is bit-exact.

The route: gather one combined one-hot (130 real columns padded to 160), concat it with z into 320 so
both pieces are tile multiples, slice back to 258. p53 priced it at 31.335 -> 6.168 ms/call, 5.08x,
50.3 ms/step. It is bit-exact by construction -- gathered 0.0/1.0 values, an exact concat, and a
slice whose padding is contiguous at the end -- so the acceptance test is a byte-identical CIF, not a
tolerance.

Arms alternate off/on/off/on in one process on one lease (`rfd3-p14` rule 4: a blocked A/B is biased
by card warmth). Shipped 200 timesteps, one design per arm, same seed.

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p54_concat_fold_ab.py
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

# argv[2] picks the rung. The concat change is default-ON at every size, not just the page
# fixture, so it needs a digest at a small rung too: the combined table is 160 wide whatever I
# is, and a change tuned at one size behaving differently at another is a named failure mode
# (tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa).
RUNG = sys.argv[2] if len(sys.argv) > 2 else "R4"
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_%s.json" % RUNG)
CKPT = "/home/ttuser/.boltz/rfd3/weights"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p54/concat_fold_ab.json")
STEPS, SEED = 200, 42
# argv[3] is the batch. Small targets ship at 8, and the combined one-hot reshapes to
# (batch, I, I, 160) from a flat gather, so the batch dim is the other axis the change has to
# be checked on. At batch > 1 every design's CIF is digested, not just the first.
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ARMS = [False, True] if BATCH > 1 else [False, True, False, True]

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def main():
    specs = json.loads(FIXTURE.read_text())
    rows = []
    for i, aligned in enumerate(ARMS):
        M._CONCAT_ALIGNED = aligned
        out_dir = "/tmp/rfd3_p54_%d" % i
        os.system("rm -rf %s" % out_dir)
        WALLS.clear()
        rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                               num_timesteps=STEPS, seed=SEED, num_designs=BATCH,
                               batch_size=BATCH, verbose=False)
        cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
        dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
               if cifs else "NO CIF")
        rows.append({"arm": "aligned" if aligned else "shipped", "rep": i,
                     "sampler_s": round(WALLS[0], 3), "cif_sha256_16": dig,
                     "n_cifs": len(cifs)})
        print("[p54] %s b=%d rep%d %-8s %8.3f s  %d cif  %s"
          % (RUNG, BATCH, i, rows[-1]["arm"], WALLS[0], len(cifs), dig[:70]), flush=True)

    def med(name):
        v = [r["sampler_s"] for r in rows if r["arm"] == name]
        return statistics.median(v), (max(v) - min(v))

    off, off_sp = med("shipped")
    on, on_sp = med("aligned")
    digs = {r["arm"]: {x["cif_sha256_16"] for x in rows if x["arm"] == r["arm"]} for r in rows}
    exact = len(digs["shipped"] | digs["aligned"]) == 1
    print("\nshipped %8.3f s (spread %.3f)   aligned %8.3f s (spread %.3f)   %.4fx"
          % (off, off_sp, on, on_sp, off / on))
    print("digests: shipped %s  aligned %s  ->  %s"
          % (sorted(digs["shipped"]), sorted(digs["aligned"]),
             "BIT-EXACT" if exact else "DIVERGES"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "num_timesteps": STEPS, "seed": SEED,
                               "rung": RUNG, "batch": BATCH, "bit_exact": exact,
                               "shipped_s_median": round(off, 3),
                               "aligned_s_median": round(on, 3),
                               "ratio": round(off / on, 4),
                               "host": "qb2", "card": 0, "ttnn": "0.68.0",
                               "torch": torch.__version__}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
