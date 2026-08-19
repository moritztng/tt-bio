#!/usr/bin/env python3
"""p65 -- fold A/B for the L1-resident pair Transition, with the A/A control that gates it.

The lever: `Transition.__call__` row-chunks the token pair tensor on dim 1 and keeps `x_norm`,
`fc1`'s output and the gated product in L1, so only `fc3`'s output reaches DRAM. Screened at the
production pair shape in perf/p64/fc2_l1.json: 141.9 -> 105.7 ms/step over the eight calls,
-36.2 ms/step, maxabs 0.0 and torch.equal at both hidden widths.

Predicted landing, written before this ran: -36.2 ms/step x 200 steps = **-7.24 s/design**, i.e.
the shipped arm's median minus 7.24 s. That number is the whole point of the run.

Arms alternate in ONE process on ONE lease under ONE benchlock hold, because the arms drift
downward with card warmth (P3.19 saw 134.155 -> 115.714 across four folds). The first two arms are
both OFF: that is the A/A control, and the published cell's own spread is 0.93 % (+/- 1.06 s), so a
result inside the A/A band is not a result. One fold runs and is discarded before the arms start,
or the first arm pays every kernel compile.

Acceptance is a byte-identical CIF, not a tolerance. Card 2's shipped digest is 5295e526ebd0b757.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-irreducible-traffic -- env TT_VISIBLE_DEVICES=2 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-irreducible-traffic PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p65_fold_ab.py \
          perf/p65/fold_ab.json 200 off,off,on,off,on
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

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p65/fold_ab.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "off,off,on,off,on").split(",")
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")   # 9q6y chain A, 585 + 100 binder
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
PREDICTED_DELTA_S = -7.24

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def fold(specs, out_dir):
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=1,
                           batch_size=1, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    return WALLS[0], dig, len(cifs)


def main():
    specs = json.loads(FIXTURE.read_text())
    print("[p65] steps=%d arms=%s  predicted delta %+.2f s/design"
          % (STEPS, ",".join(ARMS), PREDICTED_DELTA_S), flush=True)

    M._PAIR_TRANSITION_L1 = False
    s, dig, n = fold(specs, "/tmp/rfd3_p65_warm")
    print("[p65] warmup fold %.3f s (%s), discarded" % (s, dig[:20]), flush=True)
    warm = round(s, 3)

    rows = []
    for i, arm in enumerate(ARMS):
        M._PAIR_TRANSITION_L1 = (arm == "on")
        s, dig, n = fold(specs, "/tmp/rfd3_p65_%d" % i)
        rows.append({"arm": arm, "rep": i, "sampler_s": round(s, 3),
                     "cif_sha256_16": dig, "n_cifs": n})
        print("[p65] rep%d %-3s %9.3f s  %d cif  %s" % (i, arm, s, n, dig[:20]), flush=True)

    def stats(name):
        v = sorted(r["sampler_s"] for r in rows if r["arm"] == name)
        return (statistics.median(v), min(v), max(v), len(v)) if v else (None,) * 4

    off_med, off_lo, off_hi, off_n = stats("off")
    on_med, on_lo, on_hi, on_n = stats("on")
    # The A/A control is the two consecutive OFF arms the run opens with.
    aa = [r["sampler_s"] for r in rows if r["arm"] == "off"][:2]
    aa_spread = abs(aa[1] - aa[0]) if len(aa) > 1 else None
    digs = {a: sorted({r["cif_sha256_16"] for r in rows if r["arm"] == a})
            for a in set(ARMS)}
    exact = len({d for v in digs.values() for d in v}) == 1

    print("\nA/A control (the two consecutive off arms): %s -> spread %s s (%s %%)"
          % (aa, "n/a" if aa_spread is None else "%.3f" % aa_spread,
             "n/a" if aa_spread is None else "%.2f" % (100.0 * aa_spread / off_med)))
    if off_med and on_med:
        d = on_med - off_med
        print("off  median %9.3f s  [%9.3f, %9.3f]  n=%d" % (off_med, off_lo, off_hi, off_n))
        print("on   median %9.3f s  [%9.3f, %9.3f]  n=%d" % (on_med, on_lo, on_hi, on_n))
        print("delta %+.3f s/design (%.4fx)   predicted %+.2f   %s"
              % (d, off_med / on_med, PREDICTED_DELTA_S,
                 "INSIDE the A/A band, not a result" if aa_spread and abs(d) <= aa_spread
                 else "outside the A/A band"))
    print("digests %s  ->  %s" % (digs, "BIT-EXACT" if exact else "DIVERGES"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "num_timesteps": STEPS, "seed": SEED, "arms": ARMS,
        "discarded_warmup_s": warm, "bit_exact": exact, "digests": digs,
        "aa_control_s": aa, "aa_spread_s": aa_spread,
        "off_median_s": off_med, "off_min_s": off_lo, "off_max_s": off_hi,
        "on_median_s": on_med, "on_min_s": on_lo, "on_max_s": on_hi,
        "delta_s": None if not (off_med and on_med) else round(on_med - off_med, 3),
        "ratio": None if not (off_med and on_med) else round(off_med / on_med, 4),
        "predicted_delta_s": PREDICTED_DELTA_S,
        "host": "qb2", "card": 2, "torch": torch.__version__}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
