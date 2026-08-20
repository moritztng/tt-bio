#!/usr/bin/env python3
"""p75 -- the L5a fold gate and A/B: `RFD3_SOFTMAX_BF16` on the whole design.

Same harness as p68/p73. One process, one lease, one benchlock hold, one discarded warmup fold,
arms interleaved so card warmth cannot be mistaken for a lever. Every landed lever stays ON in
both arms -- this scores L5a against the 100.742 s/design baseline p73 established.

The lever is `tt_bio.softmax_generic`: ttnn's own softmax program re-driven through
`ttnn.generic_op` with the output CB declared bf16 and `ttnn.typecast`'s SFPU conversion done on
DST before the pack, which deletes the typecast every fp32 attention site pays and halves
softmax's own write. Eligibility is currently restricted to the shapes that take
`softmax_large_tensor.cpp`, the only kernel whose bf16 pack p74 proved bit-exact.

Run it twice. First at 3 timesteps: that is the digest gate. Then at 200: that is the A/B, and
the A/A control is the two consecutive off arms it opens with.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p2 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p2 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p75_softmax_fold_ab.py \
          perf/p75/smoke3.json 3 off,on,off,on
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
from tt_bio import softmax_generic                                             # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p75/dense_fold_ab.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "off,off,on,off,on").split(",")
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
PREDICTED_DELTA_S = -4.523   # p74: 7.3112 -> 4.7984 ms x 9 calls x 200 steps, atom sites only
CARD = int(os.environ.get("TT_VISIBLE_DEVICES", "1"))

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
    print("[p75] steps=%d card=%d arms=%s  predicted delta %+.3f s/design"
          % (STEPS, CARD, ",".join(ARMS), PREDICTED_DELTA_S * (STEPS / 200.0)), flush=True)

    softmax_generic.set_enabled(False)
    s, dig, n = fold(specs, "/tmp/rfd3_p75_warm")
    print("[p75] warmup fold %.3f s (%s), discarded" % (s, dig[:20]), flush=True)
    warm = round(s, 3)

    rows = []
    for i, arm in enumerate(ARMS):
        softmax_generic.set_enabled(arm == "on")
        before = softmax_generic.SSTATS[0]
        s, dig, n = fold(specs, "/tmp/rfd3_p75_%d" % i)
        served = softmax_generic.SSTATS[0] - before
        rows.append({"arm": arm, "rep": i, "sampler_s": round(s, 3),
                     "cif_sha256_16": dig, "n_cifs": n, "kernel_calls": served})
        print("[p75] rep%d %-3s %9.3f s  %d cif  %s  kernel calls %d"
              % (i, arm, s, n, dig[:20], served), flush=True)

    # An "on" arm that served zero kernel calls is an A/A, not an A/B.
    on_calls = [r["kernel_calls"] for r in rows if r["arm"] == "on"]
    off_calls = [r["kernel_calls"] for r in rows if r["arm"] == "off"]
    arms_real = bool(on_calls) and min(on_calls) > 0 and max(off_calls or [0]) == 0
    print("[p75] arm provenance: on served %s, off served %s -> %s"
          % (on_calls, off_calls, "REAL A/B" if arms_real else "ARMS NOT DISTINCT"), flush=True)

    def stats(name):
        v = sorted(r["sampler_s"] for r in rows if r["arm"] == name)
        return (statistics.median(v), min(v), max(v), len(v)) if v else (None,) * 4

    off_med, off_lo, off_hi, off_n = stats("off")
    on_med, on_lo, on_hi, on_n = stats("on")
    aa = [r["sampler_s"] for r in rows if r["arm"] == "off"][:2]
    aa_spread = abs(aa[1] - aa[0]) if len(aa) > 1 else None
    digs = {a: sorted({r["cif_sha256_16"] for r in rows if r["arm"] == a})
            for a in set(ARMS)}
    exact = len({d for v in digs.values() for d in v}) == 1

    if aa_spread is not None:
        print("\nA/A control: %s -> spread %.3f s (%.2f %%)"
              % (aa, aa_spread, 100.0 * aa_spread / off_med))
    if off_med and on_med:
        d = on_med - off_med
        print("off  median %9.3f s  [%9.3f, %9.3f]  n=%d" % (off_med, off_lo, off_hi, off_n))
        print("on   median %9.3f s  [%9.3f, %9.3f]  n=%d" % (on_med, on_lo, on_hi, on_n))
        print("delta %+.3f s/design (%.4fx)   predicted %+.3f   %s"
              % (d, off_med / on_med, PREDICTED_DELTA_S * (STEPS / 200.0),
                 "INSIDE the A/A band, not a result" if aa_spread and abs(d) <= aa_spread
                 else "outside the A/A band"))
    print("digests %s  ->  %s" % (digs, "BIT-EXACT" if exact else "DIVERGES"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "num_timesteps": STEPS, "seed": SEED, "arms": ARMS,
        "flag": "RFD3_SOFTMAX_BF16", "arms_distinct": arms_real,
        "discarded_warmup_s": warm, "bit_exact": exact, "digests": digs,
        "aa_control_s": aa, "aa_spread_s": aa_spread,
        "off_median_s": off_med, "off_min_s": off_lo, "off_max_s": off_hi,
        "on_median_s": on_med, "on_min_s": on_lo, "on_max_s": on_hi,
        "delta_s": None if not (off_med and on_med) else round(on_med - off_med, 3),
        "ratio": None if not (off_med and on_med) else round(off_med / on_med, 4),
        "predicted_delta_s": PREDICTED_DELTA_S * (STEPS / 200.0),
        "host": "qb2", "card": CARD, "torch": torch.__version__}, indent=2) + "\n")
    print("wrote", OUT)
    if not exact:
        sys.exit(2)


if __name__ == "__main__":
    main()
