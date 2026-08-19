#!/usr/bin/env python3
"""p68 -- does the landed stack stack? One interleaved fold A/B with the A/A control in front.

Two branches landed independently and both are now in origin/main:

  * wk/rfd3-b8-irreducible-traffic  -- `_PAIR_TRANSITION_L1`, 116.122 -> 109.148 (-6.974)
  * wk/rfd3-matched-batch-denominator-reopen -- `_ATTN_ROW_BLOCK` + `_PAIRBIAS_FUSED`,
    116.150 -> 108.591 (-7.559)

Both were measured against a ~116.1 s/design arm that already had `_CONCAT_ALIGNED` on, so
`_CONCAT_ALIGNED` stays ON in BOTH arms here; the off arm is exactly "neither branch".

Prediction written before the run: the sites are disjoint (pair Transition intermediates, the host
neighbour graph, the DiT pair-bias projection), so the deltas add to within 1 s ->
off ~116.1, on ~101.6 s/design, 1.143x. That is the baseline every later lever is scored against.

Card 1's shipped digest is not on record. Card 2 is 5295e526ebd0b757 and card 0 is
382e7b23ae8cfc75; the divergence is per-card and already explained. This run establishes card 1's,
and from here only the within-card control counts.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p2 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p2 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p68_stack_ab.py \
          perf/p68/stack_ab.json 200 off,off,on,off,on
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

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p68/stack_ab.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "off,off,on,off,on").split(",")
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")   # 9q6y chain A, 585 + 100 binder
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
PREDICTED_ON_S = 101.6      # 116.1 - 6.974 - 7.559, deltas assumed additive
CARD = int(os.environ.get("TT_VISIBLE_DEVICES", "1"))

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def set_arm(on):
    """Both branches at once. `_CONCAT_ALIGNED` is in the 116.1 baseline and stays on."""
    M._PAIR_TRANSITION_L1 = on
    M._PAIRBIAS_FUSED = on
    M._ATTN_ROW_BLOCK = 256 if on else 0


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
    print("[p68] steps=%d card=%d arms=%s  predicted on %.1f s/design"
          % (STEPS, CARD, ",".join(ARMS), PREDICTED_ON_S), flush=True)

    set_arm(False)
    s, dig, n = fold(specs, "/tmp/rfd3_p68_warm")
    print("[p68] warmup fold %.3f s (%s), discarded" % (s, dig[:20]), flush=True)
    warm = round(s, 3)

    rows = []
    for i, arm in enumerate(ARMS):
        set_arm(arm == "on")
        s, dig, n = fold(specs, "/tmp/rfd3_p68_%d" % i)
        rows.append({"arm": arm, "rep": i, "sampler_s": round(s, 3),
                     "cif_sha256_16": dig, "n_cifs": n})
        print("[p68] rep%d %-3s %9.3f s  %d cif  %s" % (i, arm, s, n, dig[:20]), flush=True)

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

    print("\nA/A control (the two consecutive off arms): %s -> spread %s s (%s %%)"
          % (aa, "n/a" if aa_spread is None else "%.3f" % aa_spread,
             "n/a" if aa_spread is None else "%.2f" % (100.0 * aa_spread / off_med)))
    if off_med and on_med:
        d = on_med - off_med
        print("off  median %9.3f s  [%9.3f, %9.3f]  n=%d" % (off_med, off_lo, off_hi, off_n))
        print("on   median %9.3f s  [%9.3f, %9.3f]  n=%d" % (on_med, on_lo, on_hi, on_n))
        print("delta %+.3f s/design (%.4fx)   predicted on %.1f (%+.2f vs prediction)   %s"
              % (d, off_med / on_med, PREDICTED_ON_S, on_med - PREDICTED_ON_S,
                 "INSIDE the A/A band, not a result" if aa_spread and abs(d) <= aa_spread
                 else "outside the A/A band"))
    print("digests %s  ->  %s" % (digs, "BIT-EXACT" if exact else "DIVERGES"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "num_timesteps": STEPS, "seed": SEED, "arms": ARMS,
        "flags_toggled": ["_PAIR_TRANSITION_L1", "_PAIRBIAS_FUSED", "_ATTN_ROW_BLOCK"],
        "flags_on_in_both_arms": ["_CONCAT_ALIGNED"],
        "discarded_warmup_s": warm, "bit_exact": exact, "digests": digs,
        "aa_control_s": aa, "aa_spread_s": aa_spread,
        "off_median_s": off_med, "off_min_s": off_lo, "off_max_s": off_hi,
        "on_median_s": on_med, "on_min_s": on_lo, "on_max_s": on_hi,
        "delta_s": None if not (off_med and on_med) else round(on_med - off_med, 3),
        "ratio": None if not (off_med and on_med) else round(off_med / on_med, 4),
        "predicted_on_s": PREDICTED_ON_S,
        "host": "qb2", "card": CARD, "torch": torch.__version__}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
