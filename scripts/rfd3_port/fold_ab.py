#!/usr/bin/env python3
"""The RFD3 fold A/B harness, extracted so a lever does not need a sixth copy of it.

p68, p73, p75, p85 and p91 each carry their own transcription of this loop, differing only in
which flag they flip and which fixture they load. This module is that loop, parameterised. The
protocol it enforces is the one those passes converged on and it is not optional:

* one process, one lease, one `benchlock` hold, one **discarded** warmup fold;
* arms **interleaved**, because a fold gets faster as the card warms and an uninterleaved
  `off,off,on,on` reads that drift as the lever;
* the **A/A control reported first** -- the two consecutive `off` reps -- so a delta inside the
  card's own spread is named as no result rather than as a small one;
* **arm provenance**: an `on` arm that served zero calls is an A/A wearing an A/B's label, and
  the report says so instead of printing a delta;
* the CIF digest compared across every rep. A digest change is a failed lever, not a tolerance.

`counter` is a one-element list the lever increments per served call, so provenance is read from
the lever itself rather than inferred from the wall clock.
"""
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import time

import torch

from tt_bio.rfd3 import design as rfd3_design
from tt_bio.rfd3.sampler import RFD3Sampler

WEIGHTS = pathlib.Path.home() / ".boltz" / "rfd3" / "weights"

WALLS = []
_sample = RFD3Sampler.sample


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def _fold(specs, out_dir, steps, seed, weights):
    shutil.rmtree(out_dir, ignore_errors=True)
    WALLS.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(weights), from_pdb=True,
                           num_timesteps=steps, seed=seed, num_designs=1,
                           batch_size=1, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = ("|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs)
           if cifs else "NO CIF")
    return WALLS[0], dig, len(cifs)


def fold_ab(*, flag, set_enabled, counter, fixture, out, steps, arms, tag,
            predicted_delta_s=None, seed=42, weights=WEIGHTS, extra=None):
    """Run `arms` folds of `fixture`, flipping `flag` via `set_enabled`. Returns the record."""
    specs = json.loads(pathlib.Path(fixture).read_text())
    print("[%s] flag=%s steps=%d fixture=%s arms=%s" % (tag, flag, steps, fixture, ",".join(arms)),
          flush=True)
    if predicted_delta_s is not None:
        print("[%s] cost-model prediction (an ESTIMATE, not a measurement): %+.3f s/design"
              % (tag, predicted_delta_s * (steps / 200.0)), flush=True)

    set_enabled(False)
    s, dig, _n = _fold(specs, "/tmp/%s_warm" % tag, steps, seed, weights)
    print("[%s] warmup fold %.3f s (%s), discarded" % (tag, s, dig[:20]), flush=True)
    warm = round(s, 3)

    rows = []
    for i, arm in enumerate(arms):
        set_enabled(arm == "on")
        before = counter[0]
        s, dig, n = _fold(specs, "/tmp/%s_%d" % (tag, i), steps, seed, weights)
        rows.append({"arm": arm, "rep": i, "sampler_s": round(s, 3),
                     "cif_sha256_16": dig, "n_cifs": n, "served_calls": counter[0] - before,
                     "steps": steps})
        print("[%s] rep%d %-3s %9.3f s  %d cif  %s  served %d"
              % (tag, i, arm, s, n, dig[:20], rows[-1]["served_calls"]), flush=True)
    set_enabled(False)

    on_served = [r["served_calls"] for r in rows if r["arm"] == "on"]
    off_served = [r["served_calls"] for r in rows if r["arm"] == "off"]
    arms_real = bool(on_served) and min(on_served) > 0 and max(off_served or [0]) == 0
    print("[%s] arm provenance: on served %s, off served %s -> %s"
          % (tag, on_served, off_served, "REAL A/B" if arms_real else "ARMS NOT DISTINCT"),
          flush=True)

    def stats(name):
        v = sorted(r["sampler_s"] for r in rows if r["arm"] == name)
        return (statistics.median(v), min(v), max(v), len(v)) if v else (None,) * 4

    off_med, off_lo, off_hi, off_n = stats("off")
    on_med, on_lo, on_hi, on_n = stats("on")
    aa = [r["sampler_s"] for r in rows if r["arm"] == "off"][:2]
    aa_spread = abs(aa[1] - aa[0]) if len(aa) > 1 else None
    digs = {a: sorted({r["cif_sha256_16"] for r in rows if r["arm"] == a}) for a in set(arms)}
    exact = len({d for v in digs.values() for d in v}) == 1

    # The A/A control prints BEFORE either arm's median, because it is what decides whether the
    # medians mean anything.
    if aa_spread is None:
        print("\nA/A control: NOT RUN (fewer than two consecutive off reps) -- no noise floor, so "
              "no delta from this run is a result")
    else:
        print("\nA/A control: %s -> spread %.3f s (%.2f %% of the off median)"
              % (aa, aa_spread, 100.0 * aa_spread / off_med))
    if off_med and on_med:
        d = on_med - off_med
        inside = aa_spread is not None and abs(d) <= aa_spread
        print("off  median %9.3f s  [%9.3f, %9.3f]  n=%d" % (off_med, off_lo, off_hi, off_n))
        print("on   median %9.3f s  [%9.3f, %9.3f]  n=%d" % (on_med, on_lo, on_hi, on_n))
        print("delta %+.3f s/design (%.4fx)  ->  %s"
              % (d, off_med / on_med,
                 "INSIDE the A/A band, NOT a result" if inside else "outside the A/A band"))
    print("digests %s  ->  %s" % (digs, "BIT-EXACT" if exact else "DIVERGES"))

    rec = {"flag": flag, "rows": rows, "num_timesteps": steps, "seed": seed, "arms": arms,
           "fixture": str(fixture), "arms_distinct": arms_real,
           "discarded_warmup_s": warm, "bit_exact": exact, "digests": digs,
           "aa_control_s": aa, "aa_spread_s": aa_spread,
           "off_median_s": off_med, "off_min_s": off_lo, "off_max_s": off_hi,
           "on_median_s": on_med, "on_min_s": on_lo, "on_max_s": on_hi,
           "delta_s": None if not (off_med and on_med) else round(on_med - off_med, 3),
           "ratio": None if not (off_med and on_med) else round(off_med / on_med, 4),
           "delta_inside_aa_band": (None if not (off_med and on_med and aa_spread is not None)
                                    else bool(abs(on_med - off_med) <= aa_spread)),
           "predicted_delta_s": (None if predicted_delta_s is None
                                 else predicted_delta_s * (steps / 200.0)),
           "card": int(os.environ.get("TT_VISIBLE_DEVICES", "-1")),
           "torch": torch.__version__}
    rec.update(extra or {})
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print("wrote", out)
    return rec
