#!/usr/bin/env python3
"""How far does narrow-q actually reach? Audit every baseline census against the policy function.

`TRIATT_PERSISTENT_MASK` declining on `fill_preconditions` is the pathology narrow-q was written
for -- but it is NOT the only way to fail that check. `triatt_sdpa.sdpa` requires

    nh_per_core == 1 and q_per_core == 1 and bcast_batch
    and not use_padded_mask and NKH == H and NVH == H

and narrow-q only addresses `use_padded_mask`. So a model can be dark on `fill_preconditions` for
a reason this lever cannot touch, and counting those calls as the lever's reach is the
`a-lever-can-fire-and-be-inert` error made at corpus scale.

THE CONTROL IS FREE AND IT IS IN THE DATA. narrow-q is inert exactly where the production q_chunk
divides the padded length -- 256, 320, 384, 512, 768, 1024, 1280, 1536 on the Blackhole grid. A
model that is equally dark at an INERT rung is dark for its own reason, and its darkness at a
firing rung is not evidence for this lever either.

  python3 perf/land_standing/narrowq_reach_audit.py
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "perf" / "sizegate" / "baseline"


def firing_sizes():
    """Sizes where the lever changes the offered ladder, from the policy function itself."""
    sys.path.insert(0, str(ROOT))
    import tt_bio.tenstorrent as T
    T._IS_SMALL_GRID = False                      # Blackhole
    out = {}
    for n in (256, 512, 640, 768, 896, 1024, 1088, 1152, 1280, 1408, 1536):
        got = []
        for on in (False, True):
            T._tri_att_q_chunks.cache_clear()
            T._sdpa_chunks_shipped.cache_clear()
            T._SDPA_NARROW_Q_FALLBACK = on
            got.append(T._tri_att_q_chunks(n, n))
        out[n] = got[0] != got[1]
    return out


def load():
    rows = []
    for f in sorted(glob.glob(str(BASELINE / "census_*.json"))):
        m = re.match(r".*census_(.+)_(\d+)_(\w+)\.json", f)
        if not m:
            continue
        d = json.load(open(f))
        r = next((x for x in (d.get("rows") or [])
                  if x.get("flag") == "TRIATT_PERSISTENT_MASK"), None)
        if not r:
            continue
        rows.append({"model": m.group(1), "rung": int(m.group(2)), "card": m.group(3),
                     "served": r["served"], "declined": r["declined"],
                     "fp": (r.get("rejects") or {}).get("fill_preconditions", 0)})
    return rows


def main():
    fires = firing_sizes()
    rows = load()
    print(f"{len(rows)} census cells, {len(set(r['model'] for r in rows))} models, "
          f"{sorted(set(r['card'] for r in rows))}\n")
    verdicts = {}
    for model in sorted({r["model"] for r in rows}):
        cells = [r for r in rows if r["model"] == model and r["fp"] > 0]
        if not cells:
            continue
        inert_dark = [c for c in cells if not fires.get(c["rung"], False)]
        firing_dark = [c for c in cells if fires.get(c["rung"], False)]
        if inert_dark:
            v = ("NOT narrow-q's -- dark at inert rung(s) "
                 f"{sorted({c['rung'] for c in inert_dark})}, where the lever cannot fire")
        else:
            # Per CARD, never summed across board classes: two cards folding the same shape are
            # two cells, not twice the calls in one fold.
            per_card = {c["card"]: sum(x["fp"] for x in firing_dark if x["card"] == c["card"])
                        for c in firing_dark}
            v = ("CANDIDATE -- dark only at firing rung(s) "
                 f"{sorted({c['rung'] for c in firing_dark})}, "
                 + ", ".join(f"{k} {v2} calls/fold" for k, v2 in sorted(per_card.items())))
        verdicts[model] = v
        print(f"{model:12s} {v}")
    print("\nnarrow-q's reach is exactly the CANDIDATE rows above. Everything else is a different "
          "fill_preconditions term wearing the same reject name.")
    return verdicts


if __name__ == "__main__":
    main()
