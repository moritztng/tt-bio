"""Build `insights.json` -- the single artifact the site consumes.

    python3 scripts/abag_xm_insights/build_insights.py [-o site/data/insights.json]

Deterministic and re-runnable end to end: the only inputs are the frozen parquets and the
fleet cost log, and every random draw comes from the one seeded bootstrap resample in
`core`. Re-run this after the panel completes to refresh every headline number at once.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import core
import q1_selection
import q2_confidence
import q3_epitope
import q4_pareto
import q6_forecast
import q7_antibody


def clean(o):
    """JSON has no NaN or Infinity. Non-finite values become null."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def provenance() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        sha = None
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_sha": sha,
        "dataset": str(core.DATASET),
        "bootstrap": {"B": core.BOOTSTRAP_B, "seed": core.BOOTSTRAP_SEED,
                      "unit": "target, shared resample draw across models and metrics"},
        "models": core.MODELS,
        "top_rung": core.TOP_RUNG,
        "thresholds": {n: c for n, c in core.THRESHOLDS},
    }


SECTIONS = {
    "q1_selection": q1_selection.run,
    "q2_confidence": q2_confidence.run,
    "q3_epitope": q3_epitope.run,
    "q4_pareto": q4_pareto.run,
    "q6_forecast": q6_forecast.run,
    "q7_antibody": q7_antibody.run,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="site/data/insights.json")
    ap.add_argument("--only", nargs="*", choices=sorted(SECTIONS))
    args = ap.parse_args()

    out = {"provenance": provenance()}
    for name, fn in SECTIONS.items():
        if args.only and name not in args.only:
            continue
        print(f"  {name} ...", flush=True)
        out[name] = fn()

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(out), separators=(",", ":"), allow_nan=False))
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
