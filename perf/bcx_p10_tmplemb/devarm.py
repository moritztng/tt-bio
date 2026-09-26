#!/usr/bin/env python3
"""Run a BindCraft 2 gradient round with the multimer template pair stack on card.

`tt_bio.bindcraft2.predictor(template=True)` is the swap; this only arms it, because
`perf/bcx_round/run_round.py` is the campaign's shared meter and is held for comparability. It
patches `campaign_predictor` to pass `template=True` through, captures the factory so the
swap's own counters land in the artifact, and calls `run_round.main()` unchanged.

`--template 0` is the control arm and takes the identical path with the swap off, so the two
arms of an A/B differ in one keyword.

Nothing here does less of the model's own work: the two blocks still run, on the card instead
of in JAX, and `perf/bcx_p10_tmplemb/vjp.py` grades their VJP against float64.
"""
import argparse
import contextlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"),
           str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--template", type=int, required=True,
                 help="1 runs the template pair stack on card; 0 leaves it in JAX")
_ap.add_argument("--out", required=True)
_known, _rest = _ap.parse_known_args()
sys.argv = [sys.argv[0], "--out", _known.out] + _rest

import bc2_state  # noqa: E402,F401  (puts BindCraft 2 on the path)
from tt_bio import bindcraft2  # noqa: E402

CAPTURED: dict = {}
_orig = bindcraft2.campaign_predictor


@contextlib.contextmanager
def _armed(**kwargs):
    with _orig(template=bool(_known.template), **kwargs) as build:
        CAPTURED["build"] = build
        yield build


bindcraft2.campaign_predictor = _armed

import run_round  # noqa: E402


def main():
    try:
        run_round.main()
    finally:
        build = CAPTURED.get("build")
        swap = getattr(build, "template", None) if build is not None else None
        stamp = {"armed": bool(_known.template),
                 "calls": dict(swap.calls) if swap else None,
                 "seen": dict(swap.seen) if swap else None,
                 "live_tapes": swap.live_tapes() if swap else None}
        events = pathlib.Path(_known.out) / "round_events.json"
        if events.exists():
            doc = json.loads(events.read_text())
            doc["stamp"]["template_device"] = stamp
            events.write_text(json.dumps(doc))
        print(json.dumps({"template_device": stamp}), flush=True)


if __name__ == "__main__":
    main()
