#!/usr/bin/env python3
"""The pre-registered speed bar for rungs above 1024 tokens ("not unreasonably slow").

A rung is judged against the model's OWN curve, not a wall-clock wish. Fit log(runtime) against
log(N) over the rungs the model already folds in [512, 1024]; the new rung may sit above that
line only by the curvature the model's own algorithm can legitimately add, times a noise margin:

    allowed(N) = (N / 1024) ** max(0, order - k_fit) * max(1.25, 1 + 3 * sigma)

``order`` is the highest-order op the model runs: 3 with a pair track (triangle multiplication
and triangle attention are O(N^3)), 2 for a sequence-only transformer. So a model fitted at
k = 2.2 may bend up toward N^3 between 1024 and 1536 without failing, but a chunked path that
adds a host round-trip per row block, recompiles per chunk, or spills to host memory shows as a
step above that envelope and fails. 256 is left out of the fit because its size-independent
cost (prep, confidence, save) flattens the slope.

Verdicts: PASS, FAIL, VOID (the comparison is not a measurement: rungs from different hosts,
chips or commits, or a DURING-sampled AICLK that moved more than 3%), UNGATED (fewer than three
fit rungs). A refused or crashed rung is a coverage result, not a speed result, and never reaches
this function. Rationale and scope: docs/speed-bar.md.
"""
from __future__ import annotations

import json
import math
import sys

FIT_LO, FIT_HI = 512, 1024
MIN_FIT_RUNGS = 3
MARGIN_FLOOR = 1.25
CLOCK_TOL = 0.03


def fit(runtimes: dict[int, float]) -> tuple[float, float] | None:
    """Least-squares slope and intercept of log t on log N over the fit window."""
    pts = [(math.log(n), math.log(t)) for n, t in runtimes.items() if FIT_LO <= n <= FIT_HI and t > 0]
    if len(pts) < MIN_FIT_RUNGS:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    k = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    return k, my - k * mx


def judge(runtimes: dict[int, float], n: int, t: float, *, order: int = 3, sigma: float = 0.0,
          aiclk: dict[int, float] | None = None, identity: dict[int, tuple] | None = None) -> dict:
    """Judge one measured rung ``n`` (seconds ``t``) against the model's fit rungs ``runtimes``.

    ``aiclk`` and ``identity`` (host, chip, commit) are keyed by rung like ``runtimes`` and must
    include ``n``. They are optional only so the arithmetic can be unit-tested; a caller judging a
    real measurement passes both.
    """
    fr = {r: v for r, v in runtimes.items() if FIT_LO <= r <= FIT_HI}
    if identity is not None and len({identity[r] for r in [*fr, n]}) != 1:
        return {"verdict": "VOID", "why": "fit rungs and the new rung differ in host/chip/commit"}
    if aiclk is not None:
        clocks = sorted(aiclk[r] for r in fr)
        med = clocks[len(clocks) // 2]
        drift = max(abs(aiclk[r] - med) / med for r in [*fr, n])
        if drift > CLOCK_TOL:
            return {"verdict": "VOID", "why": f"DURING-sampled AICLK moved {drift:.1%} across the rungs"}
    f = fit(fr)
    if f is None:
        return {"verdict": "UNGATED", "why": f"{len(fr)} rung(s) in [{FIT_LO},{FIT_HI}], need {MIN_FIT_RUNGS}"}
    k, b = f
    predicted = math.exp(b + k * math.log(n))
    allowed = (n / FIT_HI) ** max(0.0, order - k) * max(MARGIN_FLOOR, 1 + 3 * sigma)
    ratio = t / predicted
    out = {"verdict": "PASS" if ratio <= allowed else "FAIL", "k_fit": round(k, 3),
           "predicted_s": round(predicted, 1), "ceiling_s": round(predicted * allowed, 1),
           "ratio": round(ratio, 3), "allowed": round(allowed, 3)}
    if k > order:
        out["note"] = f"k_fit {k:.2f} already exceeds the algorithm's order {order} below 1024"
    return out


def main(argv: list[str]) -> int:
    """``speed_bar.py <runtimes.json> <N> <seconds> [order] [sigma]``, runtimes as {rung: s}."""
    if len(argv) not in (4, 5, 6):
        print(main.__doc__)
        return 2
    rt = {int(k): float(v) for k, v in json.load(open(argv[1])).items()}
    r = judge(rt, int(argv[2]), float(argv[3]), order=int(argv[4]) if len(argv) > 4 else 3,
              sigma=float(argv[5]) if len(argv) > 5 else 0.0)
    print(json.dumps(r))
    return 0 if r["verdict"] in ("PASS", "UNGATED") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
