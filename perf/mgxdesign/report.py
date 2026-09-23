#!/usr/bin/env python3
"""Turn a design walk's JSONL into the AFTER table, with the pre-registered speed bar applied.

The bar is `scripts/speed_bar.py`, registered at 97809f872 before any rung above 1024 existed,
and it is called here rather than reimplemented. What this script owns is only the plumbing the
bar refuses to guess at:

  * `runtime_s`, the FOLD's seconds, never `wall_s` -- docs/speed-bar.md excludes model load and
    process start, and a cold rung's wall time carries a multi-GB checkpoint download.
  * the AICLK MEDIAN of the samples taken DURING that rung. A Wormhole Galaxy chip sits at 500
    MHz idle and 1000 MHz busy, so the median over a compute-dominated rung is the busy clock;
    a rung whose median is the idle clock spent most of its time not computing and is reported
    rather than quietly compared.
  * `identity` as (host, card, commit). The bar VOIDs a comparison whose rungs did not share
    one chip, which on a contended box is a real outcome and not a formality.

    python3 perf/mgxdesign/report.py perf/mgxdesign/rfd3.jsonl --order 3
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from speed_bar import judge  # noqa: E402

BAR_COMMIT = "97809f872"


# A rung that never reached the model is not a measurement of anything, and the two ways that
# happens both leave a row that LOOKS like a ceiling: a contention refusal (the engine declined
# to open a chip someone else holds) and a kill (SIGTERM/SIGKILL while the walk was restarted).
# The walk driver drops the row it just wrote for a contention refusal, but it can only see its
# own last row -- a row left by an earlier driver survives. So the CONSUMER refuses them too,
# which is the check that cannot be bypassed by running report.py on an older file.
NOT_A_MEASUREMENT = "device contention, nothing ran"

# A rung killed at its own budget did not measure a ceiling either -- ladder.py says so itself:
# "a rung budget does not measure the ceiling, it measures how long the operator was willing to
# watch". That is mild on a quiet box, where a budget overrun is a real runtime result, and
# decisive on a loud one: whglx ran at load 743 on 64 cores while these walks went, which
# inflated a measured pxdesign rung 9.0x against its own recorded number. A TIMEOUT taken under
# that load says nothing about capacity, so it is reported as INCONCLUSIVE and never as the
# first failure.
TIMED_OUT = "TIMEOUT"


# ladder.py SIGKILLs every rung it ends itself, at a device throw and at the budget alike, and
# writes one of these lines to the log as it does. So rc -9 alone does not mean the rung never
# ran: the first version of ran() treated it that way and dropped rfd3's 1536 BEFORE failure, the
# one row this walk exists to produce. A kill is refused only when the ladder did not sign it.
LADDER_ENDED = (TIMED_OUT, "FATAL: ended at the throw")


def ran(r: dict) -> bool:
    blob = " ".join(r.get("diag") or []) + (r.get("tail") or "")
    if NOT_A_MEASUREMENT in blob:
        return False
    if r.get("rc") in (-15, -9):
        return any(m in (r.get("tail") or "") for m in LADDER_ENDED)
    return True


def timed_out(r: dict) -> bool:
    return TIMED_OUT in (r.get("tail") or "")


def load(path: pathlib.Path) -> tuple[list[dict], int]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    kept = [r for r in rows if ran(r)]
    dropped = len(rows) - len(kept)
    # One row per rung: a re-run supersedes an earlier attempt at the same size.
    by_size: dict[int, dict] = {}
    for r in kept:
        by_size[r["size"]] = r
    return [by_size[k] for k in sorted(by_size)], dropped


def clock(r: dict):
    a = r.get("aiclk") or {}
    return a.get("median")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=pathlib.Path)
    ap.add_argument("--order", type=int, default=3,
                    help="highest-order op the model runs; 3 for a pair track")
    ap.add_argument("--commit", default=None, help="the tree the walk ran on")
    a = ap.parse_args()
    rows, dropped = load(a.jsonl)
    if dropped:
        print(f"_{dropped} row(s) dropped: the rung never reached the model (contention refusal "
              f"or a kill), so it bounds nothing._\n")
    if not rows:
        print(f"{a.jsonl}: no rungs actually ran")
        return 1
    commit = a.commit or subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
        capture_output=True, text=True).stdout.strip()

    passed = {r["size"]: r for r in rows if r.get("verdict") == "PASS"}
    runtimes = {n: r["runtime_s"] for n, r in passed.items() if r.get("runtime_s")}
    aiclk = {n: clock(r) for n, r in passed.items() if clock(r)}
    identity = {n: (r.get("host"), r.get("card"), commit) for n, r in passed.items()}

    print(f"| rung | verdict | mech | fold s | wall s | AICLK med | card | speed bar |")
    print(f"|---|---|---|---|---|---|---|---|")
    for r in rows:
        n = r["size"]
        bar = ""
        if r.get("verdict") == "PASS" and n > 1024 and n in runtimes:
            fit = {k: v for k, v in runtimes.items() if 512 <= k <= 1024}
            if len(fit) >= 3 and all(k in aiclk for k in [*fit, n]):
                v = judge({**fit, n: runtimes[n]}, n, runtimes[n], order=a.order,
                          aiclk={k: aiclk[k] for k in [*fit, n]},
                          identity={k: identity[k] for k in [*fit, n]})
                bar = v["verdict"] + (f" ({v.get('ratio')}x of {v.get('allowed')})"
                                      if v.get("ratio") else f" — {v.get('why','')}")
            else:
                bar = f"UNGATED ({len(fit)} fit rungs with a clock)"
        elif timed_out(r):
            bar = "INCONCLUSIVE: killed at the rung budget, not refused by the chip"
        elif r.get("verdict") != "PASS":
            bar = "coverage result, not a speed result"
        verdict = "INCONCLUSIVE" if (timed_out(r) and r.get("verdict") != "PASS") \
            else r.get("verdict")
        print(f"| {n} | {verdict} | {r.get('mechanism')} | "
              f"{r.get('runtime_s', '-')} | {r.get('wall_s', '-')} | "
              f"{clock(r) or '-'} | {r.get('card')} | {bar} |")
    cards = {r.get("card") for r in rows if r.get("verdict") == "PASS"}
    if len(cards) > 1:
        print(f"\nNOTE: the passing rungs did not share one chip ({sorted(cards)}), so every "
              f"comparison across them is VOID by the bar's own rule, not merely noisy.")
    print(f"\nBar: scripts/speed_bar.py, pre-registered at {BAR_COMMIT}. Walk commit {commit}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
