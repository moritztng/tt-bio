#!/usr/bin/env python3
"""of3t-angle: stamp the environment into every reading this row produces.

Amendment 1 (orchestrator, pass 414): qb1 is shared. of3t-verbinstall was running device arms
there for the whole window this row built its references in, at load average 13.58 when the
orchestrator measured it and 16.11 at 00:03:19Z. A reading whose host is unrecorded cannot be
compared later against one taken quiet (`a-run-records-its-host-not-its-board`,
`benchmark-log-without-environment-stamp-is-incomparable`), so the host, the board and the
quiet state go into the JSON beside the numbers, not into notes.

Nothing this stamps is a timing claim. The durations that ref_grad.py records are kept, renamed
so they cannot be read as one, and the reason is carried with them.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess


def loadavg_block(path):
    if not path or not os.path.exists(path):
        return {"sampled_DURING": False,
                "why": "no sampler ran during this arm; the co-tenancy below is what is known"}
    s = sorted(float(x) for x in open(path).read().split() if x.strip())
    if not s:
        return {"sampled_DURING": False, "why": "sampler file is empty"}
    return {"sampled_DURING": True, "n": len(s), "min": s[0],
            "median": s[len(s) // 2], "max": s[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--loadavg", default="")
    ap.add_argument("--cotenant", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    d = json.load(open(a.report))
    secs = d.pop("seconds_forward_backward", None)
    env = {
        "host": socket.gethostname(),
        "row": "of3t-angle", "defect": "D242",
        "device_involved": False,
        "board": None,
        "why_no_board_and_no_aiclk":
            "CPU only. No Tenstorrent device is opened by this arm, so there is no board and no "
            "AICLK to sample. The device arms of this row carry both in their own DEV_*.json.",
        "QUIET": False,
        "co_tenant": a.cotenant or "unknown",
        "loadavg": loadavg_block(a.loadavg),
        "NO_TIMING_CLAIM_FROM_THIS_HOST":
            "Amendment 1: this row is a co-tenant on qb1 and takes no wall-clock or perf claim "
            "there. The deliverable is an accuracy decomposition (rel, r, cos, angle), every "
            "part of which is load-insensitive, so the reading stands and the duration does not.",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip() or None,
    }
    if secs is not None:
        env["seconds_forward_backward_NOT_A_TIMING_CLAIM"] = secs
    d["environment"] = env
    out = a.out or a.report
    json.dump(d, open(out, "w"), indent=1)
    print(json.dumps({"stamped": out, "host": env["host"], "quiet": env["QUIET"],
                      "loadavg": env["loadavg"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
