#!/usr/bin/env python3
"""What caps the interleave: the card's gate hold per trajectory-round at N=1 against N>1.

    knee.py DIR [DIR ...]      each DIR holds a round_events.json (perf/bcp_roofline/out/<arm>)

If the GIL were the knee, the threads waiting on it would stretch the seam that holds the gate,
and the hold per trajectory-round would grow with N. If the card is the knee, the hold stays what
one trajectory needs on the card and N only fills the gate's idle time. Rounds 1 and 2 of every
slot are dropped (compile, lazy trunk load); a gradient round is nine seams.
"""
import json
import pathlib
import sys

for p in sys.argv[1:]:
    d = json.loads((pathlib.Path(p) / "round_events.json").read_text())
    n = d["stamp"].get("trajectories", 1)
    holds = sorted((e for e in d["events"] if e["kind"] == "gate" and e["phase"] == "hold"),
                   key=lambda e: e["t0"])[18 * n:]
    rounds = len(holds) / 9
    held = sum(e["t1"] - e["t0"] for e in holds)
    span = holds[-1]["t1"] - holds[0]["t0"]
    print(f"{pathlib.Path(p).name} N={n}: gate hold {held / rounds:.3f} s per trajectory-round "
          f"over {rounds:.0f} rounds, gate busy {held / span * 100:.1f} % of the window")
