"""Summarise a client.py --log stream: folds per chip, AICLK seen, preemptions, visitor latency.

    python3 demo/booth/engine/summarize_stream.py runs/multichip-0-3/client.jsonl
"""
import json
import sys
from collections import defaultdict

msgs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
per = defaultdict(lambda: {"done": 0, "seconds": [], "aiclk_status": [], "errors": defaultdict(int)})
start, first_frame, frames = {}, {}, defaultdict(int)
for m in msgs:
    t, chip = m["type"], m.get("chip")
    if t == "status":
        for c in m["chips"]:
            if c["state"] == "busy" and c["aiclk_mhz"]:
                per[c["chip"]]["aiclk_status"].append(c["aiclk_mhz"])
    elif t == "fold_start" and m.get("source") == "live":
        start[m["id"]] = (m["t_wall"], m["kind"], m["n_res"], chip)
    elif t == "frame" and chip is not None:
        frames[m["id"]] += 1
        first_frame.setdefault(m["id"], m["t_wall"])
    elif t == "fold_done" and chip is not None:
        p = per[chip]
        p["done"] += 1
        p["seconds"].append((m["kind"], m["n_res"], m["seconds"], (m.get("aiclk_mhz") or {}).get("median")))
    elif t == "fold_error" and chip is not None:
        per[chip]["errors"][m["reason"]] += 1
for chip in sorted(per, key=lambda c: (c is None, c)):
    p = per[chip]
    a = sorted(p["aiclk_status"])
    print(f"chip {chip}: {p['done']} folds done, errors {dict(p['errors'])}, "
          f"AICLK while busy (1 Hz status) min/median/max {a[0] if a else '-'}/{a[len(a)//2] if a else '-'}/{a[-1] if a else '-'} n={len(a)}")
    for kind, n, s, clk in p["seconds"]:
        print(f"    {kind:8s} {n:4d} aa  {s:7.3f} s  AICLK median {clk}")
for jid, (t0, kind, n, chip) in start.items():
    if kind == "visitor":
        ff = first_frame.get(jid)
        print(f"visitor {jid} {n} aa on chip {chip}: first frame {ff - t0:.2f} s after fold_start, "
              f"{frames[jid]} frames" if ff else f"visitor {jid} {n} aa: no frames")
