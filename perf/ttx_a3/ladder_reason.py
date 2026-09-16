#!/usr/bin/env python3
"""Write the one judgement SDPA_FUSED_LARGE_S needs per ladder model, at its lowest rung.

The splice (`--size-ladder-record-lever`) measures the counts and leaves a TODO on every dark
rung. The judgement behind those counts is the same on every model and every card, because the
cap is a token count and not a model property, so it is authored once per model at the lowest
rung and release_gate's own `--size-ladder-fill-reasons` carries it up the ladder with a
`[carried from rung N]` tag. Refuses to write on a rung where the lever FIRED: a lever that
serves calls needs no exemption, and rf3's 1088 rung is above the cap.
"""
import json
import sys
from pathlib import Path

FLAG = "SDPA_FUSED_LARGE_S"
WHY = ("dark by the cap, not by a guard: the above-cap route is gated on "
       "`q_len > _triatt_sdpa._Q_SPLIT_MAX_S` (tt_bio/tenstorrent.py:1844) with the cap at "
       "1024, so no rung at or below 1024 tokens can reach it and the stock chunk ladder "
       "serves every call. `resolved: True` is the control: the flag is default-on and the "
       "census can see it, so served 0 is the cap holding and not an instrument that never "
       "fired.")


def main(frag: Path, card: str) -> int:
    d = json.loads(frag.read_text())
    wrote = []
    for model, entry in ((d.get("cards") or {}).get(card, {}).get("models") or {}).items():
        levers = entry.get("levers") or {}
        rungs = sorted(levers, key=int)
        for rung in rungs:
            e = (levers[rung] or {}).get(FLAG)
            if e is None:
                continue
            if (e.get("served") or 0) > 0:
                print(f"{model}/{rung}: fired (served={e['served']}), no exemption to write")
                break
            reason = str(e.get("reason") or "")
            if reason and not reason.startswith("TODO"):
                print(f"{model}/{rung}: already reasoned, nothing to do")
                break
            e["reason"] = WHY
            wrote.append(f"{model}/{rung}")
            break
    if wrote:
        frag.write_text(json.dumps(d, indent=2) + "\n")
    print("wrote:", ", ".join(wrote) or "nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "p300c"))
