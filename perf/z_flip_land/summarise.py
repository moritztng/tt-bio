#!/usr/bin/env python3
"""Render the landing table from whatever sweep/gate artifacts exist. No device, no arguments.

The table is the leg's deliverable and it gets re-rendered every pass rather than hand-edited, so a
row can never drift from the JSON it came from.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS = ("l1_out_refused_n", "l1_layer_norm", "transpose_memory_config", "trimul_throws")


def cells(path: Path):
    R = json.loads(path.read_text())
    for r in R["rows"]:
        if r.get("control_for"):
            continue
        yield R, r


def main() -> int:
    print(f"{'model':<12} {'size':>5} {'tokens':>6} {'pairN':>5} {'chan calls':>10} "
          f"{'SERVED':>7} {'plDDT':>8} {'L1ref':>5} {'ln dram':>7} {'tx':>4} {'throw':>5} {'rc':>3}")
    for p in sorted(HERE.glob("sweep_*.json")):
        for R, r in cells(p):
            size = re.search(r"_(\d+)\.yaml", r["target"])
            print(f"{R['model']:<12} {(size.group(1) if size else '117'):>5} "
                  f"{str(r['n_tokens']):>6} {r['pair_shape'][0]:>5} {r['channel_move_calls']:>10} "
                  f"{r['calls_served']:>7} {str(r['plddt']):>8} {r['l1_out_refused_n']:>5} "
                  f"{r['l1_layer_norm']['dram']:>7} "
                  f"{str(r['transpose_memory_config']).rsplit('.', 1)[-1][:4]:>4} "
                  f"{len(r['trimul_throws']):>5} {('ok' if r['rc_ok'] else 'ERR'):>3}")
            if not r["rc_ok"]:
                print(f"    error: {r['error']}")
    for p in sorted(HERE.glob("*boltzgen*.json")):
        R = json.loads(p.read_text())
        c = R.get("census", {})
        print(f"\nboltzgen {R.get('spec')} / {R.get('protocol')}: "
              f"{c.get('eligible_served_per_design')} served of "
              f"{c.get('channel_move_calls_per_design')} channel moves, "
              f"hook_saw_calls={c.get('hook_saw_calls')}")
        for s in c.get("by_shape", []):
            print(f"    N={s['N']} C={s['C']} out={s['out']} eligible={s['eligible']} "
                  f"calls={s['calls']}")
    for p in sorted(q for d in HERE.glob("logs*") for q in d.glob("gate*.log")):
        txt = p.read_text(errors="ignore")
        verdicts = [l for l in txt.splitlines()
                    if re.search(r"\b(PASS|FAIL|BLOCKED|GATE)\b", l)][-14:]
        print(f"\n--- {p.name} ---")
        for v in verdicts:
            print("   ", v[:150])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
