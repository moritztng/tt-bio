#!/usr/bin/env python3
"""Gate split: duo with the old gate against duo with the split gate, on one card.

    report.py out/a1 out/b1 ... [--json out/report.json]

Arms and their numbers come from `perf/bcx_p10_stack5/report.py` (warm rounds, duo counted
pro rata over the common window, AICLK sampled during it). Grouped here by gate mode, with the
gate's hold per round split into its phases, and the equality check of every duo digest
against every serial arm's digest for the same (trajectory, round).
"""
import importlib.util
import json
import pathlib
import statistics
import sys

# stack5's report imports duotraj's as `report`, so this one is loaded under its own name.
_spec = importlib.util.spec_from_file_location(
    "stack5_report", pathlib.Path(__file__).resolve().parents[1] / "bcx_p10_stack5" / "report.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def phases(d):
    """Gate seconds per WARM round by phase, summed over slots.

    The gate is cumulative and its snapshot rides every round boundary, so each slot's warm
    share is its last boundary minus the start of its round 2. Falls back to all rounds when
    the boundaries carry no snapshot (arms run before the snapshot existed)."""
    ev = [e for e in d["events"] if e["kind"] in ("round_start", "round_stop")
          and isinstance(e.get("reach"), dict) and "gate_phases" in e["reach"]]
    tot, n = {}, 0
    for s in sorted({e.get("slot") for e in ev} - {None}):
        mine = sorted((e for e in ev if e.get("slot") == s), key=lambda e: e["t0"])
        begin = next((e for e in mine if e["kind"] == "round_start" and e["round"] >= 2), None)
        if begin is None:
            continue
        last = mine[-1]
        n += sum(1 for e in mine if e["kind"] == "round_start" and e["round"] >= 2)
        a = dict(begin["reach"]["gate_phases"].get(s, {}), held=begin["reach"]["gate_held"].get(s, 0))
        b = dict(last["reach"]["gate_phases"].get(s, {}), held=last["reach"]["gate_held"].get(s, 0))
        for k in b:
            tot[k] = tot.get(k, 0.0) + b[k] - a.get(k, 0.0)
    warm = bool(n)
    if not warm:
        g = d["stamp"].get("gate") or {}
        n = sum(1 for e in d["events"] if e["kind"] == "round_start") or 1
        tot = {"held": sum((g.get("held_s") or {}).values())}
        for per in (g.get("phases_s") or {}).values():
            for k, v in per.items():
                tot[k] = tot.get(k, 0.0) + v
    under = sum(v for k, v in tot.items() if k not in ("wait_outside", "held"))
    out = {k: round(v / n, 3) for k, v in sorted(tot.items())}
    out["host_under_gate"] = round((tot.get("held", 0.0) - under) / n, 3)
    out["rounds"] = n
    out["warm_only"] = warm
    return out


def main():
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    paths = [a for a in sys.argv[1:] if not a.startswith("--") and a != js]
    rows, raw = [], {}
    for p in paths:
        if not (pathlib.Path(p) / "round_events.json").exists():
            print(f"{p}: no round_events.json"); continue
        r, d = S.one(p)
        r["split"] = bool(d["stamp"].get("split_gate"))
        r["phases_per_round"] = phases(d)
        rows.append(r); raw[r["arm"]] = d
    for r in rows:
        c = r.get("clock", {})
        kind = ("duo" if r["interleave"] else "ser") + ("/split" if r["split"] else "/old  ")
        print(f"{r['arm']:4s} {kind} " + (r.get("error") or
              f"round {r['round_s']:.3f} s  n {r['n']}  AICLK med {c.get('aiclk_med')} "
              f"min {c.get('aiclk_min')}  load1 {c.get('load1')}  hwm {r['host_hwm_gb']} GB  "
              f"waited {r['waited_s']}"))
        print(f"       gate/round {r['phases_per_round']}")
    summary = {}
    ok = [r for r in rows if "round_s" in r]
    old = [r for r in ok if r["interleave"] and not r["split"]]
    new = [r for r in ok if r["interleave"] and r["split"]]
    ser = [r for r in ok if not r["interleave"]]
    if old and new:
        a = statistics.mean(r["round_s"] for r in old)
        b = statistics.mean(r["round_s"] for r in new)
        summary.update(duo_old_round_s=a, duo_split_round_s=b, speedup=a / b,
                       ratio_vs_h200=b / S.H200, bar_s=10 * S.H200)
        print(f"\nduo old {a:.3f}  duo split {b:.3f}  speedup {a/b:.4f}x  "
              f"split RATIO {b/S.H200:.2f}x vs H200")
    if ser and (old or new):
        summary["equality"] = {}
        for part in (None, "pred", "grad", "loss"):
            l4 = S.leg4(ser, new or old, raw, part)
            if part and not l4["serial_keys"]:
                continue
            summary["equality"][part or "all"] = l4
            print(f"equality [{part or 'all'}]: {l4['duo_equal_to_serial']} duo outputs equal "
                  f"serial, {l4['differ']} differ, first {l4['first_differ']}; serial arms "
                  f"disagree on {l4['serial_keys_disagreeing_across_serial_arms']} of "
                  f"{l4['serial_keys']} keys")
    if js:
        pathlib.Path(js).write_text(json.dumps({"arms": rows, "summary": summary}, indent=1,
                                               default=str))


main()
