#!/usr/bin/env python3
"""The size-ladder table, built from the rung.json files rather than transcribed.

Reads every rung under `perf/bgx_size/out/`, pairs each timed rung with its footprint
rung by token axis, and prints the markdown row the state doc and the user-facing range
table are made of. A number that is not in a rung.json does not appear here.

Round 1 of a rung is a compile round and is dropped from the round time; what is quoted
is the median of the rest with the AICLK sampled inside those rounds. A rung whose rounds
do not all run at the same clock says so, because a number without its clock is not a
measurement on Blackhole.
"""
import json
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parent / "out"


def load(d: pathlib.Path):
    f = d / "rung.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return None


def outcome(r: dict) -> str:
    err = r.get("error") or ""
    if err.startswith("StopAfterRounds"):
        return "completes"
    if not err:
        return "campaign finished"
    if "Out of Memory" in err:
        return "REFUSES (allocator)"
    if "MemoryError" in err:
        return "REFUSES (guard)"
    return f"FAILS: {err.split('(')[0][:40]}"


def main():
    timed, foot = {}, {}
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        r = load(d)
        if r is None:
            continue
        (foot if r.get("footprint_probe") else timed)[r["tokens"]] = r

    print("| tokens | target | chains | binder | resident peak GB | free at peak GB | largest free "
          "block | s/round | AICLK in-round | load1 | auto chose | outcome |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tok in sorted(set(timed) | set(foot)):
        t, f = timed.get(tok), foot.get(tok)
        r = t or f
        secs = "-"
        clk = "-"
        ld = "-"
        if t:
            per = t.get("per_round") or []
            body = [p for p in per if p["round"] > 1]      # round 1 compiles
            if body:
                s = sorted(p["seconds"] for p in body)
                secs = f"{s[len(s) // 2]:.2f}"
                if len(s) > 1:
                    secs += f" ({s[0]:.2f}-{s[-1]:.2f})"
                lo = min(p.get("aiclk_min", 0) for p in body)
                hi = max(p.get("aiclk_max", 0) for p in body)
                clk = str(lo) if lo == hi else f"{lo}-{hi}"
                # The box load inside the timed rounds. A design round is a host column and
                # a device column laid end to end, so another job on this box lengthens it
                # even when it is on another card, and a round time whose load is not beside
                # it cannot be compared with one taken on a quiet box.
                ld = f"{max(p.get('load1', 0) for p in body):.1f}"
        peak = fre = lcf = "-"
        if f and f.get("resident_peak_gb"):
            # Decimal GB, which is the unit bcx-bigtarget's curve is quoted in.
            peak = f"{f['resident_peak_gb']:.4f}"
            fre = f"{f['free_at_peak_gb']:.3f}"
            if f.get("largest_free_block_at_peak_mb") is not None:
                lcf = f"{f['largest_free_block_at_peak_mb']:.0f} MB"
        auto = (r.get("auto_would_choose") or ["-"])[0]
        out = outcome(t) if t else outcome(f)
        print(f"| {tok} | {r['target']} | {r['target_chains']} | {r['binder']} | {peak} | "
              f"{fre} | {lcf} | {secs} | {clk} | {ld} | {auto} | {out} |")

    print()
    for tok in sorted(set(timed) | set(foot)):
        for r in (timed.get(tok), foot.get(tok)):
            err = (r or {}).get("error") or ""
            if r and err and not err.startswith("StopAfterRounds"):
                leg = "footprint" if r["footprint_probe"] else "timed"
                print(f"{tok} ({leg}): {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
