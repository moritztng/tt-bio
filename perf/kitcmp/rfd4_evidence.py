"""Collate a finished RFD4 B200 session into the evidence the campaign gate reads.

Takes the pulled session directory (perf/kitcmp/results/<label>/rfd4/), concatenates every arm's runs.jsonl into
`state/rfd4/evidence/gpu-b200.jsonl`, and prints the per-arm table for `state/rfd4/gpu/B200.md` plus the
per-size ratio against whatever TT numbers are given. Nothing here measures anything: it reads the lines the box
wrote and does arithmetic on them, so re-running it on the same session always gives the same answer.

    python rfd4_evidence.py --session results/rfd4-b200/rfd4 [--tt state/rfd4/fair-tt.json] [--write]

`--write` updates state/rfd4/evidence/gpu-b200.jsonl and state/rfd4/rows/rfd4-fair-gpu.json; without it the
script only prints. A line whose `designs` or `wall_s` is missing, zero or negative is dropped from the rates
and counted in `dropped`, never silently averaged in.
"""
import argparse, json, statistics
from pathlib import Path

STATE = Path("/home/moritz/.coworker/state/rfd4")
# the deck's prices and chip counts, the same ones BC4 uses
GALAXY_CHIPS, GALAXY_USD, B200_GPUS, B200_USD = 32, 160_000, 8, 562_154
PER_CHIP = (GALAXY_CHIPS / GALAXY_USD) / (B200_GPUS / B200_USD)  # 14.054

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True, type=Path, help="pulled /root/kc/rfd4 directory")
ap.add_argument("--tt", type=Path, help="JSON of {size: designs_per_s} measured on one p150a")
ap.add_argument("--write", action="store_true")
a = ap.parse_args()

lines, dropped = [], []
for f in sorted(a.session.rglob("runs.jsonl")):
    for raw in f.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            r = json.loads(raw)
        except ValueError:
            dropped.append({"file": str(f), "raw": raw[:120], "why": "not JSON"})
            continue
        lines.append(r)

warm = {}
for r in lines:
    if r.get("kind") != "warm":
        continue
    if r.get("status") == "FAILED" or not r.get("designs") or not r.get("wall_s", 0) > 0:
        dropped.append({"arm": r.get("arm"), "size": r.get("size"), "why": r.get("error", "no designs or wall_s")})
        continue
    warm.setdefault((r["size"], r["arm"], r["batch"]), []).append(r["designs"] / r["wall_s"])

print(f"{len(lines)} lines, {len(warm)} (size, arm, batch) cells, {len(dropped)} dropped\n")
print(f"| size | arm | batch | n | designs/s (median) | spread |")
print(f"| --- | --- | --- | --- | --- | --- |")
best = {}
for (size, arm, batch), rates in sorted(warm.items()):
    rates.sort()
    med = statistics.median(rates)
    print(f"| {size} | {arm} | {batch} | {len(rates)} | {med:.4f} | {rates[0]:.4f}-{rates[-1]:.4f} |")
    if len(rates) >= 5 and med > best.get(size, (0,))[0]:   # protocol: >= 5 warm batches to be the chosen arm
        best[size] = (med, arm, batch, len(rates), rates[0], rates[-1])

print("\nchosen per size (>= 5 warm batches):")
for size, (med, arm, batch, n, lo, hi) in sorted(best.items()):
    print(f"  {size}: {med:.4f} designs/s  arm={arm} batch={batch} n={n} spread {lo:.4f}-{hi:.4f}")

if a.tt and a.tt.exists():
    tt = json.loads(a.tt.read_text())
    print(f"\nratio = 14.054 x designs_s_p150a / designs_s_b200   (gate 3.00x)")
    for size, (med, arm, *_rest) in sorted(best.items()):
        t = tt.get(str(size)) or tt.get("designs_per_s", {}).get(str(size))
        if not t:
            print(f"  {size}: no TT number yet")
            continue
        ratio = PER_CHIP * t / med
        print(f"  {size}: TT {t:.4f} / B200 {med:.4f} -> {ratio:.2f}x  {'PASS' if ratio >= 3 else 'BELOW GATE'}")

if a.write:
    ev = STATE / "evidence" / "gpu-b200.jsonl"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text("".join(json.dumps(r) + "\n" for r in lines))
    row = json.loads((STATE / "rows" / "rfd4-fair-gpu.json").read_text())
    row["arms"] = sorted({r["arm"] for r in lines if r.get("arm")})
    row["designs_per_s"] = {str(s): round(v[0], 4) for s, v in best.items()}
    row["best_arm"] = max(best.values(), key=lambda v: v[0])[1] if best else None
    (STATE / "rows" / "rfd4-fair-gpu.json").write_text(json.dumps(row, indent=2) + "\n")
    print(f"\nwrote {ev} ({len(lines)} lines) and rows/rfd4-fair-gpu.json")
if dropped:
    print(f"\ndropped ({len(dropped)}):")
    for d in dropped[:20]:
        print(f"  {d}")
