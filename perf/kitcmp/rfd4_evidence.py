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
import argparse, calendar, json, statistics, time
from pathlib import Path

STATE = Path("/home/moritz/.coworker/state/rfd4")
# the deck's prices and chip counts, the same ones BC4 uses
GALAXY_CHIPS, GALAXY_USD, B200_GPUS, B200_USD = 32, 160_000, 8, 562_154
PER_CHIP = (GALAXY_CHIPS / GALAXY_USD) / (B200_GPUS / B200_USD)  # 14.054

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True, type=Path, action="append", default=[],
                help="pulled /root/kc/rfd4 directory; repeat once per rented box")
ap.add_argument("--tt", type=Path, help="JSON of {size: designs_per_s} measured on one p150a")
ap.add_argument("--write", action="store_true")
ap.add_argument("--append", action="store_true",
                help="add these sessions' lines to gpu-b200.jsonl, keep every other session's, leave the row alone")
a = ap.parse_args()


def load_smi(path):
    """nvidia-smi -l 1 samples as (epoch, sm_mhz, power_w, util_pct), sorted."""
    out = []
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        p = [c.strip() for c in raw.split(",")]
        if len(p) < 7:
            continue
        try:
            ts = calendar.timegm(time.strptime(p[0].split(".")[0], "%Y/%m/%d %H:%M:%S"))
            out.append((ts, float(p[1].split()[0]), float(p[3].split()[0]), float(p[6].split()[0])))
        except (ValueError, IndexError):
            continue
    return sorted(out)


def window(smi, end_iso, wall_s):
    """Clock and load actually seen DURING a batch, from the 1 Hz log.

    The per-record `sm_mhz`/`util_pct`/`power_w` that rfd4_gpu.py stamps are a single sample taken at the batch
    boundary, where the GPU is idle between batches: every warm line on 2026-10-21 reads `util_pct: 0` while the
    1 Hz log shows 97-100 % and 983 W of a 1000 W limit through the compute. Published as-is that would read as
    a measurement taken on an idle card -- the exact inverse of the truth -- so the window is recomputed here.
    `busy_pct` is the share of the wall clock the GPU was above 50 % util, which is what says whether a batch is
    host-bound and therefore whether a bigger batch is still owed.
    """
    if not smi or not end_iso or not wall_s:
        return {}
    try:
        t1 = calendar.timegm(time.strptime(end_iso, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return {}
    s = [x for x in smi if t1 - wall_s <= x[0] <= t1]
    if not s:
        return {}
    util = [x[3] for x in s]
    busy = [x for x in s if x[3] > 50]
    return {"sm_mhz_window": statistics.median([x[1] for x in s]),
            "power_w_peak": max(x[2] for x in s),
            "util_pct_median": statistics.median(util), "util_pct_peak": max(util),
            "busy_pct": round(100.0 * len(busy) / len(s)), "smi_samples": len(s),
            "sm_mhz_busy_median": statistics.median([x[1] for x in busy]) if busy else None}


lines, dropped = [], []
for session in a.session:
    # Each rented box logged its own clock at 1 Hz, so a line is windowed against the smi.csv of the box that
    # produced it. The session label rides on every line: a speed cell belongs to ONE box (below), because
    # pooling two rentals into one rate would publish a number no single machine ever ran.
    smi = load_smi(session / "smi.csv")
    label = session.parent.name
    for f in sorted(session.rglob("runs.jsonl")):
        # The phase that wrote the line: "final-compile-cueq", "ext-compile-cueq", "nw16-compile-cueq",
        # "reference/binder". rfd4_arms.py puts every run under <tag>/<size>-b<batch>/runs.jsonl, so the tag is
        # the run directory's parent, relative to the session root.
        tag = str(f.parent.parent.relative_to(session))
        for raw in f.read_text().splitlines():
            if not raw.strip():
                continue
            try:
                r = json.loads(raw)
            except ValueError:
                dropped.append({"file": str(f), "raw": raw[:120], "why": "not JSON"})
                continue
            r["tag"], r["session"] = tag, label
            r.update(window(smi, r.get("t"), r.get("wall_s")))
            lines.append(r)

warm, load = {}, {}
for r in lines:
    if r.get("kind") != "warm":
        continue
    if r.get("status") == "FAILED" or not r.get("designs") or not r.get("wall_s", 0) > 0:
        dropped.append({"arm": r.get("arm"), "size": r.get("size"), "why": r.get("error", "no designs or wall_s")})
        continue
    # A speed number is the PD-L1 binder at a gated size (PROTOCOL.md). The reference families run at
    # size=256 batch=8 under the same arm, so keying a cell on (size, arm, batch) alone pooled ten different
    # molecules into the published 256 rate. They stay in the evidence file; they are not a speed cell.
    if r["tag"].startswith("reference/"):
        continue
    # num_workers is part of the configuration, not of the phase: the sweep measures the same arm and batch
    # at 4/16/48, so a cell keyed without it would average three different recipes into one rate.
    nw = next((o.split("=", 1)[1] for o in r.get("overrides", []) if o.startswith("num_workers=")), "default")
    r["num_workers"] = nw
    key = (r["size"], r["arm"], r["batch"], nw, r["session"])
    warm.setdefault(key, []).append(r["designs"] / r["wall_s"])
    load.setdefault(key, []).append(r)

print(f"{len(lines)} lines, {len(warm)} (size, arm, batch, workers, box) speed cells, {len(dropped)} dropped\n")
print("| size | arm | batch | workers | box | n | designs/s (median) | spread | clock (MHz, busy) | util med/peak | busy % | peak W |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
best = {}
for (size, arm, batch, nw, box), rates in sorted(warm.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], str(kv[0][3]), kv[0][4])):
    rates.sort()
    med = statistics.median(rates)
    w = load.get((size, arm, batch, nw, box), [])
    g = lambda k: [r[k] for r in w if r.get(k) is not None]
    mhz, um, up, bp, pw = g("sm_mhz_busy_median"), g("util_pct_median"), g("util_pct_peak"), g("busy_pct"), g("power_w_peak")
    head = f"| {size} | {arm} | {batch} | {nw} | {box} | {len(rates)} | {med:.4f} | {rates[0]:.4f}-{rates[-1]:.4f} "
    print(head + (f"| {statistics.median(mhz):.0f} | {statistics.median(um):.0f}/{max(up):.0f} % "
                  f"| {statistics.median(bp):.0f} | {max(pw):.0f} |" if mhz and um and up and bp and pw
                  else "| no smi window | | | |"))
    if len(rates) >= 5 and med > best.get(size, (0,))[0]:   # protocol: >= 5 warm batches to be the chosen arm
        best[size] = (med, arm, batch, len(rates), rates[0], rates[-1], nw, box)

print("\nchosen per size (>= 5 warm batches, one box per cell):")
for size, (med, arm, batch, n, lo, hi, nw, box) in sorted(best.items()):
    print(f"  {size}: {med:.4f} designs/s  arm={arm} batch={batch} num_workers={nw} box={box} n={n} spread {lo:.4f}-{hi:.4f}")

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

if a.append:
    ev = STATE / "evidence" / "gpu-b200.jsonl"
    mine = {r["session"] for r in lines}
    kept = [l for l in ev.read_text().splitlines(keepends=True) if json.loads(l).get("session") not in mine]
    ev.write_text("".join(kept) + "".join(json.dumps(r) + "\n" for r in lines))
    print(f"\nappended {len(lines)} lines for {sorted(mine)} to {ev} ({len(kept)} other lines kept)")
elif a.write:
    ev = STATE / "evidence" / "gpu-b200.jsonl"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text("".join(json.dumps(r) + "\n" for r in lines))
    row = json.loads((STATE / "rows" / "rfd4-fair-gpu.json").read_text())
    row["arms"] = sorted({r["arm"] for r in lines if r.get("arm")})
    row["designs_per_s"] = {str(s): round(v[0], 4) for s, v in best.items()}
    # Per size, because the arms do not agree across sizes and the fastest absolute rate is always the
    # smallest size's. `best_arm` is the one arm the session published and generated its reference designs
    # from: ARMS.json's cross-size choice.
    row["best_arm_per_size"] = {str(s): {"arm": v[1], "batch": v[2], "num_workers": v[6], "box": v[7],
                                         "n": v[3], "spread": [round(v[4], 4), round(v[5], 4)]}
                                for s, v in best.items()}
    arms = [json.loads((s / "ARMS.json").read_text()).get("best_arm")
            for s in a.session if (s / "ARMS.json").exists()]
    row["best_arm"] = arms[-1] if arms else None
    (STATE / "rows" / "rfd4-fair-gpu.json").write_text(json.dumps(row, indent=2) + "\n")
    print(f"\nwrote {ev} ({len(lines)} lines) and rows/rfd4-fair-gpu.json")
if dropped:
    print(f"\ndropped ({len(dropped)}):")
    for d in dropped[:20]:
        print(f"  {d}")
