"""Table of one kitcmp session: per model x mode, warm median s/fold, n, min-max, cold, rc, the kit's ACTIVE line,
the wrapped function, and the SM clock / power / memory nvidia-smi read during the arm.

    python3 perf/kitcmp/parse.py perf/kitcmp/results/<label>/results
"""
import csv, json, statistics, sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])


def ts(p):
    return datetime.strptime(p.read_text().strip(), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()


print("| model | mode | warm median s | n | min-max | cold s | rc | wrapped | SM MHz med (min) | W med | mem GB max | engaged |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in sorted(p for p in root.iterdir() if p.is_dir()):
    smi = []
    if (m / "smi.csv").exists():
        for r in csv.reader((m / "smi.csv").open()):
            try:
                t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
                smi.append((t, float(r[1].split()[0]), float(r[3].split()[0]), float(r[8].split()[0]) / 1024))
            except (ValueError, IndexError):
                pass
    for mode in ("off", "exact", "fast"):
        d = m / mode
        if not d.exists():
            continue
        rows = [json.loads(l) for l in (d / "times.jsonl").read_text().splitlines()] if (d / "times.jsonl").exists() else []
        s = [r["s"] for r in rows]
        warm = s[1:]
        rc = (d / "rc").read_text().strip() if (d / "rc").exists() else "running"
        err = (d / "stderr.log").read_text(errors="replace") if (d / "stderr.log").exists() else ""
        act = next((l.strip()[:90] for l in err.splitlines() if "ACTIVE" in l or l.startswith("stock:")), "-")
        clk = pw = mem = "-"
        if smi and (d / "t_end").exists():
            a, b = ts(d / "t_start"), ts(d / "t_end")
            w = [x for x in smi if a <= x[0] <= b and x[2] > 100]  # samples under load only
            if w:
                c = [x[1] for x in w]
                clk = f"{statistics.median(c):.0f} ({min(c):.0f})"
                pw = f"{statistics.median(x[2] for x in w):.0f}"
                mem = f"{max(x[3] for x in w):.1f}"
        med = f"{statistics.median(warm):.3f}" if warm else "-"
        rng = f"{min(warm):.3f}-{max(warm):.3f}" if warm else "-"
        fn = rows[0]["fn"].rsplit(".", 2)[-2:] if rows else ["-"]
        print(f"| {m.name} | {mode} | {med} | {len(warm)} | {rng} | {s[0] if s else '-'} | {rc} | {'.'.join(fn)} | {clk} | {pw} | {mem} | {act} |")
