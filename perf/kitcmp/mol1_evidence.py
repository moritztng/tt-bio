"""Collate pulled MOL1 B200 sessions into state/mol1/evidence/gpu-b200.jsonl and print the per-cell table.

    python mol1_evidence.py --session results/mol1-b200-1/mol1 [--session ...] [--write]

Reads the lines the box wrote and the box's 1 Hz smi.csv; measures nothing. Clock and load on each line come from
the window the batch occupied (`t - wall_s` .. `t`), never from a boundary sample, which reads an idle card
(state/rfd4/GPU-AUDIT.md). A speed cell is (size, arm, batch, session): one box, never pooled across rentals.
"""
import argparse, calendar, json, statistics, time
from pathlib import Path

STATE = Path("/home/moritz/.coworker/state/mol1")


def load_smi(path):
    out = []
    for raw in path.read_text().splitlines() if path.exists() else []:
        p = [c.strip() for c in raw.split(",")]
        try:
            ts = calendar.timegm(time.strptime(p[0].split(".")[0], "%Y/%m/%d %H:%M:%S"))
            out.append((ts, float(p[1].split()[0]), float(p[3].split()[0]), float(p[6].split()[0])))
        except (ValueError, IndexError):
            continue
    return sorted(out)


def window(smi, end_iso, wall_s):
    try:
        t1 = calendar.timegm(time.strptime(end_iso, "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError):
        return {}
    s = [x for x in smi if t1 - (wall_s or 0) <= x[0] <= t1]
    if not s:
        return {}
    busy = [x for x in s if x[3] > 50]
    return {"clock_mhz": statistics.median([x[1] for x in busy or s]), "power_w_peak": max(x[2] for x in s),
            "util_pct_median": statistics.median([x[3] for x in s]), "busy_pct": round(100 * len(busy) / len(s)),
            "smi_samples": len(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=Path, action="append")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    lines = []
    for sess in a.session:
        smi = load_smi(sess / "smi.csv")
        gpu = next((l.split(":", 1)[1].strip() for l in (sess / "nvidia-smi-q.txt").read_text().splitlines()
                    if "Product Name" in l), "NVIDIA B200") if (sess / "nvidia-smi-q.txt").exists() else "NVIDIA B200"
        for f in sorted(sess.rglob("runs.jsonl")):
            for raw in f.read_text().splitlines():
                if raw.strip():
                    r = json.loads(raw)
                    r["session"] = r.get("session") or sess.parent.name
                    r["gpu"] = gpu
                    r["raw"] = f"artifacts/mol1-fair-gpu/{sess.parent.name}/mol1/{f.parent.relative_to(sess)}"
                    r.update(window(smi, r.get("t"), r.get("wall_s")))
                    lines.append(r)
    cells = {}
    for r in lines:
        if r["kind"] == "warm" and r.get("status") != "FAILED" and r.get("predictions") and r["wall_s"] > 0:
            cells.setdefault((r["size"], r["arm"], r["batch"], r["session"]), []).append(r)
    print("| size | arm | K | box | n | predictions/s (median) | min-max | s/prediction | clock MHz | util med | peak W |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (size, arm, k, sess), rs in sorted(cells.items()):
        v = [r["predictions"] / r["wall_s"] for r in rs]
        m = statistics.median(v)
        clk = statistics.median([r["clock_mhz"] for r in rs if "clock_mhz" in r] or [0])
        util = statistics.median([r["util_pct_median"] for r in rs if "util_pct_median" in r] or [0])
        pw = max([r["power_w_peak"] for r in rs if "power_w_peak" in r] or [0])
        print(f"| {size} | {arm} | {k} | {sess} | {len(v)} | {m:.5f} | {min(v):.5f}-{max(v):.5f} | {1/m:.1f} | {clk:.0f} | {util:.0f} | {pw:.0f} |")
    failed = [r for r in lines if r.get("status") == "FAILED"]
    for r in failed:
        print(f"FAILED {r['size']} {r['arm']} K={r['batch']} {r['kind']}: {r.get('error', '')[-200:]!r}")
    if a.write:
        (STATE / "evidence").mkdir(exist_ok=True)
        (STATE / "evidence/gpu-b200.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
        print(f"wrote {len(lines)} lines")


if __name__ == "__main__":
    main()
