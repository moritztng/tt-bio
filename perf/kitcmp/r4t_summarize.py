"""Summarize an R4T GPU training baseline session: one line per arm.

    python3 r4t_summarize.py <results/<label>/rfd4>        (the directory rfd4_session.sh pulls)

Per arm: steps recorded vs asked, mean and median full step over steps 6.. (warm-up excluded), the device step,
examples/s, recycle depths seen, peak allocated memory, every loss finite or not, and the SM clock (median) and
power over the arm's own window from the box's 1 Hz smi.csv. An arm with any non-finite loss, or fewer steps than
asked, is printed but never ranked.
"""

import csv
import datetime as dt
import json
import math
import statistics as st
import sys
from pathlib import Path

WARM = 5


def smi_window(rows, t0, t1):
    w = [r for r in rows if t0 <= r[0] <= t1]
    if not w:
        return None
    clk = st.median(r[1] for r in w)
    return {"sm_mhz_median": clk, "sm_mhz_min": min(r[1] for r in w), "power_w_max": max(r[2] for r in w), "n": len(w)}


def load_smi(path):
    rows = []
    if not path.exists():
        return rows
    for r in csv.reader(open(path)):
        try:
            t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
            rows.append((t, float(r[1].split()[0]), float(r[3].split()[0])))
        except (ValueError, IndexError):
            continue
    return rows


def main(root):
    root = Path(root)
    smi = load_smi(root / "smi.csv")
    out = []
    for d in sorted((root / "r4t").glob("*/")):
        p = d / "steps.jsonl"
        s = [json.loads(line) for line in open(p)] if p.exists() else []
        rc = (d / "rc").read_text().strip() if (d / "rc").exists() else "?"
        row = {"arm": d.name, "rc": rc, "steps": len(s)}
        if len(s) > WARM:
            t = s[WARM:]
            full = [x["full_s"] for x in t]
            row.update(
                full_mean_s=sum(full) / len(full), full_median_s=st.median(full),
                device_mean_s=sum(x["device_s"] for x in t) / len(t),
                examples_per_step=t[0]["examples"], physical_bs=t[0]["physical_bs"], tokens=t[0]["tokens"],
                recycles=[x["n_recycle"] for x in t],
                finite=all(x["loss"] is not None and math.isfinite(x["loss"]) for x in s),
                peak_alloc_gib=max((x["peak_alloc_gib"] or 0) for x in s),
                n_timed=len(t),
            )
            row["examples_per_s"] = row["examples_per_step"] / row["full_mean_s"]
            t0 = t[0]["t_end_unix"] - t[0]["full_s"]
            row["clock"] = smi_window(smi, t0, t[-1]["t_end_unix"])
        out.append(row)
    for r in out:
        print(json.dumps(r))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
