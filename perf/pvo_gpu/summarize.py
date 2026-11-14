"""Per-cell table for a pvo_gpu session: s/fold, cold, peak memory, clock and power over each cell's own windows.

    python3 -I summarize.py <results/<session>/pvo> [--usd-per-s 0.000583]

A cell is one (model, arm dir, input). s/fold = median warm round wall / predictions in that round (K processes
share the GPU, so a K=2 round makes two folds). The clock is the median SM MHz of the 1 Hz nvidia-smi rows that fall
inside the cell's warm rounds, never a reading taken before or after them.
"""
import argparse, csv, json, statistics as st, sys
from datetime import datetime, timezone
from pathlib import Path


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def smi_rows(p):
    out = []
    for r in csv.reader(open(p)):
        if len(r) < 9:
            continue
        try:
            t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            out.append((t, float(r[1].split()[0]), float(r[3].split()[0]), r[8].strip()))
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--usd-per-s", type=float, default=0.000583, help="Modal A100-40 GPU $/s (PRICES.md)")
    a = ap.parse_args()
    root = Path(a.root)
    smi = smi_rows(root / "smi.csv")
    print("| model | arm | input | K | cold s | warm s/fold (n) | spread s | SM MHz | W | throttle | peak GiB | $ GPU/fold |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(root.glob("*/*/runs.jsonl")):
        cells = {}
        for line in open(f):
            r = json.loads(line)
            if r.get("input"):
                cells.setdefault(r["input"], []).append(r)
        for inp, rs in cells.items():
            warm = [r for r in rs if r["kind"] == "warm" and r["predictions"]]
            first = [r for r in rs if r["kind"] in ("cold", "first")]
            if not warm:
                print(f"| {f.parts[-3]} | {f.parts[-2]} | {inp} | {rs[0]['k']} | - | no warm round | | | | | | |")
                continue
            per = [r["wall_s"] / r["predictions"] for r in warm]
            win = [s for r in warm for s in smi if ts(r["start"]) <= s[0] <= ts(r["t"])]
            mhz = st.median(s[1] for s in win) if win else float("nan")
            w = st.median(s[2] for s in win) if win else float("nan")
            thr = ",".join(sorted({s[3] for s in win})) or "-"
            peak = max(r.get("peak_reserved_gib") or 0 for r in rs)
            cold = f"{first[0]['wall_s'] / first[0]['predictions']:.1f}" if first else "-"
            m = st.median(per)
            print(f"| {f.parts[-3]} | {f.parts[-2]} | {inp} | {rs[0]['k']} | {cold} | {m:.1f} ({len(per)}) | "
                  f"{max(per) - min(per):.1f} | {mhz:.0f} | {w:.0f} | {thr} | {peak:.2f} | {m * a.usd_per_s:.4f} |")


if __name__ == "__main__":
    sys.exit(main())
