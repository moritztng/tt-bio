"""PVO: seconds per fold -> $ per fold and chips per workload, from the fair harness's runs.jsonl.

    python cost.py <cell dir or runs.jsonl> [...] [--rate 0.2568] [--label wh|bh]

A cell is one (model, mode, complex) on one side. Only clean warm rounds count: a line with `discard` is reported
and never averaged (a contended or under-clocked round is discarded, not averaged -- state/mol1/PROTOCOL.md s10).
Prints per cell: tokens, n, median s/prediction, AICLK median and min, then $/fold at --rate and the chips needed
for PopVax's 375k folds a month (730 h) and for a 20k day (24 h), with the Galaxy count at 32 chips.
"""
import argparse
import json
import statistics as st
from pathlib import Path

MONTH_FOLDS, MONTH_H, DAY_FOLDS, DAY_H, PER_GALAXY = 375_000, 730, 20_000, 24, 32


def cells(paths):
    for p in paths:
        p = Path(p)
        f = p / "runs.jsonl" if p.is_dir() else p
        if not f.exists():
            continue
        lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        warm = [r for r in lines if r.get("kind") == "warm" and r.get("phase") == "measure"]
        clean = [r for r in warm if not r.get("discard")]
        if not clean:
            yield f, None, lines, warm, []
            continue
        yield f, clean[0], lines, warm, clean


def main():
    a = argparse.ArgumentParser()
    a.add_argument("paths", nargs="+")
    a.add_argument("--rate", type=float, default=0.2568, help="$ per chip-hour (state/pvo/PRICES.md)")
    a.add_argument("--json", action="store_true")
    a = a.parse_args()
    rows = []
    for f, first, lines, warm, clean in cells(a.paths):
        if first is None:
            print(f"{f.parent.name}: no clean warm round ({len(warm)} warm, "
                  f"{sorted({r.get('discard', '')[:40] for r in warm if r.get('discard')})})")
            continue
        s = [r["wall_s"] / r["predictions"] for r in clean]
        clk = [r["aiclk_mhz"] for r in clean if r.get("aiclk_mhz")]
        med = st.median(s)
        row = dict(cell=f.parent.name, model=first["model"], arm=first["arm"], part=first["part"],
                   tokens=first.get("tokens"), chips=first["chips"], n=len(clean), discarded=len(warm) - len(clean),
                   s_per_pred=round(med, 2), s_min=round(min(s), 2), s_max=round(max(s), 2),
                   aiclk_med=st.median(clk) if clk else None,
                   aiclk_min=min((r["aiclk_min_mhz"] for r in clean if r.get("aiclk_min_mhz")), default=None),
                   sha=first["sha"][:9], fw=first.get("firmware"), host=first["host"],
                   usd_per_fold=round(med / 3600 * a.rate, 5),
                   chips_375k=round(MONTH_FOLDS * med / (MONTH_H * 3600), 1),
                   chips_20k_day=round(DAY_FOLDS * med / (DAY_H * 3600), 1))
        row["galaxies_375k"] = round(row["chips_375k"] / PER_GALAXY, 2)
        row["galaxies_20k_day"] = round(row["chips_20k_day"] / PER_GALAXY, 2)
        rows.append(row)
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    hdr = ("cell", "tokens", "n", "s_per_pred", "aiclk_med", "aiclk_min", "usd_per_fold",
           "chips_375k", "galaxies_375k", "chips_20k_day", "sha", "fw")
    print(f"$ per chip-hour {a.rate}; 375k folds / {MONTH_H} h and {DAY_FOLDS} folds / {DAY_H} h; {PER_GALAXY} chips per Galaxy")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["part"], r["model"], r["arm"], r["tokens"] or 0)):
        print("| " + " | ".join(str(r.get(h)) for h in hdr) + " |")


if __name__ == "__main__":
    main()
