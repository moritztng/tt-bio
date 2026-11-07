"""bat-fold: reduce perf/bat/fold_climb.sh's cells to s/fold, the trunk/diffusion split and the two-request price.

    python3 perf/bat/fold_table.py <run dir>

Per cell: median warm wall, AICLK median (min), foreign cores, pLDDT, samples. Per (model, input), with
fold(S, W) = trunk + S x d(W):
    d(1) = (fold(5,1) - fold(1,1)) / 4            per-sample diffusion, one sample a call
    trunk = fold(1,1) - d(1)                      everything that runs once a request
    d(5) = (fold(5,5) - trunk) / 5, d(10) likewise from fold(10,10)
    two requests on one chip, their 10 samples in one batch = fold(10,10) + trunk, against 2 x fold(5,5) serial.
"""
import json
import statistics
import sys
from pathlib import Path


def cell(d):
    warm = []
    for line in open(d / "runs.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "warm" and not r.get("discard") and all(not c.get("err") for c in r.get("per_chip", [])):
            warm.append(r)
    if not warm:
        return None
    w = statistics.median(r["wall_s"] for r in warm)
    pc = warm[0]["per_chip"][0]
    return dict(wall=w, n=len(warm), spread=(max(r["wall_s"] for r in warm) - min(r["wall_s"] for r in warm)) / w,
                clk=statistics.median(r["aiclk_mhz"] for r in warm), clkmin=min(r["aiclk_min_mhz"] for r in warm),
                fc=max(r.get("foreign_cores_median", 0) for r in warm), plddt=pc.get("plddt"), samples=pc.get("samples"))


def main():
    R = Path(sys.argv[1])
    cells = {}
    for d in sorted(p for p in R.iterdir() if p.is_dir()):
        m, c, s, w = d.name.split("-")
        got = cell(d)
        if got:
            cells[(m, c, int(s[1:]), int(w[1:]))] = got
    print("| model | input | S | W | s/fold | warm n | spread | AICLK med (min) | foreign cores | pLDDT | samples |")
    print("|---|---|--:|--:|--:|--:|--:|---|--:|--:|--:|")
    for (m, c, s, w), g in sorted(cells.items()):
        print(f"| {m} | {c} | {s} | {w} | {g['wall']:.2f} | {g['n']} | {100 * g['spread']:.1f} % | {g['clk']:.0f} ({g['clkmin']:.0f})"
              f" | {g['fc']:.2f} | {g['plddt']:.3f} | {g['samples']} |")
    print()
    for m, c in sorted({k[:2] for k in cells}):
        f = lambda s, w: cells.get((m, c, s, w), {}).get("wall")  # noqa: E731
        if not (f(1, 1) and f(5, 1) and f(5, 5)):
            continue
        d1 = (f(5, 1) - f(1, 1)) / 4
        trunk = f(1, 1) - d1
        d5 = (f(5, 5) - trunk) / 5
        out = f"{m} {c}: trunk {trunk:.1f} s, diffusion/sample W1 {d1:.2f} s, W5 {d5:.2f} s (W5 gain {d1 / d5:.2f}x)"
        if f(10, 10):
            d10 = (f(10, 10) - trunk) / 10
            two = f(10, 10) + trunk
            out += (f", W10 {d10:.2f} s ({d1 / d10:.2f}x); two requests sharing the chip {two:.1f} s vs serial"
                    f" {2 * f(5, 5):.1f} s = {2 * f(5, 5) / two:.3f}x")
        if f(10, 5):
            out += f"; S10 W5 {f(10, 5):.1f} s"
        out += f"; diffusion share of a served fold {5 * d5 / f(5, 5):.0%}"
        print(out)


if __name__ == "__main__":
    main()
