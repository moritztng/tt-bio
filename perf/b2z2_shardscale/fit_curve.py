"""The scaling curve, fitted from the committed JSON: where does the marginal chip stop paying?

Every number in `state/b2z2-trunk-shard-scale-wh.md` rebuilds from here, so the doc cannot drift
from the artifacts. Block walls come from `b2z2_scale_sweep_*.json`, the link from the `pair_shape`
arm of `b2z2_gathercurve_*.json` -- the rank-4 pair tensor the block actually gathers, not a
rank-3 tensor of the same byte count, which is 1.5x cheaper and would under-price the link.

The model:

    T(N) = A + B/N + 4 * G(N)

A is what every chip computes whether or not the block is sharded: the `b` role of both triangle
products reads the WHOLE pair tensor on every device, and the s track is replicated on purpose.
B/N is the row-local work. G(N) is measured at each width, not fitted, because the four gathers
are most of the reason a wide mesh stops paying.

    python3 perf/b2z2_shardscale/fit_curve.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
CELL_S = 20.113          # the published Boltz-2 512 aa Blackhole cell, CONTEXT 1-CORRECTION-D
PAIRFORMER_S = 10.22     # PairformerLayer device spans inside it, CONTEXT 2-CORRECTION


def load(pat):
    return {json.loads(p.read_text())["mesh_n"]: json.loads(p.read_text())
            for p in sorted(HERE.glob(pat))}


def lstsq(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return my - sxy_b(xs, ys, mx, my) * mx, sxy_b(xs, ys, mx, my)


def sxy_b(xs, ys, mx, my):
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def r2(xs, ys, a, b):
    my = sum(ys) / len(ys)
    return 1 - (sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
                / sum((y - my) ** 2 for y in ys))


def main():
    blocks, links = load("b2z2_scale_sweep_*.json"), load("b2z2_gathercurve_*.json")
    fits = {}
    for S in (512, 768):
        base = blocks[1]["by_size"][str(S)]["whole"]["median_ms"]
        print(f"\n=== WH, {S} aa, one PairformerLayer block. One chip: {base:.3f} ms ===")
        print(f"{'N':>2} {'rows/chip':>9} {'mesh tax':>9} {'shard ms':>9} {'vs 1 chip':>10} "
              f"{'marginal':>9} {'4 gathers':>10} {'compute':>9}")
        rows, prev = [], base
        for n in (2, 4, 8):
            a = blocks[n]["by_size"].get(str(S))
            if not a:
                continue
            g = next(p for p in links[n]["pair_shape"] if p["S"] == S)
            link = 4 * g["median_us"] / 1e3
            comp = a["shard"]["median_ms"] - link
            marg = prev / a["shard"]["median_ms"]
            print(f"{n:>2} {S // n:>9} {a['whole']['median_ms'] / base:>8.4f}x "
                  f"{a['shard']['median_ms']:>9.3f} {base / a['shard']['median_ms']:>9.4f}x "
                  f"{marg:>8.4f}x {link:>10.3f} {comp:>9.3f}")
            rows.append((n, comp, link))
            prev = a["shard"]["median_ms"]

        xs, ys = [1 / n for n, _, _ in rows], [c for _, c, _ in rows]
        A, B = lstsq(xs, ys)
        g_max = max(g for _, _, g in rows)
        fits[S] = (base, A, B, g_max)
        print(f"\ncompute(N) = {A:.3f} ms + {B:.3f} ms / N     R2 {r2(xs, ys, A, B):.4f}")
        print(f"the un-shardable floor A is {A / base * 100:.1f} % of the one-chip block")
        print(f"with a FREE link the block tops out at {base / A:.3f}x at any width")
        print(f"with the measured link it tops out at {base / (A + g_max):.3f}x "
              f"(4 gathers saturate near {g_max:.3f} ms)")
        prev = rows[-1][1] + rows[-1][2]
        for n in (16, 32, 64):
            t = A + B / n + g_max
            print(f"  projected N={n:<3d} {t:7.3f} ms  {base / t:.3f}x  "
                  f"marginal over N={n // 2:<3d} {prev / t:.3f}x")
            prev = t

    # --- what it is worth on the fold, as a projection with its assumption named ------------------
    base, A, B, g_max = fits[512]
    print(f"\n=== projected onto the Blackhole cell ({CELL_S} s, {PAIRFORMER_S} s of it "
          f"PairformerLayer) ===")
    print("WH block ratios on BH seconds: what transfers is the un-shardable FRACTION, which is a "
          "property of\nthe split and not of the silicon. Stated as a projection, not a measurement.")
    print(f"{'width':>7} {'block':>9} {'trunk s':>9} {'fold s':>9} {'fold x':>8}")
    for label, r in ([(str(n), base / blocks[n]["by_size"]["512"]["shard"]["median_ms"])
                      for n in (2, 4, 8)]
                     + [("inf", base / (A + g_max)), ("inf/free", base / A),
                        ("trunk=0", float("inf"))]):
        trunk = PAIRFORMER_S / r
        fold = CELL_S - PAIRFORMER_S + trunk
        print(f"{label:>7} {r:>8.3f}x {trunk:>9.3f} {fold:>9.3f} {CELL_S / fold:>7.4f}x")
    print(f"\n2x on this cell needs {CELL_S / 2:.3f} s. Deleting EVERY PairformerLayer second "
          f"leaves {CELL_S - PAIRFORMER_S:.3f} s.")


if __name__ == "__main__":
    main()
