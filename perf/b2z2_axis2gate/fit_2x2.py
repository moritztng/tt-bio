"""The re-fit: `compute(N) = A + B/N` for all four arms of the window x `b_shard` square.

`b2z2-trunk-shard-scale-wh` fit `28.679 ms + 58.916 ms/N` at R2 0.9964 with the fused move silently
declined at N=4 and N=8 and served at N=2. A defect that only bites from four chips up bends the
curve in exactly the direction that reads as irreducible replicated work, so the constant it
produced -- the campaign's 33.4 % floor and its 2.299x block cap -- was measured through the bend.

Two levers attack the same constant, so this fits both arms of both, which is the only way two rows
can move one fit without double counting it: `b2z2-bshard-timing` owns `b_shard`'s own re-fit, this
row owns the window's, and the cross term is measured here rather than multiplied.

Model and method are the parent's, unchanged: block wall minus four measured pair-shape gathers,
least squares in 1/N over N = 2, 4, 8. The gathers are re-measured on THESE cards, value-checked
per device, and are identical across the four arms -- the window changes what a device computes,
not what it sends.

    python3 perf/b2z2_axis2gate/fit_2x2.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
S = "512"
PARENT_A, PARENT_B, PARENT_CAP = 28.679, 58.916, 2.299
CELL_S = 20.113          # the published Boltz-2 512 aa Blackhole cell, CONTEXT 1-CORRECTION-D
PAIRFORMER_S = 10.22     # PairformerLayer device spans inside it, CONTEXT 2-CORRECTION
ARMS = ["shard_narrow", "shard_long", "bshard_narrow", "bshard_long"]
WIDTHS = (2, 4, 8)


def lstsq(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    r2 = 1 - (sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
              / sum((y - my) ** 2 for y in ys))
    return a, b, r2


def main():
    blocks = {n: json.loads((OUT / f"scale_gate_{S}_n{n}.json").read_text()) for n in WIDTHS}
    links = {n: json.loads((OUT / f"gathercurve_{n}.json").read_text()) for n in WIDTHS}
    link = {n: 4 * next(p for p in links[n]["pair_shape"] if p["S"] == int(S))["median_us"] / 1e3
            for n in WIDTHS}
    base = sum(blocks[n]["by_size"][S]["whole"]["median_ms"] for n in WIDTHS) / len(WIDTHS)

    print(f"one chip (the `whole` arm, mean of the three meshes): {base:.3f} ms")
    print(f"link, 4 value-checked pair-shape gathers: "
          + "  ".join(f"N={n} {link[n]:.3f} ms" for n in WIDTHS))
    print(f"\n{'arm':>14} " + " ".join(f"{'N=' + str(n):>17}" for n in WIDTHS)
          + f" {'A ms':>8} {'B ms':>8} {'R2':>7} {'floor':>7} {'cap':>7}")

    fits = {}
    g_max = max(link.values())
    for arm in ARMS:
        ys, cells = [], []
        for n in WIDTHS:
            wall = blocks[n]["by_size"][S][arm]["median_ms"]
            moves = blocks[n]["by_size"][S][arm]["gated_moves_per_block"]
            ys.append(wall - link[n])
            cells.append(f"{wall:8.3f} ({base / wall:.4f}x,{moves})")
        A, B, r2 = lstsq([1 / n for n in WIDTHS], ys)
        fits[arm] = (A, B, r2)
        print(f"{arm:>14} " + " ".join(f"{c:>17}" for c in cells)
              + f" {A:8.3f} {B:8.3f} {r2:7.4f} {A / base * 100:6.1f}% "
                f"{base / (A + g_max):6.3f}x")

    print("\nper-cell: block ms (ratio vs one chip, fused moves served per block). "
          "2 of 4 is the ending trimul declined.")

    a_n, a_l = fits["shard_narrow"][0], fits["shard_long"][0]
    b_n, b_l = fits["bshard_narrow"][0], fits["bshard_long"][0]
    print(f"\nthe window is worth {a_n - a_l:.3f} ms of the constant without `b_shard` and "
          f"{b_n - b_l:.3f} ms with it")
    print(f"`b_shard` is worth {a_n - b_n:.3f} ms with the old window and {a_l - b_l:.3f} ms "
          f"with the fixed one")
    print(f"the two together: {a_n - b_l:.3f} ms, against {(a_n - a_l) + (a_n - b_n):.3f} ms if "
          f"they were added -- overlap {(a_n - a_l) + (a_n - b_n) - (a_n - b_l):.3f} ms")

    print(f"\nthe parent fit was {PARENT_A:.3f} ms + {PARENT_B:.3f} ms/N, cap {PARENT_CAP:.3f}x. "
          f"This row's unfixed arm reproduces it at {a_n:.3f} ms "
          f"({abs(a_n - PARENT_A) / PARENT_A * 100:.1f} % apart), and the fixed window takes the "
          f"constant to {a_l:.3f} ms and the cap to {base / (a_l + g_max):.3f}x.")

    print(f"\n=== projected onto the Blackhole cell ({CELL_S} s, {PAIRFORMER_S} s of it "
          f"PairformerLayer). A projection, not a measurement. ===")
    print(f"{'arm':>14} {'N=8':>9} {'N->inf':>9} {'fold at N=8':>13} {'fold at N->inf':>15}")
    for arm in ARMS:
        A, B, _ = fits[arm]
        r8 = base / blocks[8]["by_size"][S][arm]["median_ms"]
        rinf = base / (A + g_max)
        f8 = CELL_S - PAIRFORMER_S + PAIRFORMER_S / r8
        finf = CELL_S - PAIRFORMER_S + PAIRFORMER_S / rinf
        print(f"{arm:>14} {r8:8.3f}x {rinf:8.3f}x {CELL_S / f8:12.4f}x {CELL_S / finf:14.4f}x")


if __name__ == "__main__":
    main()
