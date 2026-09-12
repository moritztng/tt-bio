"""The scaling curve, fitted from the committed JSON: where does the marginal chip stop paying?

Every number this prints comes out of `b2z2_scale_*.json` in this directory, so the table in
`state/b2z2-trunk-shard-scale-wh.md` rebuilds from artifacts rather than being retyped.

The model is Amdahl plus a link term that RISES with width:

    T(N) = A + B/N + 4 * G(N)

A is the part of the block that every chip computes whether or not it is sharded -- the `b` role of
both triangle products reads the whole pair tensor on every device, and the s track is replicated
on purpose. B/N is the row-local work. G(N) is one all_gather of the pair tensor at width N, and
it is MEASURED at each width rather than fitted, because the four gathers are 4/5 of the reason a
wide mesh stops paying.

    python3 perf/b2z2_shardscale/fit_curve.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
S = 512


def load():
    out = {}
    for p in sorted(HERE.glob("b2z2_scale_*.json")):
        d = json.loads(p.read_text())
        out[d["mesh_n"]] = d
    return out


def arms_at(d, s=S):
    if "by_size" in d:
        return d["by_size"].get(str(s))
    return d.get("arms") if d.get("S") == s else None


def gather_at(d, out_bytes):
    """The measured cost of the one gather the block actually pays, at this width."""
    if not d.get("gather"):
        return None
    return min(d["gather"], key=lambda g: abs(g["out_bytes"] - out_bytes))


def lstsq(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def r2(xs, ys, a, b):
    my = sum(ys) / len(ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return 1 - ss_res / ss_tot


def main():
    data = load()
    base = arms_at(data[1])["whole"]["median_ms"]
    full = S * S * 128 * 2

    print(f"WH, {S} aa, one PairformerLayer block. Single chip: {base:.3f} ms\n")
    print(f"{'N':>2} {'whole/mesh':>11} {'shard':>9} {'vs 1 chip':>10} {'marginal':>9} "
          f"{'1 gather':>9} {'4 gathers':>10} {'compute':>9}")
    rows = []
    prev = None
    for n in sorted(data):
        a = arms_at(data[n])
        if not a:
            continue
        whole = a["whole"]["median_ms"]
        if "shard" not in a:
            print(f"{n:>2} {whole:>11.3f} {'-':>9} {base / whole:>9.4f}x {'-':>9} "
                  f"{'-':>9} {'-':>10} {'-':>9}")
            continue
        shard = a["shard"]["median_ms"]
        g = gather_at(data[n], full)
        g_ms = g["median_us"] / 1e3 if g else float("nan")
        comp = shard - 4 * g_ms
        marg = prev / shard if prev else float("nan")
        print(f"{n:>2} {whole:>11.3f} {shard:>9.3f} {base / shard:>9.4f}x {marg:>8.4f}x "
              f"{g_ms:>9.3f} {4 * g_ms:>10.3f} {comp:>9.3f}")
        rows.append((n, shard, comp, 4 * g_ms))
        prev = shard

    xs = [1 / n for n, _, _, _ in rows]
    ys = [c for _, _, c, _ in rows]
    A, B = lstsq(xs, ys)
    print(f"\ncompute(N) = {A:.3f} ms + {B:.3f} ms / N      R2 {r2(xs, ys, A, B):.4f}")
    print(f"the un-shardable floor is {A / base * 100:.1f} % of the single-chip block")
    g_inf = max(4 * g for _, _, _, g in rows)
    print(f"link saturates near {g_inf:.3f} ms per block (4 gathers, widest width measured)")
    print(f"T(inf) -> {A + g_inf:.3f} ms, so the shard's ceiling at any width is "
          f"{base / (A + g_inf):.3f}x on this part\n")
    for n in (8, 16, 32):
        t = A + B / n + g_inf
        print(f"  projected N={n:<3d} {t:7.3f} ms  {base / t:.3f}x")


if __name__ == "__main__":
    main()
