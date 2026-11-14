#!/usr/bin/env python3
"""Protein line art for tt-bio.com: a structure's C-alpha trace as fine vector lines, in the deck's style.

    python3 scripts/site_art.py <structure.cif> <out.svg> [--back A] [--spin RAD] [--size W H] [--no-pca] [--on wash]

The two drawings on tt-bio.com, from the deck's structures (PDB 1BRS chains A+D; a BindCraft 2 binder on
PD-L1 designed on Tenstorrent):

    site_art.py 1brs_AD.cif site/assets/art/barnase-barstar.svg --back A --spin -1.5708 --size 900 1000
    site_art.py pdl1_binder_l73.cif site/assets/art/pdl1-binder.svg --back A --size 900 900 --no-pca --on wash

A smoothed spline per chain, drawn back to front one residue at a time. Each piece carries a halo in the
page's own ground colour, so crossings read as over and under, and its tint fades toward the ground with
depth. Residues within 7 A of another chain are drawn in the accent. `--back` names a partner chain drawn in
grey. Colours are CSS classes with a light and a dark palette (prefers-color-scheme), so one file serves both
themes when it is used as an <img>. Keep PALETTE in step with site/assets/site.css.
"""
import argparse
import math

import numpy as np

# (ground, near ink, far ink, accent, partner near, partner far) per theme
PALETTE = {
    "light": ("#faf8f4", "#2a2824", "#cfc8bc", "#1f9ccc", "#a59e92", "#e3ddd2"),
    "dark": ("#101318", "#ece8df", "#3b3f46", "#3cb8e0", "#6c6d70", "#262a30"),
}
# --on wash: the halo matches the tinted panels (--wash in site.css) instead of the page
WASH = {"light": "#f2eee6", "dark": "#1a1e25"}
DEPTHS, ACCENTS = 6, 3


def ca(path):
    cols, ch = [], {}
    for ln in open(path):
        if ln.startswith("_atom_site."):
            cols.append(ln.split(".", 1)[1].strip())
        elif ln.startswith("ATOM"):
            p = dict(zip(cols, ln.split()))
            if p["label_atom_id"] == "CA":
                ch.setdefault(p["label_asym_id"], []).append([float(p[f"Cartn_{a}"]) for a in "xyz"])
    return {k: np.array(v) for k, v in ch.items()}


def spline(p, n):
    """Uniform Catmull-Rom through the points, n samples per segment."""
    q = np.vstack([2 * p[:1] - p[1:2], p, 2 * p[-1:] - p[-2:-1]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    out = []
    for i in range(1, len(q) - 2):
        p0, p1, p2, p3 = q[i - 1], q[i], q[i + 1], q[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-1:])
    return np.vstack(out)


def rgb(h):
    return np.array([int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], float)


def hexc(c):
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def art(path, w, h, back=None, spin=0.0, pca=True, ground=None, lw=2.3, halo=3.2, n=6, contact=7.0, fill=0.96):
    C = ca(path)
    allp = np.vstack(list(C.values()))
    mu = allp.mean(0)
    R = np.linalg.svd(allp - mu, full_matrices=False)[2] if pca else np.eye(3)
    cs, sn = math.cos(spin), math.sin(spin)
    R = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]]) @ R
    P = {k: (v - mu) @ R.T for k, v in C.items()}
    acc = {}
    for k, v in P.items():
        o = np.vstack([P[j] for j in P if j != k])
        d = np.linalg.norm(v[:, None] - o[None], axis=2).min(1)
        acc[k] = np.clip((contact + 1 - d) / 2, 0, 1) if k != back else np.zeros(len(v))
    sp = {k: spline(v, n) for k, v in P.items()}
    allp = np.vstack(list(sp.values()))
    lo, hi = allp[:, :2].min(0), allp[:, :2].max(0)
    sc = min(w * fill / (hi[0] - lo[0]), h * fill / (hi[1] - lo[1]))
    mid = (lo + hi) / 2
    zlo, zhi = allp[:, 2].min(), allp[:, 2].max()
    pieces = []
    for k, p in sp.items():
        a = np.repeat(acc[k], n)[:len(p)]
        X = (p[:, 0] - mid[0]) * sc + w / 2
        Y = -(p[:, 1] - mid[1]) * sc + h / 2
        for s0 in range(0, len(p) - 1, n):
            s1 = min(len(p), s0 + n + 1)
            zm = p[s0:s1, 2].mean()
            t = float(np.clip((zm - zlo) / (zhi - zlo), 0, 1)) ** 0.9
            pieces.append((zm, X[s0:s1], Y[s0:s1], min(DEPTHS - 1, int(t * DEPTHS)),
                           int(round(a[s0:s1].max() * (ACCENTS - 1))), "b" if k == back else "m"))
    pieces.sort(key=lambda q: q[0])

    def d(X, Y):
        return "M" + "L".join(f"{x:.0f} {y:.0f}" if w >= 600 else f"{x:.1f} {y:.1f}" for x, y in zip(X, Y))

    body = []
    for _, X, Y, di, ai, grp in pieces:
        m = max(1, len(X) // 4)
        if len(X) > 2 * m + 1:
            body.append(f'<path class="h{di}" d="{d(X[m:-m], Y[m:-m])}"/>')
        body.append(f'<path class="{grp}{di}{ai}" d="{d(X, Y)}"/>')

    def rules(theme):
        g, near, far, ac, bn, bf = (rgb(c) for c in PALETTE[theme])
        if ground:
            g = rgb(WASH[theme])
        out = [f".h{i}{{stroke:{hexc(g)};stroke-width:{lw * (0.8 + 0.4 * i / (DEPTHS - 1)) * halo:.2f}}}"
               for i in range(DEPTHS)]
        for i in range(DEPTHS):
            t = i / (DEPTHS - 1)
            sw = f"stroke-width:{lw * (0.8 + 0.4 * t):.2f}"
            for j in range(ACCENTS):
                aj = j / (ACCENTS - 1)
                col = (far * (1 - t) + near * t) * (1 - aj) + ac * aj
                out.append(f".m{i}{j}{{stroke:{hexc(col)};{sw}}}")
            out.append(f".b{i}0{{stroke:{hexc(bf * (1 - t) + bn * t)};{sw}}}")
        return "".join(out)

    style = (f"<style>{rules('light')}@media (prefers-color-scheme:dark){{{rules('dark')}}}</style>")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{style}'
            f'<g fill="none" stroke-linecap="round" stroke-linejoin="round">{"".join(body)}</g></svg>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif"); ap.add_argument("out")
    ap.add_argument("--back"); ap.add_argument("--spin", type=float, default=0.0)
    ap.add_argument("--size", type=int, nargs=2, default=(900, 900))
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--on", choices=["page", "wash"], default="page")
    a = ap.parse_args()
    svg = art(a.cif, *a.size, back=a.back, spin=a.spin, pca=not a.no_pca, ground=a.on == "wash")
    open(a.out, "w").write(svg)
    print(a.out, len(svg), "bytes")


if __name__ == "__main__":
    main()
