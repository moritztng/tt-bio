#!/usr/bin/env python3
"""Frame-to-frame rotation of a recorded fold, under each way of aligning its frames for display.

    python3 demo/sc26/science/rotation.py demo/sc26/gallery/trajectories/top7.jsonl ...

For every pair of consecutive displayed frames it finds the rigid rotation that best superposes the
later onto the earlier (Kabsch) and reports its angle, two ways:

  points     the displayed xyz, literally what is drawn. While the cloud is still mostly noise this
             is the fit chasing fresh noise, not a pose (Boltz-2 at step 100 of 200 is still 38 A
             RMSD from its final structure), so read it where the structure has formed.
  structure  the displayed x0, the shape the points are condensing toward. This says whether the
             emerging protein holds still.

over two windows: the second half of the steps, and the formed phase (displayed xyz within 5 A
RMSD of the final structure). A still picture reads ~0 deg; a tumbling one reads tens of degrees.

Policies:
  raw        the sampler's coordinates as dumped (each step lives in a freshly rotated frame)
  sent       each frame moved by the R/T the stream carries with it
  xyz>final  each frame's xyz superposed onto the final structure (the renderer before 10-03)
  x0>final   the rigid transform that superposes the frame's own x0 onto the final, applied to xyz.
             x0 is the same step's denoised estimate, in the same frame as xyz, and already has the
             protein's shape, so the fit does not chase noise. What the renderer does now.
  x0>first   live, final unknown: x0 superposed onto the fold's first x0, one fixed reference.
             What the engine sends now.
"""
import base64
import json
import sys

import numpy as np


def dec(s):
    return np.frombuffer(base64.b64decode(s), "<f4").reshape(-1, 3).astype(np.float64)


def kabsch(mob, ref):
    mc, rc = mob.mean(0), ref.mean(0)
    u, _, vt = np.linalg.svd((mob - mc).T @ (ref - rc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, rc - r @ mc


def angle(r):
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def apply(x, rt):
    return x @ rt[0].T + rt[1]


def transforms(xyz, x0, frames):
    """Per policy, one rigid (R, t) per frame."""
    fin = xyz[-1]
    first = next(x for x in x0 if x is not None)
    ref0 = lambda i: x0[i] if x0[i] is not None else xyz[i]
    eye = (np.eye(3), np.zeros(3))
    return {
        "raw": [eye] * len(xyz),
        "sent": [(np.array(f["R"]).reshape(3, 3), np.array(f["T"])) for f in frames],
        "xyz>final": [kabsch(x, fin) for x in xyz[:-1]] + [eye],
        "x0>final": [kabsch(ref0(i), fin) for i in range(len(xyz) - 1)] + [eye],
        "x0>first": [kabsch(ref0(i), first) if x0[i] is not None else eye for i in range(len(xyz))],
    }


def measure(path):
    evs = [json.loads(l) for l in open(path)]
    frames = sorted((e for e in evs if e["type"] == "frame"), key=lambda e: e["step"])
    xyz = [dec(f["xyz"]) for f in frames]
    x0 = [dec(f["x0"]) if f.get("x0") else None for f in frames]
    fin, n = xyz[-1], len(frames)
    good = transforms(xyz, x0, frames)["x0>final"]
    rmsd = [float(np.sqrt(((apply(x, rt) - fin) ** 2).sum(1).mean())) for x, rt in zip(xyz, good)]
    windows = {"half": range(n // 2, n - 1), "formed": [i for i in range(n - 1) if rmsd[i] < 5.0]}
    out = {}
    for name, rts in transforms(xyz, x0, frames).items():
        pts = [angle(kabsch(apply(xyz[i + 1], rts[i + 1]), apply(xyz[i], rts[i]))[0]) for i in range(n - 1)]
        # the noise frame has no x0: it has no shape to hold still, so the structure series starts at step 0
        shp = [apply(x0[i], rts[i]) if x0[i] is not None else None for i in range(n - 1)] + [apply(fin, rts[-1])]
        st = [angle(kabsch(shp[i + 1], shp[i])[0]) if shp[i] is not None else None for i in range(n - 1)]
        row = {}
        for wname, w in windows.items():
            p = [pts[i] for i in w]
            s = [st[i] for i in w if st[i] is not None]
            row[wname] = dict(points=(float(np.median(p)), float(np.max(p))) if p else None,
                              structure=(float(np.median(s)), float(np.max(s))) if s else None)
        out[name] = row
    return n, len(windows["formed"]), out


def main():
    f = lambda v: "-" if v is None else f"{v[0]:6.3f} /{v[1]:7.2f}"
    for p in sys.argv[1:]:
        n, nf, res = measure(p)
        print(f"{p.rsplit('/', 1)[-1]}: {n} frames, {nf} formed. Frame-to-frame rotation, deg, median / max")
        print(f"  {'':10s} {'half points':>16s} {'half structure':>16s} {'formed points':>16s} {'formed struct':>16s}")
        for k, v in res.items():
            print(f"  {k:10s} {f(v['half']['points']):>16s} {f(v['half']['structure']):>16s} "
                  f"{f(v['formed']['points']):>16s} {f(v['formed']['structure']):>16s}")


if __name__ == "__main__":
    main()
