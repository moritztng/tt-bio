#!/usr/bin/env python3
"""of3t-denoise: where does the denoise arm's run-to-run variation enter?

    probe_aa.py --batch B --out F.pt        (one process: dump)
    probe_aa.py --compare A.pt B.pt [...]   (host: first input that differs)

Runs the adapter's training forward with the denoise arm on and dumps, to host, every tensor
argument of the taped `OF3DiffusionModule` call, its output, and the same module re-run untaped
twice on the same device inputs. Two dumps from two processes say whether the module's inputs
already differ (the variation is upstream of the module) or only its output does.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def dump(bpath, out):
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.train.openfold3 import OpenFold3Dataset, OpenFold3Forward
    import tt_bio.openfold3_diffusion_module as dmm

    ds = OpenFold3Dataset(bpath)
    fwd = OpenFold3Forward(Path.home() / "of3-weights/of3-p2-155k.pt", rollout=20,
                           num_cycles=1, seed=20260922)
    batch = ds.batch([0])
    cls = dmm.OF3DiffusionModule
    orig = cls.__call__
    cap = {}

    def spy(self, *args, **kw):
        y = orig(self, *args, **kw)
        if any(isinstance(a, ag.Tensor) for a in args):
            cap["args"], cap["kw"], cap["self"], cap["y"] = args, kw, self, y
        return y

    cls.__call__ = spy
    outputs = fwd(batch)
    cls.__call__ = orig
    raw = lambda t: t.value if isinstance(t, ag.Tensor) else t  # noqa: E731
    host = lambda t: ttnn.to_torch(raw(t))  # noqa: E731
    args = [raw(a) for a in cap["args"]]
    y1 = orig(cap["self"], *args, **dict(cap["kw"], cache={}))
    y2 = orig(cap["self"], *args, **dict(cap["kw"], cache={}))
    rec = {"args": {i: host(a) for i, a in enumerate(args) if isinstance(a, ttnn.Tensor)},
           "kw": {k: host(v) for k, v in cap["kw"].items() if isinstance(raw(v), ttnn.Tensor)},
           "y_taped": host(cap["y"]), "y_untaped": host(y1), "y_untaped2": host(y2),
           "outputs": {k: host(v) for k, v in outputs.items() if isinstance(raw(v), ttnn.Tensor)}}
    torch.save(rec, out)
    for k in ("y_taped", "y_untaped", "y_untaped2"):
        print(k, sha(rec[k]), flush=True)


def compare(paths):
    recs = [torch.load(p) for p in paths]
    a = recs[0]
    for p, b in zip(paths[1:], recs[1:]):
        print(f"== {paths[0]} vs {p}")
        for grp in ("args", "kw"):
            for k, t in a[grp].items():
                u = b[grp][k]
                d = (t.double() - u.double()).abs()
                print(f"{grp}[{k}] {tuple(t.shape)} {t.dtype} equal={torch.equal(t, u)} "
                      f"max|d|={float(d.max()):.3e} n_diff={int((d > 0).sum())}")
        for k in ("y_taped", "y_untaped", "y_untaped2"):
            d = (a[k].double() - b[k].double()).abs()
            print(f"{k} equal={torch.equal(a[k], b[k])} max|d|={float(d.max()):.3e}")
        for k, t in a["outputs"].items():
            print(f"outputs[{k}] equal={torch.equal(t, b['outputs'][k])}")
    for p, r in zip(paths, recs):
        print(p, "untaped A/A in process:", torch.equal(r["y_untaped"], r["y_untaped2"]),
              "taped==untaped:", torch.equal(r["y_taped"], r["y_untaped"]))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[0] == "--compare":
        compare(argv[1:])
    else:
        dump(Path(argv[argv.index("--batch") + 1]), Path(argv[argv.index("--out") + 1]))
