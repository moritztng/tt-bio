#!/usr/bin/env python3
"""Localise card error in `ipa_fused.layer`: every op runs on the card and in float64 numpy on the
card's own inputs, so each line is that op's LOCAL error, not error carried in from upstream.

  TT_VISIBLE_DEVICES=<card> python perf/bc4_ipa/ipa_fused_opdiff.py --params <npz> [--n 224] [--top 25]
"""
import argparse, os, sys, traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipa_fused as F
import ipa_fused_tt as FT

LOG = []


class P:
    """A card tensor plus the float64 copy of it."""
    def __init__(self, tt, np_): self.tt, self.np = tt, np_
    @property
    def shape(self): return self.np.shape


def unwrap(v, side):
    if isinstance(v, P):
        return getattr(v, side)
    if isinstance(v, (list, tuple)):
        return type(v)(unwrap(x, side) for x in v)
    return v


class Diff:
    def __init__(self, dev):
        self.tt, self.np = FT.TT(dev), F.NP()

    def put(self, x):
        return P(self.tt.put(x), np.asarray(x, np.float64))

    def get(self, x):
        return self.tt.get(x.tt)

    def card_in(self, v):
        if isinstance(v, P):
            return self.tt.get(v.tt)
        if isinstance(v, (list, tuple)):
            return type(v)(self.card_in(x) for x in v)
        return v

    def __getattr__(self, name):
        tf, nf = getattr(self.tt, name), getattr(self.np, name)

        def run(*a, **k):
            site = traceback.extract_stack(limit=2)[0].lineno
            out_tt = tf(*unwrap(a, "tt"), **unwrap(k, "tt"))
            if name == "softmax" and os.environ.get("IPA_DUMP_LOGITS"):
                np.save(os.environ["IPA_DUMP_LOGITS"], self.card_in(a[0]))
            ref = np.asarray(nf(*[self.card_in(x) for x in a], **{kk: self.card_in(v) for kk, v in k.items()}))
            got = self.tt.get(out_tt)
            if ref.size == got.size:
                ref = ref.reshape(got.shape)
            nr = np.linalg.norm(ref)
            rel = float(np.linalg.norm(got - ref) / nr) if nr else float(np.linalg.norm(got))
            LOG.append((rel, name, site, tuple(got.shape), float(np.abs(ref).max()), float(np.abs(got - ref).max())))
            return P(out_tt, ref)
        return run


def main(a):
    from tt_bio import tenstorrent as T
    raw = np.load(a.params)
    p = {kk: np.asarray(raw[kk], np.float64) for kk in raw.files if "/structure_module/" in kk}
    k = F.pack(p)
    rng = np.random.default_rng(1)
    n, real = a.n, a.n - 4
    act = rng.standard_normal((n, F.CS)); z = rng.standard_normal((n, n, F.CZ))
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    quat = rng.standard_normal((n, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = F.G.quaternion_rotation(quat[:, 1] / quat[:, 0], quat[:, 2] / quat[:, 0], quat[:, 3] / quat[:, 0]).reshape(n, 9)
    t = rng.standard_normal((n, 3)) * 5.0
    bias = F.pair_bias(F.NP(), k, z, mask)
    dev = T.get_device()
    o = Diff(dev)
    kd = FT.upload(o, k)
    out = F.layer(o, kd, o.put(act), o.put(R), o.put(t), o.put(bias), o.put(z))
    ref = F.layer(F.NP(), k, act, R, t, bias, z)
    for nm, x, y in zip(("act", "R", "t"), ref, out):
        g = o.get(y)
        print(f"end-to-end {nm}: rel L2 {np.linalg.norm(g - x) / np.linalg.norm(x):.3e}")
    print(f"{len(LOG)} ops; worst local errors (rel L2, op, ipa_fused.py line, shape, max|ref|, max|err|):")
    for r in sorted(LOG, reverse=True)[:a.top]:
        print(f"  {r[0]:.3e}  {r[1]:<10} L{r[2]:<4} {str(r[3]):<16} {r[4]:.3e}  {r[5]:.3e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--top", type=int, default=25)
    main(ap.parse_args())
