"""Where SaProt's attention loses accuracy on a real sequence, per layer, against float64 on the device's own inputs.

For every block, one forward of the page batch (the first --length residues of --pdb, x7 so the page programs run),
captured on sequence 0: the residual x going in, the q, k, v shsdpa receives and its output o. Graded in float64:
  qk:   q, k against LN + qkv + bias + RoPE in float64 from the same x (rel_l2, signed bias as mm_rope_check)
  sdpa: o against float64 softmax(scale q k^T) v on the device's own q, k, v (rel_l2), and the logit scale s that fits
        o best (o(s * scale)); s < 1 means the kernel's attention is softer than its inputs say, a deflated logit.
Arms flip module switches on one loaded model (--arm NAME=mod.ATTR[,..], all True; 'off' first).

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/attn_drift.py --model saprot-650m --pdb perf/bgx_size/targets/hPCSK9.pdb \
        --arm rope=tt_bio.saprot._SAPROT_MM_ROPE,tt_bio.qk_rope_heads.SFPU --out drift.jsonl
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mgx_embed"))
import accuracy as acc  # noqa: E402

SCALES = (0.990, 0.994, 0.996, 0.998, 1.0, 1.002, 1.004, 1.006, 1.010)


def attn64(q, k, v, n, sc):
    """float64 attention over the first n keys, [H, L, D]."""
    s = (q[:, :n] @ k[:, :n].transpose(-1, -2)) * sc
    return torch.softmax(s, -1) @ v[:, :n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--layers", default=None, help="comma list; default every layer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tt_bio import saprot as S

    aa = acc.pdb_sequence(a.pdb)[: a.length]
    n = len(aa) + 2   # cls + eos
    model = S.load_saprot(a.model, fast=a.fast)
    attns, seen = [], set()

    def find(o, depth=6):   # every ESM2Attention under the loaded model, in layer order
        if id(o) in seen or depth == 0:
            return
        seen.add(id(o))
        if isinstance(o, S.ESM2Attention):
            attns.append(o)
            return
        for v in (vars(o).values() if hasattr(o, "__dict__") and not isinstance(o, type) else ()):
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                if hasattr(x, "__dict__") and not isinstance(x, type):
                    find(x, depth - 1)
    find(model)
    assert attns, "no ESM2Attention found"
    index = {id(t): i for i, t in enumerate(attns)}
    want = set(int(i) for i in a.layers.split(",")) if a.layers else set(range(len(attns)))

    cap = {}
    cur = [None]
    call0, sdpa0 = S.ESM2Attention.__call__, S.lm_sdpa

    def call(self, x, *args, **kw):
        cur[0] = index.get(id(self))
        if cur[0] in want:
            cap[cur[0]] = {"x": ttnn.to_torch(x)[0].double()}
        return call0(self, x, *args, **kw)

    def sdpa(q, k, v, *args, **kw):
        o = sdpa0(q, k, v, *args, **kw)
        if cur[0] in want:
            cap[cur[0]].update({n_: ttnn.to_torch(t)[0].double() for n_, t in (("q", q), ("k", k), ("v", v), ("o", o))})
        return o
    S.ESM2Attention.__call__, S.lm_sdpa = call, sdpa

    arms = [("off", [])] + [(nm, [(importlib.import_module(m), t) for m, t in (x.rsplit(".", 1) for x in v.split(","))])
                            for nm, v in (s.split("=", 1) for s in a.arm)]
    sw = {(m, t): getattr(m, t) for _, ms in arms for m, t in ms}
    out = open(a.out, "a")
    for arm, ms in arms:
        for (m, t), v in sw.items():
            setattr(m, t, v)
        for m, t in ms:
            setattr(m, t, True)
        cap.clear()
        S.embed_sequences(model, {f"q{i}": aa for i in range(7)}, batch_size=8)
        for li in sorted(cap):
            r = grade(cap[li], attns[li], n)
            r.update(arm=arm, layer=li, model=a.model, fast=a.fast)
            print(json.dumps(r), flush=True)
            out.write(json.dumps(r) + "\n"); out.flush()
    return 0


def grade(c, at, n):
    """One layer's float64 grades (see the module doc)."""
    w = at.weights
    H, hd = at.n_heads, at.head_dim
    x = c["x"][:n]
    xn = torch.nn.functional.layer_norm(x, x.shape[-1:], w["norm.weight"].double(), w["norm.bias"].double(), 1e-5)
    qkv = xn @ w["qkv.weight"].double().t() + w["qkv.bias"].double()
    d = H * hd
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    f = torch.outer(torch.arange(n, dtype=torch.float64), inv)
    cs, sn = f.cos(), f.sin()

    def heads(t, rot):
        t = t.reshape(n, H, hd).transpose(0, 1)
        if not rot:
            return t
        t1, t2 = t[..., : hd // 2], t[..., hd // 2:]
        return torch.cat([t1 * cs - t2 * sn, t2 * cs + t1 * sn], -1)
    ref = {"q": heads(qkv[:, :d], True), "k": heads(qkv[:, d:2 * d], True), "v": heads(qkv[:, 2 * d:], False)}
    r = {}
    got = {}
    for nm in "qkv":
        g = c[nm][:, :n, :hd] if c[nm].dim() == 3 else c[nm]
        got[nm] = g
        dl = g - ref[nm]
        r[nm] = {"rel_l2": float(dl.norm() / ref[nm].norm()), "bias": float((dl * ref[nm].sign()).mean() / ref[nm].abs().mean())}
    sc = hd ** -0.5
    o = c["o"][:n].reshape(n, H, -1)[..., :hd].transpose(0, 1)   # merged [L, H*D] -> [H, L, D]
    errs = {}
    for s in SCALES:
        o64 = attn64(got["q"], got["k"], got["v"], n, sc * s)
        errs[s] = float((o - o64).norm() / o64.norm())
    o_true = attn64(ref["q"], ref["k"], ref["v"], n, sc)
    r["sdpa"] = {"rel_l2": errs[1.0], "fit_scale": min(errs, key=errs.get), "fit_rel_l2": min(errs.values()),
                 "vs_true_rel_l2": float((o - o_true).norm() / o_true.norm())}
    return r


if __name__ == "__main__":
    raise SystemExit(main())
