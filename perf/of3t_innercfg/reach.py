#!/usr/bin/env python3
"""Which of `softmax_bw_inner`'s three callers reach it under the SHIPPED default.

The helper's docstring names three: `autograd.softmax`, `autograd.triangle_attention` and
`taped_ttnn._v_softmax`. Under the shipped training default (`exact_training` ON, ops
`("softmax", "layer_norm")`) `_EXACT_OPS` replaces the VERBS, so one of the three is
answered on the host in float64 and never reaches the card. This counts the fires rather
than arguing it, and runs the control arm -- `exact_training(False)` -- so the counter is
shown able to move.
"""
from __future__ import annotations

import json, pathlib, platform, subprocess, sys, time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio  # noqa: E402
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT)

import torch, ttnn  # noqa: E402
from tt_bio import autograd as ag  # noqa: E402
from tt_bio import taped_ttnn as tt  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

FIRES = {"n": 0}
_real = ag.softmax_bw_inner


def _counting(y, g, dim=-1, config=None):
    FIRES["n"] += 1
    return _real(y, g, dim=dim, config=config)


def dev_t(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def run(dev, which, exact_on):
    """One caller, one arm. Returns the fire count and the exact-softmax counters."""
    FIRES["n"] = 0
    before = dict(ag.EXACT_SOFTMAX_STATS)
    torch.manual_seed(7)
    B, H, n, hd = 1, 4, 256, 64
    ctx = ag.exact_training(True) if exact_on else ag.exact_training(False)
    with ctx:
        with ag.tape():
            installed = ag.exact_softmax_installed()
            ops = ag.exact_training_ops()
            if which == "verb_softmax":
                x = ag.Tensor(dev_t(torch.randn(B, H, n, n).to(torch.bfloat16), dev),
                              requires_grad=True)
                y = tt.taped_ttnn().softmax(x, dim=-1)
                ag.backward(y, dev_t(torch.randn(B, H, n, n).to(torch.bfloat16), dev))
            elif which == "ag_softmax":
                x = ag.Tensor(dev_t(torch.randn(B, H, n, n).to(torch.bfloat16), dev),
                              requires_grad=True)
                y = ag.softmax(x, dim=-1)
                ag.backward(y, dev_t(torch.randn(B, H, n, n).to(torch.bfloat16), dev))
            elif which == "sdpa_verb":
                q = ag.Tensor(dev_t((torch.randn(B, H, n, hd) * .5).to(torch.bfloat16), dev),
                              requires_grad=True)
                k = ag.Tensor(dev_t((torch.randn(B, H, n, hd) * .5).to(torch.bfloat16), dev),
                              requires_grad=True)
                v = ag.Tensor(dev_t((torch.randn(B, H, n, hd) * .5).to(torch.bfloat16), dev),
                              requires_grad=True)
                b = ag.Tensor(dev_t((torch.randn(1, H, n, n) * .5).to(torch.bfloat16), dev),
                              requires_grad=True)
                y = tt.taped_ttnn().transformer.scaled_dot_product_attention(
                    q, k, v, attn_mask=b, is_causal=False)
                ag.backward(y, dev_t(torch.randn(B, H, n, hd).to(torch.bfloat16), dev))
            elif which == "triangle_attention":
                q = ag.Tensor(dev_t((torch.randn(B, H, n, hd) * .5).to(torch.bfloat16), dev),
                              requires_grad=True)
                k = ag.Tensor(dev_t((torch.randn(B, H, n, hd) * .5).to(torch.bfloat16), dev),
                              requires_grad=True)
                v = ag.Tensor(dev_t((torch.randn(B, H, n, hd) * .5).to(torch.bfloat16), dev),
                              requires_grad=True)
                b = ag.Tensor(dev_t((torch.randn(1, H, n, n) * .5).to(torch.bfloat16), dev),
                              requires_grad=True)
                y = ag.triangle_attention(q, k, v, b, scale=hd ** -0.5)
                ag.backward(y, dev_t(torch.randn(B, H, n, hd).to(torch.bfloat16), dev))
            else:
                raise ValueError(which)
    after = dict(ag.EXACT_SOFTMAX_STATS)
    return dict(caller=which, exact_training=exact_on, exact_softmax_installed=installed,
                exact_training_ops=list(ops),
                softmax_bw_inner_fires=FIRES["n"],
                exact_softmax_verb_calls=after["verb"] - before["verb"],
                exact_softmax_raw_calls=after["raw"] - before["raw"])


def main():
    dev = get_device()
    ag.softmax_bw_inner = _counting
    rep = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               host=platform.node(),
               commit=subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
               shipped_default_ops=list(ag.EXACT_TRAINING_OPS),
               renorm=bool(ag.SOFTMAX_BW_RENORM), fused=bool(ag.SOFTMAX_BW_FUSED))
    rows = []
    for which in ("verb_softmax", "ag_softmax", "sdpa_verb", "triangle_attention"):
        for exact_on in (True, False):
            try:
                rows.append(run(dev, which, exact_on))
            except Exception as e:                       # noqa: BLE001
                rows.append(dict(caller=which, exact_training=exact_on,
                                 error=f"{type(e).__name__}: {e}"))
            print(json.dumps(rows[-1]))
    rep["rows"] = rows
    p = pathlib.Path(_ROOT) / "perf/of3t_innercfg/REACH.json"
    p.write_text(json.dumps(rep, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
