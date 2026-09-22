#!/usr/bin/env python3
'''T1 and T3 in isolation: the two D55 reductions no model scope reaches.

The diffusion and trunk backwards both record zero firings for T1 (`layer_norm`'s recomputed
E[x]) and T3 (`triangle_attention`'s dbias `ttnn.sum(ds, dim=0)`), and `reach_probe.py` says
why: `layer_norm` is the explicit-call twin of `_taped_layer_norm` and production dispatches
the intercepted one (`_TAPED["layer_norm"]`), while `triangle_attention` is reached only
through the fused-SDPA verb, which neither boundary exercises.

Unreached is not inert, so this calls both ops DIRECTLY, on the card, at DiT shapes, and runs
the same three arms. T3's reduction is further gated on `bias_bcast`, so the bias here has a
leading extent of 1 -- the pair-track layout the docstring names -- or the branch under test
would not be the branch that runs.

Scored arm against arm. No reference, so D141 does not touch these numbers.
'''
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'perf', 'of3t_ditcot'))

import torch  # noqa: E402


def build(dev, seed):
    import ttnn
    import tt_bio.autograd as ag
    g = torch.Generator().manual_seed(seed)

    def tt(x, dt=ttnn.bfloat16):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)

    return ttnn, ag, g, tt


def case_t1(ttnn, ag, g, tt, N=512, C=768):
    '''ag.layer_norm backward -- T1 fires in its recomputed mean.'''
    x = ag.parameter(tt(torch.randn(1, N, C, generator=g)))
    gam = ag.parameter(tt(torch.randn(C, generator=g).abs() + 0.5))
    bet = ag.parameter(tt(torch.randn(C, generator=g) * 0.1))
    out = ag.layer_norm(x, gam, bet, eps=1e-5)
    out.backward(tt(torch.randn(1, N, C, generator=g)))
    return {'x': x, 'gamma': gam, 'beta': bet}


def case_t3(ttnn, ag, g, tt, B=8, H=16, N=128, D=64):
    '''ag.triangle_attention backward with a BROADCAST bias -- T3's gated branch.'''
    q = ag.parameter(tt(torch.randn(B, H, N, D, generator=g) * 0.3))
    k = ag.parameter(tt(torch.randn(B, H, N, D, generator=g) * 0.3))
    v = ag.parameter(tt(torch.randn(B, H, N, D, generator=g) * 0.3))
    bias = ag.parameter(tt(torch.randn(1, H, N, N, generator=g) * 0.2))
    out = ag.triangle_attention(q, k, v, bias)
    out.backward(tt(torch.randn(B, H, N, D, generator=g)))
    return {'q': q, 'k': k, 'v': v, 'bias': bias}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--kcfg-arm', required=True, dest='kcfg_arm',
                    choices=('none', 'pull', 'lofi'))
    ap.add_argument('--case', required=True, choices=('t1', 't3'))
    ap.add_argument('--seed', type=int, default=20260921)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fire-out', default='')
    a = ap.parse_args()

    import kcfg_pull as K
    lmap, rep, census = K.resolve()

    import ttnn
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    try:
        ttnn_, ag, g, tt = build(dev, a.seed)
        K.install(a.kcfg_arm, lmap)
        params = (case_t1 if a.case == 't1' else case_t3)(ttnn_, ag, g, tt)
        grads = {}
        for nm, p in params.items():
            gr = p.grad
            grads[nm] = ttnn.to_torch(gr.value if hasattr(gr, 'value') else gr).clone()
    finally:
        pass

    fired = {f'T{t}': K.FIRED.get(t, 0) for t in range(1, 5)}
    torch.save({'grads': grads, 'arm': a.kcfg_arm, 'case': a.case, 'fired': fired}, a.out)
    print('FIRINGS ' + json.dumps(fired), flush=True)
    if a.fire_out:
        pathlib.Path(a.fire_out).write_text(json.dumps(
            {'arm': a.kcfg_arm, 'case': a.case, 'fired': fired, 'targets': rep}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
