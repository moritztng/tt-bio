#!/usr/bin/env python3
'''Why do T1 and T3 never fire? Count the entries and the branch, do not infer them.

T1 (`layer_norm`) and T3 (`triangle_attention`'s dbias `ttnn.sum(ds, dim=0)`) recorded zero
firings on both the diffusion scope and the trunk. A zero can mean the op never runs or that
it runs and takes the other branch, and the kernel-config question is only meaningful for the
first reading. This counts:

  * `autograd.layer_norm` entries and its backward entries          -- is T1's rule live at all
  * `triangle_attention` entries, and for each, whether `bias is None`
    and what `bias_bcast` evaluates to                              -- which branch T3 sits on

No kernel config is injected. Measurement only.
'''
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'perf', 'of3t_diffusion'))
sys.path.insert(0, os.path.join(os.getcwd(), 'perf', 'of3t_trunkg043'))
sys.path.insert(0, os.path.join(os.getcwd(), 'perf', 'of3t_gradients'))

C = {'layer_norm_fw': 0, 'layer_norm_bw': 0, 'taped_layer_norm_fw': 0,
     'tri_attn_fw': 0, 'tri_attn_bias_none': 0, 'tri_attn_bias_bcast': 0,
     'tri_attn_bias_not_bcast': 0, 'tri_attn_bias_shape0': []}


def install():
    import tt_bio.autograd as ag

    real_ln = ag.layer_norm
    real_tln = ag._taped_layer_norm
    real_ta = ag.triangle_attention

    def ln(*a, **k):
        C['layer_norm_fw'] += 1
        return real_ln(*a, **k)

    def tln(*a, **k):
        C['taped_layer_norm_fw'] += 1
        return real_tln(*a, **k)

    def ta(*a, **k):
        C['tri_attn_fw'] += 1
        bias = k.get('bias')
        if bias is None and len(a) >= 4:
            bias = a[3]
        if bias is None:
            C['tri_attn_bias_none'] += 1
        else:
            try:
                s0 = int(bias.value.shape[0])
            except Exception:
                s0 = -1
            if len(C['tri_attn_bias_shape0']) < 12:
                C['tri_attn_bias_shape0'].append(s0)
            if s0 == 1:
                C['tri_attn_bias_bcast'] += 1
            else:
                C['tri_attn_bias_not_bcast'] += 1
        return real_ta(*a, **k)

    ag.layer_norm = ln
    ag._taped_layer_norm = tln
    ag._TAPED['layer_norm'] = tln
    ag.triangle_attention = ta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--harness', default='trunk', choices=('diffusion', 'trunk'))
    ap.add_argument('--probe-out', default='')
    a, rest = ap.parse_known_args()

    install()
    if a.harness == 'diffusion':
        import device_gradient as H
    else:
        import dev_grad as H
    sys.argv = [sys.argv[0]] + rest
    rc = H.main()

    print('REACH ' + json.dumps(C), flush=True)
    if a.probe_out:
        pathlib.Path(a.probe_out).write_text(
            json.dumps({'harness': a.harness, 'counts': C}, indent=1))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
