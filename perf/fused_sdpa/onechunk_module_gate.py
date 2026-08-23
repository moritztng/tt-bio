#!/usr/bin/env python3
"""Does `TriangleAttention(tri_att_one_k_chunk=True)` actually reach the one-k-chunk config?

`onechunk_optin_gate.py` proves the helper. This proves the kwarg is threaded from the module
constructor through `forward`'s `_attend_heads` closure to the kernel, by building a real
TriangleAttention on synthetic weights and reading `TRIATT_FUSED_HIFI_PICKS` afterwards. Synthetic
weights are fine: the question is which config the ladder picks, which depends on the shapes and
the flags, not on the values. The two arms' outputs are also compared, so "the flag did something"
is a measurement rather than an assumption.

Needs TT_BIO_TRIATT_FUSED_HIFI=1; the fused-HiFi route is off by default and this gate scores that
route. Exits non-zero on failure.

Usage: onechunk_module_gate.py [--sizes 320,512]
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HEADS, HEAD_DIM, HIDDEN = 4, 32, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='320,512')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).with_name('onechunk_module_gate.json'))
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    assert Path(T.__file__).resolve().is_relative_to(ROOT)
    assert T._TRIATT_FUSED_HIFI, 'set TT_BIO_TRIATT_FUSED_HIFI=1: this gate scores that route'

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=False)

    inner = HEADS * HEAD_DIM
    torch.manual_seed(0)
    sd = {
        'layer_norm.weight': torch.ones(HIDDEN),
        'layer_norm.bias': torch.zeros(HIDDEN),
        'linear.weight': torch.randn(HEADS, HIDDEN) * 0.05,
        'linear_q.weight': torch.randn(inner, HIDDEN) * 0.05,
        'linear_k.weight': torch.randn(inner, HIDDEN) * 0.05,
        'linear_v.weight': torch.randn(inner, HIDDEN) * 0.05,
        'linear_g.weight': torch.randn(inner, HIDDEN) * 0.05,
        'linear_o.weight': torch.randn(HIDDEN, inner) * 0.05,
    }

    res, fails = {'sizes': {}}, []
    for S in [int(x) for x in a.sizes.split(',')]:
        torch.manual_seed(1)
        x = torch.randn(1, S, S, HIDDEN)
        shipped_k = T._sdpa_chunks_shipped(S, S)[1]
        padded_k = T._padded_sdpa_len(S)
        rec, outs = {'shipped_k': shipped_k, 'padded_k': padded_k}, {}
        for label, one in (('off', False), ('on', True)):
            mod = T.TriangleAttention(HEAD_DIM, HEADS, ending=False, state_dict=sd,
                                      compute_kernel_config=ckc, fp32_softmax=True,
                                      tri_att_one_k_chunk=one)
            T.TRIATT_FUSED_HIFI_PICKS.clear()
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            o = mod(xt)
            picks = {str(kk): vv for kk, vv in T.TRIATT_FUSED_HIFI_PICKS.items()}
            outs[label] = ttnn.to_torch(o)
            rec[f'picks_{label}'] = picks
            ttnn.deallocate(o)
            assert picks, f'S={S} {label}: no fused-HiFi call was served at all'
            k_served = sorted({v[1] for v in picks.values()})
            rec[f'k_served_{label}'] = k_served
            want = padded_k if one else shipped_k
            if k_served != [want]:
                fails.append(f'S={S} {label}: served k_chunk {k_served}, wanted [{want}]')
        rec['outputs_differ'] = not bool(torch.equal(outs['off'], outs['on']))
        rec['rel_delta'] = ((outs['on'].float() - outs['off'].float()).pow(2).mean().sqrt()
                            / outs['off'].float().pow(2).mean().sqrt()).item()
        if padded_k != shipped_k and not rec['outputs_differ']:
            fails.append(f'S={S}: the flag changed the config but not the output')
        print(f"S={S:5d} off k={rec['k_served_off']} on k={rec['k_served_on']} "
              f"outputs differ={rec['outputs_differ']} rel_delta={rec['rel_delta']:.6f}", flush=True)
        res['sizes'][str(S)] = rec
        a.out.write_text(json.dumps(res, indent=1) + '\n')

    res['fails'] = fails
    a.out.write_text(json.dumps(res, indent=1) + '\n')
    T.cleanup()
    if fails:
        print('GATE FAIL:'); [print(' -', f) for f in fails]; sys.exit(1)
    print('GATE PASS: the constructor kwarg reaches the kernel and selects the one-k-chunk config')


if __name__ == '__main__':
    main()
