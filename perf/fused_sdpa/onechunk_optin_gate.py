#!/usr/bin/env python3
"""Gate for the `one_k_chunk` opt-in on `_tri_att_sdpa_hifi`.

A silently-declined config is indistinguishable from an absent one, so "the opt-in is wired" is
only true if the call can be made to say which config it ran. Two arms per size:

  off   one_k_chunk=False -- must serve the shipped k_chunk and be BYTE-IDENTICAL to a direct
        `triatt_sdpa.sdpa` at that config, i.e. the patch changes nothing when left alone.
  on    one_k_chunk=True  -- must serve k_chunk = the padded key length wherever that differs from
        the shipped pick, and must be less coherently biased than `off` there.

This gate covers the helper only. `onechunk_module_gate.py` covers the other half, that the
`TriangleAttention(tri_att_one_k_chunk=...)` kwarg reaches the helper from a real module.

Exits non-zero on any failure. Usage: onechunk_optin_gate.py [--sizes 128,256,320,512,768,1024]
"""
import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HEADS, HEAD_DIM, BATCH_CAP = 4, 32, 256


def stats(t):
    import torch
    d = (t.float() - 1.0).flatten()
    n = d.numel()
    mean, std = d.mean().item(), d.std().item()
    return {'n': n, 'mean': mean, 'std': std,
            'frac_above_one': (d > 0).float().mean().item(),
            'mean_over_sem': (abs(mean) / (std / math.sqrt(n))) if std > 0 else float('inf'),
            'rel_rms': d.pow(2).mean().sqrt().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='128,256,320,512,768,1024')
    ap.add_argument('--out', type=Path, default=Path(__file__).with_name('onechunk_optin_gate.json'))
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as TS
    assert Path(T.__file__).resolve().is_relative_to(ROOT)

    dev = T.get_device()
    res, fails = {'sizes': {}}, []

    for S in [int(x) for x in a.sizes.split(',')]:
        torch.manual_seed(0)
        B, scale_inv = min(S, BATCH_CAP), HEAD_DIM ** -0.5
        mk = lambda *sh: ttnn.from_torch(
            torch.randn(*sh, dtype=torch.float32).to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, bias = mk(B, HEADS, S, HEAD_DIM), mk(B, HEADS, S, HEAD_DIM), mk(1, HEADS, S, S)
        v = ttnn.from_torch(torch.ones(B, HEADS, S, HEAD_DIM, dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        padded_k = T._padded_sdpa_len(S)
        shipped_k = T._sdpa_chunks_shipped(S, S)[1]
        rec = {'padded_k': padded_k, 'shipped_k': shipped_k, 'batch': B,
               'one_chunk_is_shipped': padded_k == shipped_k}

        def hifi(one):
            T.TRIATT_FUSED_HIFI_PICKS.clear()
            o = T._tri_att_sdpa_hifi(q, k, v, bias, scale_inv, one_k_chunk=one)
            assert o is not None, f'S={S} one_k_chunk={one} declined outright'
            return o, list(T.TRIATT_FUSED_HIFI_PICKS[(S, S)])

        off, pick_off = hifi(False)
        on, pick_on = hifi(True)
        # the pre-patch behaviour, reconstructed: the widest q the ladder offers at the shipped k
        ref = None
        for qc in T._tri_att_q_chunks(S, S):
            ref = TS.sdpa(q, k, v, bias, scale_inv, qc, shipped_k,
                          ckc_default=T._TRIATT_FUSED_HIFI_CKC)
            if ref is not None:
                rec['reference_q_chunk'] = qc
                break
        assert ref is not None, f'S={S} no shipped-k config serves at all'

        t_off, t_on, t_ref = (ttnn.to_torch(x) for x in (off, on, ref))
        rec['pick_off'] = pick_off
        rec['pick_on'] = pick_on
        rec['off_is_shipped_k'] = pick_off[1] == shipped_k
        rec['on_is_one_chunk'] = pick_on[1] == padded_k
        rec['off_bit_identical_to_reference'] = bool(torch.equal(t_off, t_ref))
        rec['on_equals_off'] = bool(torch.equal(t_on, t_off))
        rec['stats_off'] = stats(t_off)
        rec['stats_on'] = stats(t_on)
        rec['bias_factor_on_vs_off'] = abs(rec['stats_off']['mean']) / abs(rec['stats_on']['mean'])

        if not rec['off_is_shipped_k']:
            fails.append(f'S={S}: one_k_chunk=False served k_chunk {pick_off[1]}, not {shipped_k}')
        if not rec['off_bit_identical_to_reference']:
            fails.append(f'S={S}: one_k_chunk=False is not byte-identical to the pre-patch config')
        if not rec['on_is_one_chunk']:
            fails.append(f'S={S}: one_k_chunk=True served k_chunk {pick_on[1]}, not {padded_k}')
        if padded_k == shipped_k and not rec['on_equals_off']:
            fails.append(f'S={S}: one chunk IS the shipped pick here, so the arms must be identical')
        if padded_k != shipped_k and rec['bias_factor_on_vs_off'] < 1.0:
            fails.append(f'S={S}: one chunk is MORE biased '
                         f"({rec['stats_on']['mean']:+.6f} vs {rec['stats_off']['mean']:+.6f})")

        print(f"S={S:5d} off {pick_off} mean={rec['stats_off']['mean']:+.6f} | "
              f"on {pick_on} mean={rec['stats_on']['mean']:+.6f} | "
              f"{rec['bias_factor_on_vs_off']:.2f}x less biased | "
              f"off==pre-patch {rec['off_bit_identical_to_reference']}", flush=True)
        res['sizes'][str(S)] = rec
        for x in (off, on, ref, q, k, v, bias):
            ttnn.deallocate(x)
        a.out.write_text(json.dumps(res, indent=1) + '\n')

    res['fails'] = fails
    a.out.write_text(json.dumps(res, indent=1) + '\n')
    T.cleanup()
    if fails:
        print('GATE FAIL:'); [print(' -', f) for f in fails]; sys.exit(1)
    print('GATE PASS: opt-in reaches the one-k-chunk config at every size, and off is byte-identical'
          ' to the pre-patch config')


if __name__ == '__main__':
    main()
