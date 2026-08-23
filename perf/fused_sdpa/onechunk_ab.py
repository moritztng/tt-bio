#!/usr/bin/env python3
"""Interleaved A/B of the one-k-chunk fused SDPA against the arm `_tri_att_sdpa_hifi` serves today.

`onechunk_l1.py` timed each arm in a block and measured two byte-identical programs 2x apart
(0.129 vs 0.263 ms at S=128), so its times are noise. This interleaves the arms rep by rep and
carries an A/A control -- the shipped arm measured twice under two labels -- so the noise floor is
in the same table as the effect. Accuracy is not re-measured here; `onechunk_l1.py` owns that.

Usage: onechunk_ab.py [--sizes 320,512,768,1024] [--reps 15] [--drop 5]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l1_account as LA

HEADS, HEAD_DIM, BATCH_CAP = 4, 32, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='320,512,768,1024')
    ap.add_argument('--reps', type=int, default=15)
    ap.add_argument('--drop', type=int, default=5)
    ap.add_argument('--out', type=Path, default=Path(__file__).with_name('onechunk_ab.json'))
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as TS
    assert Path(T.__file__).resolve().is_relative_to(ROOT)

    dev = T.get_device()
    ckc = T._TRIATT_FUSED_HIFI_CKC
    res = {'reps': a.reps, 'drop': a.drop, 'sizes': {}}

    for S in [int(x) for x in a.sizes.split(',')]:
        torch.manual_seed(0)
        B, scale_inv = min(S, BATCH_CAP), HEAD_DIM ** -0.5
        mk = lambda *sh: ttnn.from_torch(
            torch.randn(*sh, dtype=torch.float32).to(torch.bfloat16), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v, bias = mk(B, HEADS, S, HEAD_DIM), mk(B, HEADS, S, HEAD_DIM), \
            mk(B, HEADS, S, HEAD_DIM), mk(1, HEADS, S, S)
        padded = T._padded_sdpa_len(S)
        shipped_k = T._sdpa_chunks_shipped(S, S)[1]
        shipped_q = next(qc for qc in T._tri_att_q_chunks(S, S)
                         if LA.report(padded, qc, shipped_k)['fits'])
        # widest dividing q that fits at one k chunk, at each buffer factor
        qs = sorted({padded // n for n in range(1, padded // 32 + 1)
                     if padded % n == 0 and (padded // n) % 32 == 0}, reverse=True)
        legs = {'shipped': (shipped_q, shipped_k, 2), 'AA_early': (shipped_q, shipped_k, 2)}
        for kvbf in (2, 1):
            qc = next((x for x in qs
                       if LA.report(padded, x, padded, kv_buffer_factor=kvbf)['fits']), None)
            if qc:
                legs[f'1chunk_kvbf{kvbf}'] = (qc, padded, kvbf)
        # a second copy of the shipped leg LAST, so a systematic position effect inside one rep
        # shows up as AA_early != AA_late instead of being attributed to the arm in that slot.
        legs['AA_late'] = (shipped_q, shipped_k, 2)

        def call(cfg):
            o = TS.sdpa(q, k, v, bias, scale_inv, cfg[0], cfg[1], ckc_default=ckc,
                        kv_buffer_factor=cfg[2])
            assert o is not None, f'{S} {cfg} declined'
            return o

        for cfg in legs.values():                     # compile + warm every leg before timing
            ttnn.deallocate(call(cfg))
        ttnn.synchronize_device(dev)

        t = {n: [] for n in legs}
        for _ in range(a.reps):                       # interleaved: one rep of each leg, in turn
            for n, cfg in legs.items():
                t0 = time.perf_counter()
                o = call(cfg)
                ttnn.synchronize_device(dev)
                t[n].append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(o)

        def med(xs):
            xs = sorted(xs[a.drop:])
            return xs[len(xs) // 2]
        base = med(t['shipped'])
        out = {n: {'q_chunk': legs[n][0], 'k_chunk': legs[n][1], 'kv_buffer_factor': legs[n][2],
                   'ms_med': med(t[n]), 'ms_min': min(t[n][a.drop:]),
                   'speedup_vs_shipped': base / med(t[n]),
                   'l1': LA.report(padded, legs[n][0], legs[n][1],
                                   kv_buffer_factor=legs[n][2])['device_total']}
               for n in legs}
        for n, r in out.items():
            print(f"S={S:5d} {n:14s} q={r['q_chunk']:5d} k={r['k_chunk']:5d} "
                  f"kvbf={r['kv_buffer_factor']} {r['ms_med']:8.3f} ms med "
                  f"{r['speedup_vs_shipped']:6.3f}x  L1 {r['l1']}", flush=True)
        res['sizes'][str(S)] = {'batch': B, 'legs': out}
        for x in (q, k, v, bias):
            ttnn.deallocate(x)
        a.out.write_text(json.dumps(res, indent=1) + '\n')
    print('wrote', a.out, flush=True)
    T.cleanup()


if __name__ == '__main__':
    main()
