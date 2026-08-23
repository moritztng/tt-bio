#!/usr/bin/env python3
"""Does the one-k-chunk fused SDPA fit at 512, what does it cost in L1, and what does it buy?

`state/fused-sdpa-adopt.md` §2 measured the one-k-chunk arm as 4.9x less coherently biased than the
shipped two-chunk arm at S=320 and recorded that it does not fit at S=512: L1 refused at 1591808 B
against 1572864 B, over by 18944 B. That measurement paired k_chunk = S with q_chunk = S. The two
chunk sizes are independent and only k_chunk sets the online-softmax reduction order, so this sweeps
q_chunk and the k/v buffer factor at k_chunk = S and reads the refusal, the bias and the time out of
one run.

Row-sum probe, as `rowsum_probe.py`: with v = 1 every output element is the row sum of the attention
weights, which is exactly 1.0, so the deviation IS the normalisation error and its mean is the
coherent component rel_rms cannot see.

Every arm's L1 total is also predicted by `l1_account.py` before it runs, and the prediction is
printed next to what the device says, so a refusal validates the accounting instead of just failing.

Usage: onechunk_l1.py [--sizes 320,512,768,1024] [--out .../onechunk_l1.json] [--reps 3]
"""
import argparse, json, math, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import l1_account as LA

HEADS, HEAD_DIM = 4, 32   # rf3/remap.py PAIRFORMER_DIMS = (32, 4, 24, 16)
# Batch = sequence is triangle attention's own geometry. Capped above 512 to bound DRAM: the row-sum
# statistic is per row and the L1 plan does not depend on the batch, so the cap changes neither.
BATCH_CAP = 256


def stats(t):
    import torch
    d = (t.float() - 1.0).flatten()
    n = d.numel()
    mean, std = d.mean().item(), d.std().item()
    return {'n': n, 'mean': mean, 'std': std, 'max_abs': d.abs().max().item(),
            'frac_above_one': (d > 0).float().mean().item(),
            'mean_over_sem': (abs(mean) / (std / math.sqrt(n))) if std > 0 else float('inf'),
            'rel_rms': d.pow(2).mean().sqrt().item()}


def reported_bytes(msg):
    """The `grow to N B which is beyond max L1 size of M B` figures tt-metal puts in the throw."""
    m = re.search(r'grow to (\d+) B which is beyond max L1 size of (\d+) B', msg or '')
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='320,512,768,1024')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', type=Path, default=Path(__file__).with_name('onechunk_l1.json'))
    a = ap.parse_args()

    import torch, ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.triatt_sdpa as TS

    assert Path(T.__file__).resolve().is_relative_to(ROOT), \
        f'tt_bio resolves to {T.__file__}, not this checkout -- set PYTHONPATH'

    dev = T.get_device()
    # the fused kernel takes the 4-tuple; the materialised path takes a real ttnn config object
    ckc = T._TRIATT_FUSED_HIFI_CKC
    fid, approx, fp32_acc, dst_full = ckc
    ckc_obj = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=fid, math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc, packer_l1_acc=False)
    grid = tuple(T.COMPUTE_GRID_MAIN)

    res = {'heads': HEADS, 'head_dim': HEAD_DIM, 'arch': str(dev.arch()),
           'grid': list(grid), 'reps': a.reps, 'ckc': str(ckc),
           'l1_budget': LA.L1_BUDGET, 'l1_base': LA.BASE, 'sizes': {}}

    for S in [int(x) for x in a.sizes.split(',')]:
        torch.manual_seed(0)
        scale_inv = HEAD_DIM ** -0.5
        B = min(S, BATCH_CAP)
        mk = lambda *sh: ttnn.from_torch(
            torch.randn(*sh, dtype=torch.float32).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = mk(B, HEADS, S, HEAD_DIM)
        k = mk(B, HEADS, S, HEAD_DIM)
        bias = mk(1, HEADS, S, S)
        v = ttnn.from_torch(torch.ones(B, HEADS, S, HEAD_DIM, dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        padded = T._padded_sdpa_len(S)
        shipped_k = T._sdpa_chunks_shipped(S, S)[1]
        arms, raw = {}, {}

        def fused(name, q_chunk, k_chunk, kv_bf):
            pred = LA.report(padded, q_chunk, k_chunk, kv_buffer_factor=kv_bf)
            key = (S, S, q_chunk, k_chunk, kv_bf)
            TS._PM_OVER_L1.discard(key)
            rec = {'q_chunk': q_chunk, 'k_chunk': k_chunk, 'kv_buffer_factor': kv_bf,
                   'k_num_chunks': pred['k_num_chunks'],
                   'l1_predicted': pred['device_total'], 'l1_fits_predicted': pred['fits']}
            out = TS.sdpa(q, k, v, bias, scale_inv, q_chunk, k_chunk, ckc_default=ckc,
                          kv_buffer_factor=kv_bf)
            if out is None:
                err = TS.PM_L1_ERRORS.get(key, '')
                got, budget = reported_bytes(err)
                rec.update(declined=True, l1_device=got, l1_device_budget=budget,
                           l1_model_error=(None if got is None else got - pred['device_total']),
                           error=(err or '')[:200])
                arms[name] = rec
                print(f"S={S} {name:26s} q={q_chunk} k={k_chunk} kvbf={kv_bf} DECLINED "
                      f"device={got} predicted={pred['device_total']} "
                      f"delta={rec['l1_model_error']}", flush=True)
                return
            # warm, then time with a sync immediately before the clock stops
            ttnn.synchronize_device(dev)
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                o2 = TS.sdpa(q, k, v, bias, scale_inv, q_chunk, k_chunk, ckc_default=ckc,
                             kv_buffer_factor=kv_bf)
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(o2)
            t = ttnn.to_torch(out)
            raw[name] = t
            rec.update(stats(t), declined=False, ms_min=min(ts), ms_med=sorted(ts)[len(ts) // 2])
            arms[name] = rec
            ttnn.deallocate(out)
            print(f"S={S} {name:26s} q={q_chunk} k={k_chunk} kvbf={kv_bf} "
                  f"mean={rec['mean']:+.6f} std={rec['std']:.6f} "
                  f"mean/sem={rec['mean_over_sem']:.1f} rel_rms={rec['rel_rms']:.6f} "
                  f"frac>1={rec['frac_above_one']:.4f} {rec['ms_min']:.3f} ms "
                  f"(L1 {rec['l1_predicted']})", flush=True)

        # the shipped materialised path
        o = T._fp32_softmax_attention(q, k, v, bias, scale_inv=scale_inv,
                                      compute_kernel_config=ckc_obj, out_dtype=ttnn.bfloat16,
                                      bias_scale_inv=1.0)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            o2 = T._fp32_softmax_attention(q, k, v, bias, scale_inv=scale_inv,
                                           compute_kernel_config=ckc_obj, out_dtype=ttnn.bfloat16,
                                           bias_scale_inv=1.0)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o2)
        arms['fp32_materialised'] = dict(stats(ttnn.to_torch(o)), declined=False,
                                         ms_min=min(ts), ms_med=sorted(ts)[len(ts) // 2])
        ttnn.deallocate(o)
        s = arms['fp32_materialised']
        print(f"S={S} {'fp32_materialised':26s} mean={s['mean']:+.6f} std={s['std']:.6f} "
              f"mean/sem={s['mean_over_sem']:.1f} rel_rms={s['rel_rms']:.6f} "
              f"{s['ms_min']:.3f} ms", flush=True)

        # what `_tri_att_sdpa_hifi` serves today: widest q from its own ladder, shipped k
        shipped_q = next(qc for qc in T._tri_att_q_chunks(S, S)
                         if LA.report(padded, qc, shipped_k)['fits'])
        fused('hifi_shipped', shipped_q, shipped_k, 2)

        # one k chunk: the dividing q ladder widest-first, at both k/v buffer factors
        qs = sorted({padded // n for n in range(1, padded // 32 + 1)
                     if padded % n == 0 and (padded // n) % 32 == 0}, reverse=True)
        for kv_bf in (2, 1):
            for qc in qs:
                fused(f'hifi_1chunk_q{qc}_kvbf{kv_bf}', qc, padded, kv_bf)
                if not arms[f'hifi_1chunk_q{qc}_kvbf{kv_bf}'].get('declined'):
                    break

        # every one-k-chunk arm must be byte-identical: q blocking and the k/v buffer factor change
        # neither the reduction order nor the arithmetic, only the L1 plan.
        eq, names = {}, [n for n in raw if n.startswith('hifi_1chunk_')]
        for n in names[1:]:
            eq[f'{names[0]} vs {n}'] = bool(torch.equal(raw[names[0]], raw[n]))
        print(f'S={S} one-chunk arms bit-identical: {eq}', flush=True)

        for t in (q, k, v, bias):
            ttnn.deallocate(t)
        res['sizes'][str(S)] = {'padded': padded, 'batch': B, 'shipped_k_chunk': shipped_k,
                                'shipped_q_chunk': shipped_q, 'arms': arms,
                                'one_chunk_bit_identical': eq}
        raw.clear()
        a.out.write_text(json.dumps(res, indent=1) + '\n')

    print('wrote', a.out, flush=True)
    T.cleanup()


if __name__ == '__main__':
    main()
