#!/usr/bin/env python3
"""Does BindCraft 2's forward-only predict agree with its own trunk at the size it runs?

The gradient stages go through the taped path; `predict` goes through the primal one
(`splice.EvoformerOnDevice._primal`), and the campaign's first use of it is the mutate
stage. On the shipped pool's first full trajectory -- binder 146, n=261 padded to 288 --
every mutate round came back ptm = iptm = 1.0 (2 dp) and binder pLDDT 0.6, where BindCraft 2's
own JAX gave 0.65-0.87 and 0.78. The primal path had only ever been checked at n=192.

One checkpoint, one state per size, device against BindCraft 2's trunk, dropout off as
`run_mutation_polish` sets it.
"""
import argparse, json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / 'perf' / 'bcx_afgrad'), str(ROOT / 'perf' / 'bcx_stack')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import bc2_state as B
import afgrad as A
import multimer_pool as MP
from splice import EvoformerOnDevice, evoformer_on_device
from ttbio_predictor import TTBioAlphaFoldDesignModel

ap = argparse.ArgumentParser()
ap.add_argument('--params', default='/home/ttuser/bcx_e2e/af2_params')
ap.add_argument('--bucket', type=int, default=32)
ap.add_argument('--card', type=int, default=2)
ap.add_argument('--lengths', default='146,77')
ap.add_argument('--dropout', action='store_true')
ap.add_argument('--json', default=str(HERE / 'parity_artifacts' / 'predict_parity_size.json'))
args = ap.parse_args()

M1 = 'model_1_multimer_v3'

def state_for(binder_length):
    overrides = [f'length_bucket_size={args.bucket}', 'campaign_seed=0']
    if binder_length:
        overrides.append(f'binder_lengths=[{binder_length}]')
    settings = B.campaign_settings(overrides=overrides)
    _ds, states, _losses = B.design_state(settings)
    return states, sum(sum(c.values()) for c in B.state_shape(states).values())

def build(trunk):
    m = TTBioAlphaFoldDesignModel(presets=(M1,), data_dir=args.params, models=(M1,),
                                  num_recycle=1, length_bucket_size=args.bucket,
                                  max_cache_size=6, trunk=trunk, pool=None)
    m.dropout = args.dropout
    return m

def read(pred, tag, n, secs):
    m = pred['hPDL1'].metrics
    plddt = np.asarray(m['plddt'])
    return {'arm': tag, 'n_residues': n, 'padded': (n + 31) // 32 * 32,
            'seconds': round(secs, 1), 'ptm': round(float(m['ptm']), 4),
            'iptm': round(float(m['iptm']), 4),
            'plddt_mean': round(float(plddt.mean()), 4),
            'plddt_binder': round(float(np.asarray(
                pred['hPDL1'].protein_complex['binder'].atoms).shape[0]), 0),
            'pae_mean_A': round(float(np.asarray(m['pae']).mean()), 3)}

lengths = [int(x) for x in args.lengths.split(',')]
states = {L: state_for(L) for L in lengths}
out = {'bucket': args.bucket, 'checkpoint': M1, 'dropout': args.dropout, 'rows': []}

pool = MP.MultimerPool(args.params, resident=1, log_path=None)
pool.use(M1)
evo = EvoformerOnDevice(pool, k_evo=48)
with evoformer_on_device(evo):
    dev = build('device')
    for L in lengths:
        s, n = states[L]
        t0 = time.time()
        row = read(dev.predict(s, model=M1), 'device', n, time.time() - t0)
        row['binder_length'] = L; out['rows'].append(row); print(json.dumps(row), flush=True)

ref = build('jax')
for L in lengths:
    s, n = states[L]
    t0 = time.time()
    row = read(ref.predict(s, model=M1), 'jax', n, time.time() - t0)
    row['binder_length'] = L; out['rows'].append(row); print(json.dumps(row), flush=True)

out['device_calls'] = dict(evo.calls)
out['stamp'] = A.stamp(args.card)
p = pathlib.Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != 'stamp'}, indent=1, default=str))
