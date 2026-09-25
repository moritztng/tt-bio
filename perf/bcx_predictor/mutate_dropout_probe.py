#!/usr/bin/env python3
"""Why did the mutate stage report ptm = iptm = 1.0 on card?

`run_mutation_polish` (`bindcraft/trajectory.py:262`) changes exactly one thing about the
model before it starts: `design_model.dropout = False`. Every stage before it predicted
with dropout on. On the shipped pool's first full trajectory every mutate round came back
ptm = iptm = 1.0 and pLDDT 0.6, against 0.65-0.87 on BindCraft 2's own JAX, and the
trajectory was rejected on that pLDDT.

Four predicts of one state: dropout on and off, on card and on BindCraft 2's trunk. The
checkpoint is model_1_multimer_v3 on both sides throughout.
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
ap.add_argument('--json', default=str(HERE / 'parity_artifacts' / 'mutate_dropout.json'))
args = ap.parse_args()

M1 = 'model_1_multimer_v3'
settings = B.campaign_settings(overrides=[f'length_bucket_size={args.bucket}', 'campaign_seed=0'])
_ds, states, _losses = B.design_state(settings)
n = sum(sum(c.values()) for c in B.state_shape(states).values())

def build(trunk):
    return TTBioAlphaFoldDesignModel(presets=(M1,), data_dir=args.params, models=(M1,),
                                     num_recycle=1, length_bucket_size=args.bucket,
                                     max_cache_size=4, trunk=trunk, pool=None)

def read(pred, tag, dropout, secs):
    m = pred['hPDL1'].metrics
    return {'arm': tag, 'dropout': dropout, 'seconds': round(secs, 1),
            'ptm': round(float(m['ptm']), 4), 'iptm': round(float(m['iptm']), 4),
            'plddt_mean': round(float(np.asarray(m['plddt']).mean()), 4),
            'pae_mean_A': round(float(np.asarray(m['pae']).mean()), 3)}

out = {'n_residues': n, 'bucket': args.bucket, 'checkpoint': M1, 'rows': []}

pool = MP.MultimerPool(args.params, resident=1, log_path=None)
pool.use(M1)
evo = EvoformerOnDevice(pool, k_evo=48)
with evoformer_on_device(evo):
    dev = build('device')
    for dropout in (True, False):
        dev.dropout = dropout
        t0 = time.time()
        row = read(dev.predict(states, model=M1), 'device', dropout, time.time() - t0)
        out['rows'].append(row); print(json.dumps(row), flush=True)

ref = build('jax')
for dropout in (True, False):
    ref.dropout = dropout
    t0 = time.time()
    row = read(ref.predict(states, model=M1), 'jax', dropout, time.time() - t0)
    out['rows'].append(row); print(json.dumps(row), flush=True)

out['device_calls'] = dict(evo.calls)
out['stamp'] = A.stamp(args.card)
p = pathlib.Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != 'stamp'}, indent=1, default=str))
