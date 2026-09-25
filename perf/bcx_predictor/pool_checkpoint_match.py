#!/usr/bin/env python3
"""Does the card's trunk have to be the same checkpoint as the JAX side around it?

The shipped pool's first full trajectory returned ptm = iptm = 1.0 for all fifteen mutate
rounds, against 0.65-0.87 on BindCraft 2's own JAX for the same stage, and the trajectory
was rejected on the pLDDT it reported there. Both the predict path and the gradient path
call `_resolve_model_name` themselves (`bindcraft/af2.py:287,381`), so resolving it a
second time in `ttbio_predictor` to pick the card's trunk SAMPLES A SECOND TIME: the card
held one checkpoint and the embedder, templates, structure module and heads around it came
from another, four calls in five.

Three predicts of the same state, one variable:

  matched    card model_1_multimer_v3, JAX model_1_multimer_v3
  mismatched card model_2_multimer_v3, JAX model_1_multimer_v3
  jax        BindCraft 2's own trunk, model_1_multimer_v3     (the reference)

Forward only, no tape, no levers.
"""
import argparse, json, os, pathlib, sys, time

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
ap.add_argument('--json', default=str(HERE / 'parity_artifacts' / 'pool_checkpoint_match.json'))
args = ap.parse_args()

M1, M2 = 'model_1_multimer_v3', 'model_2_multimer_v3'
settings = B.campaign_settings(overrides=[f'length_bucket_size={args.bucket}', 'campaign_seed=0'])
_ds, states, _losses = B.design_state(settings)
shape = B.state_shape(states)
n = sum(sum(c.values()) for c in shape.values())

def build(trunk, pool=None):
    return TTBioAlphaFoldDesignModel(presets=MP.POOL, data_dir=args.params, models=MP.POOL,
                                     num_recycle=1, length_bucket_size=args.bucket,
                                     max_cache_size=4, trunk=trunk, pool=pool)

def read(pred, tag, secs):
    m = pred['hPDL1'].metrics
    return {'arm': tag, 'seconds': round(secs, 1),
            'ptm': float(m['ptm']), 'iptm': float(m['iptm']),
            'plddt_mean': float(np.asarray(m['plddt']).mean()),
            'pae_mean_A': float(np.asarray(m['pae']).mean())}

def ca(pred, chain):
    return np.asarray(pred['hPDL1'].protein_complex[chain].atoms)[:, 1, :].astype(np.float64)

def kabsch_rmsd(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(-1).mean()))

out = {'n_residues': n, 'state_shape': shape, 'bucket': args.bucket, 'rows': []}
preds = {}

pool = MP.MultimerPool(args.params, resident=1, log_path=None)
pool.use(M1)
evo = EvoformerOnDevice(pool, k_evo=48)
with evoformer_on_device(evo):
    dev_model = build('device', pool)
    for tag, card in (('matched', M1), ('mismatched', M2)):
        pool.use(card)
        t0 = time.time()
        preds[tag] = dev_model.predict(states, model=M1)
        row = read(preds[tag], tag, time.time() - t0)
        row.update({'card_checkpoint': card, 'jax_checkpoint': M1})
        out['rows'].append(row); print(json.dumps(row), flush=True)
out['device_calls'] = dict(evo.calls)

t0 = time.time()
preds['jax'] = build('jax').predict(states, model=M1)
row = read(preds['jax'], 'jax', time.time() - t0)
row.update({'card_checkpoint': None, 'jax_checkpoint': M1})
out['rows'].append(row); print(json.dumps(row), flush=True)

out['binder_ca_rmsd_kabsch_A'] = {
    tag: round(kabsch_rmsd(ca(preds['jax'], 'binder'), ca(preds[tag], 'binder')), 3)
    for tag in ('matched', 'mismatched')}
out['stamp'] = A.stamp(2)
p = pathlib.Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != 'stamp'}, indent=1, default=str))
