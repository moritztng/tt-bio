#!/usr/bin/env python3
"""What does it cost when the card's trunk is not the checkpoint the JAX side is using?

The shipped pool's first full trajectory returned ptm = iptm = 1.0 for all fifteen mutate
rounds against 0.65-0.87 on BindCraft 2's own JAX, and was rejected on the pLDDT it measured
there. `ttbio_predictor` resolved BindCraft 2's per-step model a second time to pick the
card's trunk, and resolving `None` samples rather than looks up, so the two sides disagreed
four times in five (`test_pool_checkpoint_match.py`, no card needed).

This is the same question with a number on it. Three predicts of one state, one variable:

  matched     card model_1_multimer_v3, JAX model_1_multimer_v3
  mismatched  card model_2_multimer_v3, JAX model_1_multimer_v3
  jax         BindCraft 2's own trunk, model_1_multimer_v3      -- the reference

The JAX side is model_1 in all three, and the predictor's own selection is switched off here
(`pool=None`) so the card is moved by hand and nothing else differs. Forward only.
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
ap.add_argument('--json', default=str(HERE / 'parity_artifacts' / 'pool_checkpoint_match.json'))
args = ap.parse_args()

M1, M2 = 'model_1_multimer_v3', 'model_2_multimer_v3'
settings = B.campaign_settings(overrides=[f'length_bucket_size={args.bucket}', 'campaign_seed=0'])
_ds, states, _losses = B.design_state(settings)
shape = B.state_shape(states)
n = sum(sum(c.values()) for c in shape.values())
print(json.dumps({'n_residues': n, 'state_shape': shape}), flush=True)

def build(trunk):
    # pool=None on purpose: the probe moves the card itself, so the predictor's own
    # selection is not in the way.
    return TTBioAlphaFoldDesignModel(presets=(M1,), data_dir=args.params, models=(M1,),
                                     num_recycle=1, length_bucket_size=args.bucket,
                                     max_cache_size=2, trunk=trunk, pool=None)

def read(pred, tag, secs, card):
    m = pred['hPDL1'].metrics
    return {'arm': tag, 'card_checkpoint': card, 'jax_checkpoint': M1,
            'seconds': round(secs, 1), 'ptm': round(float(m['ptm']), 4),
            'iptm': round(float(m['iptm']), 4),
            'plddt_mean': round(float(np.asarray(m['plddt']).mean()), 4),
            'pae_mean_A': round(float(np.asarray(m['pae']).mean()), 3)}

def ca(pred, chain):
    return np.asarray(pred['hPDL1'].protein_complex[chain].atoms)[:, 1, :].astype(np.float64)

def kabsch_rmsd(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(-1).mean()))

out = {'n_residues': n, 'state_shape': shape, 'bucket': args.bucket, 'rows': []}
preds = {}
clock = None
try:
    import stack as S
    clock = S.Clock()
except Exception as exc:                                  # levers are not on main
    print(f'no clock sampler: {exc}', flush=True)

pool = MP.MultimerPool(args.params, resident=2, log_path=None)
pool.use(M1)
evo = EvoformerOnDevice(pool, k_evo=48)
t_dev0 = time.time()
with evoformer_on_device(evo):
    dev_model = build('device')
    for tag, card in (('matched', M1), ('mismatched', M2)):
        pool.use(card)
        t0 = time.time()
        preds[tag] = dev_model.predict(states, model=M1)
        row = read(preds[tag], tag, time.time() - t0, card)
        out['rows'].append(row); print(json.dumps(row), flush=True)
t_dev1 = time.time()
out['device_calls'] = dict(evo.calls)
if clock is not None:
    clock.stop()
    out['aiclk'] = clock.window([(t_dev0, t_dev1)])

t0 = time.time()
preds['jax'] = build('jax').predict(states, model=M1)
row = read(preds['jax'], 'jax', time.time() - t0, None)
out['rows'].append(row); print(json.dumps(row), flush=True)

out['binder_ca_rmsd_kabsch_A'] = {
    tag: round(kabsch_rmsd(ca(preds['jax'], 'binder'), ca(preds[tag], 'binder')), 3)
    for tag in ('matched', 'mismatched')}
out['stamp'] = A.stamp(args.card)
p = pathlib.Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != 'stamp'}, indent=1, default=str))
