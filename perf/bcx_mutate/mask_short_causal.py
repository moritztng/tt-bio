#!/usr/bin/env python3
"""Does an MSA mask short by N real residues, on its own, drive pTM to 1.0? Card-free.

`EvoformerOnDevice._mask` cached the Evoformer MSA mask on the padded token shape, so the
second and later draws in a 32-bucket ran with an earlier draw's mask (`15716afea` fixes
it). On `profile_s1` the two draws served a stale mask both reported i_pTM 1.00 with pLDDT
collapsed and the two served their own mask did not. That is an order match, not a cause.

This isolates the variable with no device at all: BindCraft 2's own JAX `predict`, one
sequence, one checkpoint, and the ONLY difference between legs is that `alphafold_input_features`
returns an `msa_mask` with its last N real positions zeroed. `seq_mask` is left alone, so the
pair masks and every metric are computed exactly as before -- which is faithful, because
`_pair_mask` keys on `(shape, sum)` and did NOT collide, so the real arm ran a stale MSA mask
beside a correct pair mask.

N=9 is what draw l151 was served (266 real residues, given a 257-one mask) and N=25 is what
draw l167 is being served right now.
"""
import argparse, json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parents[1] / 'bcx_predictor'
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / 'perf' / 'bcx_afgrad'), str(ROOT / 'perf' / 'bcx_stack')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import jax.numpy as jnp
import bc2_state as B
import bindcraft.af2 as BAF2
from ttbio_predictor import TTBioAlphaFoldDesignModel

ap = argparse.ArgumentParser()
ap.add_argument('--params', default='/home/ttuser/bcx_e2e/af2_params')
ap.add_argument('--bucket', type=int, default=32)
ap.add_argument('--binder', type=int, default=151)
ap.add_argument('--recycle', type=int, default=1)
ap.add_argument('--short', default='0,9,25')
ap.add_argument('--model', default='model_4_multimer_v3')
ap.add_argument('--json', default='/home/ttuser/bcx_mutate_art/mask_short_causal.json')
args = ap.parse_args()

_ORIG = BAF2.alphafold_input_features
SHORT = {'n': 0, 'real': 0}


def patched(*a, **k):
    """Identical to BindCraft 2's featuriser, then zero the last N REAL msa_mask positions.

    `seq_mask` is a tracer inside the jitted `predict_complex_arrays`, so the real residue
    count is passed in from the host rather than read off it.
    """
    feats = _ORIG(*a, **k)
    n, real = SHORT['n'], SHORT['real']
    if n:
        feats = dict(feats)
        feats['msa_mask'] = feats['msa_mask'].at[:, real - n:real].set(0.0)
    return feats


BAF2.alphafold_input_features = patched

settings = B.campaign_settings(overrides=[f'length_bucket_size={args.bucket}',
                                          'campaign_seed=0',
                                          f'binder_lengths=[{args.binder}]'])
_ds, states, _losses = B.design_state(settings)
n_res = sum(sum(c.values()) for c in B.state_shape(states).values())
print(json.dumps({'binder': args.binder, 'n_residues': n_res,
                  'padded': (n_res + args.bucket - 1) // args.bucket * args.bucket,
                  'checkpoint': args.model, 'recycles': args.recycle}), flush=True)

out = {'n_residues': n_res, 'padded': (n_res + 31) // 32 * 32, 'checkpoint': args.model,
       'recycles': args.recycle, 'trunk': 'jax', 'card': None, 'rows': []}
SHORT['real'] = n_res
for short in [int(x) for x in args.short.split(',')]:
    SHORT['n'] = short
    # A fresh model per leg: `predict` caches its jitted callable, and the patch has to be
    # visible at TRACE time, not just at call time.
    m = TTBioAlphaFoldDesignModel(presets=(args.model,), data_dir=args.params,
                                  models=(args.model,), num_recycle=args.recycle,
                                  length_bucket_size=args.bucket, max_cache_size=2,
                                  trunk='jax', pool=None)
    m.dropout = False
    t0 = time.time()
    pred = m.predict(states, model=args.model)
    secs = time.time() - t0
    met = pred['hPDL1'].metrics
    plddt = np.asarray(met['plddt'])
    row = {'msa_mask_short_by': short,
           'real_residues_masked': short,
           'ptm': round(float(met['ptm']), 6),
           'iptm': round(float(met['iptm']), 6),
           'plddt_mean': round(float(plddt.mean()), 6),
           'plddt_min': round(float(plddt.min()), 6),
           'pae_mean_A': round(float(np.asarray(met['pae']).mean()), 4),
           'seconds': round(secs, 1)}
    out['rows'].append(row)
    print(json.dumps(row), flush=True)
    del m

pathlib.Path(args.json).write_text(json.dumps(out, indent=1))
print('wrote', args.json)
ctrl = out['rows'][0]
sat = [r for r in out['rows'][1:] if r['ptm'] >= 0.995]
print()
print('control ptm %.4f iptm %.4f plddt %.4f' % (ctrl['ptm'], ctrl['iptm'], ctrl['plddt_mean']))
print('%d of %d shortened legs reach ptm >= 0.995' % (len(sat), len(out['rows']) - 1))
