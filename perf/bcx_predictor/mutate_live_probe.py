#!/usr/bin/env python3
"""Fold the state the mutate stage ACTUALLY passes `predict`, on card and on JAX, in the act.

Every probe so far reconstructed that state and every reconstruction came back healthy:
`mutate_sequence_probe.py` folds the recorded sequence at device 0.8386 ptm against BindCraft
2's own JAX at 0.8404, `mutate_onset_probe.py` is bit-identical across 12 taped steps and 12
backward passes, and again across the pool's checkpoint evictions and reloads. The live
trajectory still logs ptm = iptm = 1.0 from mutate round 1 for that sequence. So the
reconstruction is what to stop doing: this runs BindCraft 2's own `run_campaign` and reads the
state at the call, not a rebuild of it.

Two things make it cheap enough to iterate on. The stage budgets are settings
(`{stage}_steps`, `bindcraft/settings.py:612`), so the mutate stage is reachable in a handful
of gradient steps instead of 125. And when a prediction comes back saturated, the SAME
`protein_states` object is folded again on BindCraft 2's own trunk in the same process, which
is the device-vs-JAX reading on the live state that nobody has taken: if JAX also saturates,
the state is degenerate and the card is exonerated a fourth time; if it does not, the device
`predict` is wrong in a context no reconstruction reproduces.
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
import bindcraft.campaign as campaign
from bindcraft.settings import parse_setting_overrides, read_settings
from bindcraft.preflight import cleaned_campaign_settings
from ttbio_predictor import sequence_letters, scalar_metrics

ap = argparse.ArgumentParser()
ap.add_argument('--params', default='/home/ttuser/bcx_e2e/af2_params')
ap.add_argument('--out', default='/home/ttuser/bcx_multimer_art/mutate_live')
ap.add_argument('--card', type=int, default=2)
ap.add_argument('--bucket', type=int, default=32)
ap.add_argument('--binder-length', type=int, default=146)
ap.add_argument('--stage-steps', default='screen=2,refine=1,anneal=1,harden=1,mutate=3',
                help='BindCraft 2 stage budgets. The defaults total 125 gradient steps and '
                     '2.25 h; this reaches the mutate stage in about five.')
ap.add_argument('--pool-resident', type=int, default=1)
ap.add_argument('--no-levers', action='store_true')
ap.add_argument('--shadow-budget', type=int, default=3,
                help='how many saturated predicts to re-fold on JAX; each is ~110 s at n=288')
ap.add_argument('--saturated-at', type=float, default=0.99)
ap.add_argument('--relax-filters', action='store_true',
                help='drop min_iptm_/min_plddt_ to 0 on every gradient stage. The shipped '
                     'thresholds reject a short run at screen before it ever reaches mutate, '
                     'and the mutate predict is the whole point of the run.')
ap.add_argument('--override', action='append', default=[],
                help='any further BindCraft 2 setting override, repeatable')
args = ap.parse_args()

project = args.out
pathlib.Path(project).mkdir(parents=True, exist_ok=True)
ROWS = pathlib.Path(project) / 'live_predicts.jsonl'

overrides = ['campaign_seed=0', 'max_trajectories=1', f'length_bucket_size={args.bucket}',
             f'binder_lengths=[{args.binder_length}]']
STAGES = ('screen', 'refine', 'anneal', 'harden', 'mutate')
for item in args.stage_steps.split(','):
    stage, rounds = item.split('=')
    overrides.append(f'{stage}_steps={rounds}')
if args.relax_filters:
    overrides += [f'min_iptm_{stage}=0.0' for stage in STAGES]
    overrides += [f'min_plddt_{stage}=0.0' for stage in STAGES]
overrides += list(args.override)
settings = cleaned_campaign_settings(
    read_settings(os.path.join(B.BC2, 'examples', 'pdl1.json'),
                  parse_setting_overrides(overrides)))

import multimer_pool as MP
pool = MP.MultimerPool(args.params, resident=args.pool_resident or None,
                       log_path=os.path.join(project, 'pool_selections.jsonl'))

import ttbio_predictor as T
import functools

STATE = {'taped': 0, 'predicts': 0, 'shadow_used': 0, 'shadow': None, 'ctor': None}


def digests(protein_states):
    """Everything the trunk is fed, not just the letters. A reconstruction can match the
    sequence and differ in flags, residue_index or the template atoms, and those are exactly
    what a rebuilt state gets to choose."""
    out = {}
    for state, complex_ in protein_states.items():
        out[state] = {}
        for chain, protein in complex_.items():
            entry = {'len': int(len(protein)), 'letters': sequence_letters(protein)}
            for field in ('sequence', 'atoms', 'atom_mask', 'flags', 'residue_index'):
                a = np.asarray(getattr(protein, field))
                entry[field] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                                'sum': float(np.asarray(a, dtype=np.float64).sum()),
                                'min': float(np.asarray(a, dtype=np.float64).min()),
                                'max': float(np.asarray(a, dtype=np.float64).max())}
            out[state][chain] = entry
    return out


def read(predictions):
    return {state: scalar_metrics(getattr(prediction, 'metrics', None))
            for state, prediction in (predictions or {}).items()}


def saturated(metrics):
    for m in metrics.values():
        for key in ('ptm', 'iptm'):
            v = m.get(key)
            if isinstance(v, (int, float)) and v > args.saturated_at:
                return True
    return False


orig_init = T.TTBioAlphaFoldDesignModel.__init__


def init(self, *a, **kw):
    orig_init(self, *a, **kw)
    if STATE['ctor'] is None:
        STATE['ctor'] = (a, dict(kw))


T.TTBioAlphaFoldDesignModel.__init__ = init

orig_select = T.TTBioAlphaFoldDesignModel._select


def select(self, model):
    self._probe_model = model
    return orig_select(self, model)


T.TTBioAlphaFoldDesignModel._select = select

orig_grad = T.TTBioAlphaFoldDesignModel.sequence_gradients


def sequence_gradients(self, protein_states, losses, model=None, *a, **kw):
    STATE['taped'] += 1
    return orig_grad(self, protein_states, losses, model, *a, **kw)


T.TTBioAlphaFoldDesignModel.sequence_gradients = sequence_gradients

orig_predict = T.TTBioAlphaFoldDesignModel.predict


def shadow_for():
    """BindCraft 2's own trunk, same constructor, built once and only when needed."""
    if STATE['shadow'] is None:
        a, kw = STATE['ctor']
        kw = {**kw, 'trunk': 'jax', 'pool': None, 'seqlog': None}
        t0 = time.time()
        STATE['shadow'] = T.TTBioAlphaFoldDesignModel(*a, **kw)
        print(f'[probe] shadow jax trunk built in {time.time() - t0:.1f}s', flush=True)
    return STATE['shadow']


def predict(self, protein_states, model=None, *a, **kw):
    self._probe_model = None
    t0 = time.time()
    predictions = orig_predict(self, protein_states, model, *a, **kw)
    device_metrics = read(predictions)
    STATE['predicts'] += 1
    row = {'t': round(time.time(), 1), 'predict': STATE['predicts'],
           'taped_calls_before': STATE['taped'],
           'model': getattr(self, '_probe_model', None), 'trunk': self.trunk,
           'seconds': round(time.time() - t0, 1),
           'predict_args': [list(a), {k: (v if isinstance(v, (int, float, str)) else str(v))
                                      for k, v in kw.items()}],
           'device': device_metrics, 'states': digests(protein_states)}
    if saturated(device_metrics) and STATE['shadow_used'] < args.shadow_budget:
        STATE['shadow_used'] += 1
        s = shadow_for()
        s.dropout = self.dropout          # whatever the stage set it to, on both arms
        t1 = time.time()
        row['jax'] = read(s.predict(protein_states, row['model'], *a, **kw))
        row['jax_seconds'] = round(time.time() - t1, 1)
        row['SATURATED'] = True
        print('[probe] SATURATED device predict re-folded on JAX:', flush=True)
        print(json.dumps({'device': device_metrics, 'jax': row['jax']})[:2000], flush=True)
    with open(ROWS, 'a') as fh:
        fh.write(json.dumps(row, default=str) + '\n')
    print(json.dumps({k: row[k] for k in ('predict', 'taped_calls_before', 'model', 'seconds')}
                     | {'ptm': {s: m.get('ptm') for s, m in device_metrics.items()},
                        'iptm': {s: m.get('iptm') for s, m in device_metrics.items()}}), flush=True)
    return predictions


T.TTBioAlphaFoldDesignModel.predict = predict

campaign.AlphaFoldDesignModel = functools.partial(
    T.TTBioAlphaFoldDesignModel, trunk='device', pool=pool,
    seqlog=os.path.join(project, 'sequences.jsonl'))

import afgrad as _A
from splice import EvoformerOnDevice, evoformer_on_device
import stack as _S

_lv = None if args.no_levers else _S.Levers()
pool.use(pool.models[0])
if _lv is not None:
    _lv.arm('stack')
evo = EvoformerOnDevice(pool, k_evo=48)

mpnn = os.path.join(B.BC2, 'bindcraft', 'weights', 'proteinmpnn', 'weights_neutral')
stamp = {'stage_steps': args.stage_steps, 'overrides': overrides, 'binder_length': args.binder_length,
         'bucket': args.bucket, 'pool_resident': args.pool_resident,
         'levers': not args.no_levers, 'started_utc': time.strftime('%FT%TZ', time.gmtime())}
t0 = time.time()
with evoformer_on_device(evo):
    count = campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=mpnn, max_trajectories=1)
stamp.update({'trajectories_run': count, 'wall_seconds': round(time.time() - t0, 1),
              'device_calls': dict(evo.calls), 'predicts': STATE['predicts'],
              'taped': STATE['taped'], 'shadow_used': STATE['shadow_used'],
              'mask_shape_collisions': getattr(evo, 'mask_shape_collisions', 'n/a'),
              'pool': pool.stamp(), 'stamp': A.stamp(args.card)})
(pathlib.Path(project) / 'probe_stamp.json').write_text(json.dumps(stamp, indent=1, default=str))
print(json.dumps(stamp, indent=1, default=str), flush=True)
