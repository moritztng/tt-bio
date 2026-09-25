#!/usr/bin/env python3
"""Fold the sequence the mutate stage actually saw, on card and on BindCraft 2's own trunk.

Every device-vs-JAX `predict` this row has taken was on a state whose binder logits are
`Protein.empty`'s `0.01 * normal` -- an arbitrary sequence. The mutate stage folds a
sequence 125 gradient steps optimised against the card, and that is where ptm and iptm come
back saturated while pLDDT collapses. The sequence was never on disk until `298f59976`
recorded it; `pool_ckpt2/sequences.jsonl` has it, from the trajectory that ran with the
card/JAX checkpoint match fixed.

So: same state, same checkpoint, same process, binder sequence swapped for the recorded one,
device against BindCraft 2's JAX. An arbitrary-sequence control rides along in the same
process, so a difference between the two rows is the sequence and nothing else.

If both arms saturate, the trunk is exonerated and the finding is about what our gradient
produced. If only the card saturates, the primal device path is wrong on an optimised
sequence and right on an arbitrary one.
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
from ttbio_predictor import TTBioAlphaFoldDesignModel, sequence_letters
from bindcraft.protein import AMINO_ACIDS

ap = argparse.ArgumentParser()
ap.add_argument('--params', default='/home/ttuser/bcx_e2e/af2_params')
ap.add_argument('--seqlog', default='/home/ttuser/bcx_multimer_art/pool_ckpt2/sequences.jsonl')
ap.add_argument('--line', type=int, default=-1, help='which recorded predict to replay')
ap.add_argument('--models', default='model_1_multimer_v3')
ap.add_argument('--bucket', type=int, default=32)
ap.add_argument('--card', type=int, default=2)
ap.add_argument('--binder-length', type=int, default=146)
ap.add_argument('--json', default=str(HERE / 'parity_artifacts' / 'mutate_sequence_probe.json'))
args = ap.parse_args()

MODELS = tuple(args.models.split(','))
INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}

records = [json.loads(line) for line in open(args.seqlog) if line.strip()]
record = records[args.line]
recorded = record['states']['hPDL1']

settings = B.campaign_settings(overrides=[f'length_bucket_size={args.bucket}', 'campaign_seed=0',
                                          f'binder_lengths=[{args.binder_length}]'])
_ds, states, _losses = B.design_state(settings)
shape = B.state_shape(states)
n = sum(sum(c.values()) for c in shape.values())


def one_hot(letters):
    out = np.zeros((len(letters), len(AMINO_ACIDS)), dtype=np.float32)
    out[np.arange(len(letters)), [INDEX[a] for a in letters]] = 1.0
    return out


def with_recorded_binder(protein_states):
    out = {}
    for state_name, complex_ in protein_states.items():
        chains = {}
        for chain, protein in complex_.items():
            letters = recorded.get(chain)
            if letters is not None and len(letters) == len(protein):
                chains[chain] = protein.replace(sequence=one_hot(letters))
            else:
                chains[chain] = protein
        out[state_name] = chains
    return out


hallucinated = with_recorded_binder(states)

# the swap has to be exactly what was recorded, read back the way a filter reads it
checks = {chain: {'recorded': recorded.get(chain), 'replayed': sequence_letters(protein),
                  'match': recorded.get(chain) == sequence_letters(protein)}
          for chain, protein in hallucinated['hPDL1'].items()}
for chain, c in checks.items():
    if c['recorded'] is not None and not c['match']:
        raise SystemExit(f'replayed {chain} is not the recorded sequence')
print(json.dumps({'n_residues': n, 'shape': shape, 'replayed_line': args.line,
                  'recorded_model': record.get('model'),
                  'sequence_roundtrip': {k: v['match'] for k, v in checks.items()}}), flush=True)


def read(pred, arm, kind, model, secs):
    m = pred['hPDL1'].metrics
    plddt = np.asarray(m['plddt'], dtype=np.float64)
    pae = np.asarray(m['pae'], dtype=np.float64)
    return {'arm': arm, 'sequence': kind, 'model': model, 'n_residues': n,
            'seconds': round(secs, 1),
            'ptm': float(m['ptm']), 'iptm': float(m['iptm']),
            'plddt_mean': float(plddt.mean()), 'plddt_min': float(plddt.min()),
            'plddt_max': float(plddt.max()), 'plddt_n': int(plddt.size),
            'pae_mean_A': float(pae.mean()), 'pae_min_A': float(pae.min()),
            'pae_max_A': float(pae.max())}


def build(trunk):
    return TTBioAlphaFoldDesignModel(presets=MODELS, data_dir=args.params, models=MODELS,
                                     num_recycle=1, length_bucket_size=args.bucket,
                                     max_cache_size=6, trunk=trunk, pool=None)


ARMS = (('hallucinated', hallucinated), ('arbitrary', states))
rows = []

pool = MP.MultimerPool(args.params, resident=1, log_path=None)
dev = build('device')
dev.dropout = False                      # trajectory.py:262, entering the mutate stage
for model in MODELS:
    pool.use(model)
    evo = EvoformerOnDevice(pool, k_evo=48)
    with evoformer_on_device(evo):
        for kind, s in ARMS:
            t0 = time.time()
            row = read(dev.predict(s, model=model), 'device', kind, model, time.time() - t0)
            rows.append(row); print(json.dumps(row), flush=True)

ref = build('jax')
ref.dropout = False
for model in MODELS:
    for kind, s in ARMS:
        t0 = time.time()
        row = read(ref.predict(s, model=model), 'jax', kind, model, time.time() - t0)
        rows.append(row); print(json.dumps(row), flush=True)

by = {(r['arm'], r['sequence'], r['model']): r for r in rows}
delta = {}
for model in MODELS:
    for kind, _ in ARMS:
        d, j = by[('device', kind, model)], by[('jax', kind, model)]
        delta[f'{kind}/{model}'] = {k: round(d[k] - j[k], 6)
                                    for k in ('ptm', 'iptm', 'plddt_mean', 'pae_mean_A')}

out = {'replayed_line': args.line, 'recorded_model': record.get('model'),
       'binder_length': args.binder_length, 'n_residues': n, 'bucket': args.bucket,
       'models': list(MODELS), 'dropout': False, 'rows': rows,
       'device_minus_jax': delta, 'sequence_roundtrip': checks,
       'stamp': A.stamp(args.card)}
p = pathlib.Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({'device_minus_jax': delta}, indent=1), flush=True)
