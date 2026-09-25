#!/usr/bin/env python3
"""When does the primal `predict` go bad, counted in taped steps?

`mutate_sequence_probe.py` settles what the mutate stage's numbers are NOT. Folding the
sequence that stage actually saw, in a fresh process at the size it runs, gives device
ptm 0.8386 / iptm 0.8111 / pLDDT 0.8571 against BindCraft 2's own JAX at 0.8404 / 0.8146 /
0.8632 -- and harden round 5, one stage earlier in the trajectory itself, logged 0.84 / 0.81.
The trajectory's mutate stage logged ptm = iptm = 1.0 and pLDDT 0.38 for that same sequence
from its very first round, before a single mutation was proposed.

So the trunk, the sequence, the size, the checkpoint and dropout are all cleared, and what
is left is the process: 125 taped gradient steps run before that predict.
`used_process_predict.py` bounded the effect at 3 steps and n=192 and found exactly zero on
every metric. This runs the same protocol where the trajectory runs it -- n=288, the
hallucinated sequence -- and predicts after EVERY taped step, so one run gives the onset
rather than a yes/no at one k.

The state is never updated, so every row folds identical input and the only thing that
changes between rows is how many taped steps the process has behind it.
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
ap.add_argument('--line', type=int, default=-1)
ap.add_argument('--bucket', type=int, default=32)
ap.add_argument('--card', type=int, default=2)
ap.add_argument('--binder-length', type=int, default=146)
ap.add_argument('--steps', type=int, default=12)
ap.add_argument('--rotate', action='store_true',
                help='cycle the five checkpoints on the taped steps the way the shipped pool '
                     'does, while the predict stays pinned to model_1 so the rows compare')
ap.add_argument('--no-levers', action='store_true')
ap.add_argument('--json', default=str(HERE / 'parity_artifacts' / 'mutate_onset_probe.json'))
args = ap.parse_args()

M1 = 'model_1_multimer_v3'
POOL = tuple(f'model_{i}_multimer_v3' for i in range(1, 6))
MODELS = POOL if args.rotate else (M1,)
INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}

record = [json.loads(l) for l in open(args.seqlog) if l.strip()][args.line]
recorded = record['states']['hPDL1']

settings = B.campaign_settings(overrides=[f'length_bucket_size={args.bucket}', 'campaign_seed=0',
                                          f'binder_lengths=[{args.binder_length}]'])
_ds, states, losses = B.design_state(settings)
n = sum(sum(c.values()) for c in B.state_shape(states).values())


def one_hot(letters):
    out = np.zeros((len(letters), len(AMINO_ACIDS)), dtype=np.float32)
    out[np.arange(len(letters)), [INDEX[a] for a in letters]] = 1.0
    return out


states = {s: {c: (p.replace(sequence=one_hot(recorded[c]))
                  if recorded.get(c) and len(recorded[c]) == len(p) else p)
              for c, p in complex_.items()} for s, complex_ in states.items()}
for c, p in states['hPDL1'].items():
    if recorded.get(c) and sequence_letters(p) != recorded[c]:
        raise SystemExit(f'replayed {c} is not the recorded sequence')

out = {'n_residues': n, 'bucket': args.bucket, 'binder_length': args.binder_length,
       'predict_checkpoint': M1, 'taped_checkpoints': list(MODELS), 'rotate': args.rotate,
       'levers': not args.no_levers, 'sequence': 'hallucinated', 'replayed_line': args.line,
       'steps_requested': args.steps, 'rows': [], 'taped_losses': []}
path = pathlib.Path(args.json); path.parent.mkdir(parents=True, exist_ok=True)


def flush():
    out['stamp'] = A.stamp(args.card)
    path.write_text(json.dumps(out, indent=1, default=str))


def read(pred, steps_before, secs):
    m = pred['hPDL1'].metrics
    plddt = np.asarray(m['plddt'], dtype=np.float64)
    pae = np.asarray(m['pae'], dtype=np.float64)
    return {'taped_steps_before': steps_before, 'seconds': round(secs, 1),
            'ptm': float(m['ptm']), 'iptm': float(m['iptm']),
            'plddt_mean': float(plddt.mean()), 'plddt_min': float(plddt.min()),
            'pae_mean_A': float(pae.mean())}


pool = MP.MultimerPool(args.params, resident=1, log_path=None)
pool.use(M1)
if not args.no_levers:
    import stack as _S
    _S.Levers().arm("stack")
evo = EvoformerOnDevice(pool, k_evo=48)

with evoformer_on_device(evo):
    model = TTBioAlphaFoldDesignModel(presets=MODELS, data_dir=args.params, models=MODELS,
                                      num_recycle=1, length_bucket_size=args.bucket,
                                      max_cache_size=6, trunk='device', pool=None)
    for step in range(args.steps + 1):
        if step:
            taped = MODELS[(step - 1) % len(MODELS)]
            pool.use(taped)
            model.dropout = True                    # what the gradient stages run with
            t0 = time.time()
            _p, _g, loss = model.sequence_gradients(states, losses, model=taped,
                                                    softmax_weight=1.0, one_hot_weight=0.0,
                                                    temperature=1.0, logit_scale=2.0)
            out['taped_losses'].append({'step': step, 'model': taped,
                                        'loss': round(float(loss), 5),
                                        'seconds': round(time.time() - t0, 1)})
            print(json.dumps(out['taped_losses'][-1]), flush=True)
        pool.use(M1)
        model.dropout = False                       # what run_mutation_polish sets
        t0 = time.time()
        row = read(model.predict(states, model=M1), step, time.time() - t0)
        out['rows'].append(row); print(json.dumps(row), flush=True)
        flush()
        if row['ptm'] > 0.99 or row['iptm'] > 0.99:
            out['onset'] = {'taped_steps': step, 'row': row}
            print(f'SATURATED after {step} taped steps', flush=True)
            break

out['device_calls'] = dict(evo.calls)
base = out['rows'][0]
out['drift_from_fresh'] = [{'taped_steps': r['taped_steps_before'],
                            **{k: round(r[k] - base[k], 6)
                               for k in ('ptm', 'iptm', 'plddt_mean', 'pae_mean_A')}}
                           for r in out['rows']]
flush()
print(json.dumps({'drift_from_fresh': out['drift_from_fresh'],
                  'onset': out.get('onset', 'none within steps')}, indent=1), flush=True)
