#!/usr/bin/env python3
"""Does a primal `predict` change after the same process has run the taped backward?

The mutate stage is the only place in a trajectory where BindCraft 2 goes back to a plain
`predict` after 125 gradient steps (`bindcraft/trajectory.py:270`), and it is the only
stage the card gets wrong: screen 0.79 / refine 0.85 / anneal 0.87 / harden 0.87 on i_pTM,
then mutate at 1.0 with pLDDT collapsing 0.88 -> 0.6, against 0.74 / 0.78 on BindCraft 2's
own JAX.

Everything else that is different about that stage has been measured and cleared:

  * dropout, flipped in one process, device against JAX -- `mutate_dropout_probe.py`,
    0.5873 against 0.5884 ptm.
  * the size, device against JAX at the n the loop runs -- `predict_parity_size.py`,
    0.4468 against 0.4462 ptm at n=288.
  * the checkpoint mismatch fixed in 1127f9ea8, which moves ptm by 0.012 on an
    unoptimised sequence.

What none of them did is run the tape first. Each of those processes did two forward
passes and stopped, and this stack is known to carry process-global state across a taped
call: one taped call once switched fused triangle attention off for the whole process.

So: the SAME state, predicted twice in one process with k taped gradient steps in
between, and nothing else changed. The sequence is never updated, so rows 1 and 2 differ
only in what the process did between them. The JAX arm runs the identical protocol and is
the control -- if it moves too, the mover is BindCraft 2's own key or dropout state and
not the card.
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
ap.add_argument('--steps', type=int, default=3,
                help='taped gradient steps between the two predicts. The trajectory runs '
                     '125; the question is whether the effect exists at all, so a handful '
                     'that shows a move is an answer and a handful that does not bounds it.')
ap.add_argument('--arms', default='device,jax')
ap.add_argument('--no-levers', action='store_true',
                help="the trajectory runs with them armed, so the probe does too; "
                     "levers-off is ~400 s a taped step at this size against ~70 s on")
ap.add_argument('--json', default=str(HERE / 'parity_artifacts' / 'used_process_predict.json'))
args = ap.parse_args()

M1 = 'model_1_multimer_v3'
settings = B.campaign_settings(overrides=[f'length_bucket_size={args.bucket}', 'campaign_seed=0'])
_ds, states, losses = B.design_state(settings)
n = sum(sum(c.values()) for c in B.state_shape(states).values())


def build(trunk):
    return TTBioAlphaFoldDesignModel(presets=(M1,), data_dir=args.params, models=(M1,),
                                     num_recycle=1, length_bucket_size=args.bucket,
                                     max_cache_size=4, trunk=trunk, pool=None)


def read(pred, arm, when, secs, steps_before):
    m = pred['hPDL1'].metrics
    return {'arm': arm, 'when': when, 'taped_steps_before': steps_before,
            'seconds': round(secs, 1),
            'ptm': round(float(m['ptm']), 6), 'iptm': round(float(m['iptm']), 6),
            'plddt_mean': round(float(np.asarray(m['plddt']).mean()), 6),
            'plddt_min': round(float(np.asarray(m['plddt']).min()), 6),
            'pae_mean_A': round(float(np.asarray(m['pae']).mean()), 4)}


def protocol(model, arm, rows):
    """fresh predict -> k taped gradient steps -> the same predict again."""
    model.dropout = False                      # what run_mutation_polish sets
    t0 = time.time()
    rows.append(read(model.predict(states, model=M1), arm, 'fresh', time.time() - t0, 0))
    print(json.dumps(rows[-1]), flush=True)

    model.dropout = True                       # what the gradient stages run with
    losses_seen = []
    for step in range(args.steps):
        t0 = time.time()
        _preds, _grads, loss = model.sequence_gradients(
            states, losses, model=M1, softmax_weight=1.0, one_hot_weight=0.0,
            temperature=1.0, logit_scale=2.0)
        losses_seen.append(round(float(loss), 5))
        print(f'  {arm} taped step {step + 1}/{args.steps} loss={losses_seen[-1]} '
              f'{time.time() - t0:.1f}s', flush=True)

    model.dropout = False
    t0 = time.time()
    rows.append(read(model.predict(states, model=M1), arm, 'used', time.time() - t0,
                     args.steps))
    print(json.dumps(rows[-1]), flush=True)
    return losses_seen


out = {'n_residues': n, 'bucket': args.bucket, 'checkpoint': M1, 'taped_steps': args.steps,
       'levers': not args.no_levers, 'rows': [], 'losses': {}}
arms = args.arms.split(',')

if 'device' in arms:
    pool = MP.MultimerPool(args.params, resident=1, log_path=None)
    pool.use(M1)
    if not args.no_levers:
        import stack as _S
        _S.Levers().arm("stack")
    evo = EvoformerOnDevice(pool, k_evo=48)
    with evoformer_on_device(evo):
        out['losses']['device'] = protocol(build('device'), 'device', out['rows'])
    out['device_calls'] = dict(evo.calls)

if 'jax' in arms:
    out['losses']['jax'] = protocol(build('jax'), 'jax', out['rows'])


def delta(arm, field):
    got = [r[field] for r in out['rows'] if r['arm'] == arm]
    return round(got[1] - got[0], 6) if len(got) == 2 else None


out['used_minus_fresh'] = {arm: {f: delta(arm, f) for f in
                                 ('ptm', 'iptm', 'plddt_mean', 'pae_mean_A')}
                           for arm in arms}
out['stamp'] = A.stamp(args.card)
p = pathlib.Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != 'stamp'}, indent=1, default=str))
