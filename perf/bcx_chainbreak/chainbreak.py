"""Do BindCraft 2's two AlphaFold2 entry points see the same molecule?

Pure upstream BindCraft 2, CPU only. No structure file: both chains are built from sequence, so a
clean checkout plus the monomer AlphaFold2 parameters is everything this needs.

`bindcraft/af2.py:305-306` renumbers a multi-chain complex before a monomer model sees it. The
gradient path at `af2.py:337-345` does not. This script measures what that costs on the metrics
BindCraft 2's own filters read.

One leg per process, because both entry points memoise their traced function and a patch installed
after a trace cannot move the answer.

  python chainbreak.py census                       # arithmetic only, no model, no parameters
  python chainbreak.py run --leg predict  --binder 32 --target 96 --params DIR
  python chainbreak.py run --leg gradient --binder 32 --target 96 --params DIR
"""
import argparse, json, os, platform, subprocess, sys, time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def deterministic_sequence(length, salt):
    """A fixed pseudo-random sequence. Content is irrelevant here; only that every leg gets the same one."""
    state, letters = 1234567 + 7919 * salt, []
    for _ in range(length):
        state = (1103515245 * state + 12345) % (1 << 31)
        letters.append(AMINO_ACIDS[state % 20])
    return ''.join(letters)

def residue_indices(binder_length, target_length, target_start):
    binder = list(range(1, binder_length + 1))
    target = list(range(target_start, target_start + target_length))
    return binder, target

def window_census(binder_length, target_length, target_start, window=32):
    """Cross-chain pairs the monomer relpos window admits, from the inputs alone."""
    binder, target = residue_indices(binder_length, target_length, target_start)
    inside = sum(1 for i in binder for j in target if abs(j - i) <= window)
    at_zero = sum(1 for i in binder for j in target if j == i)
    total = binder_length * target_length
    return {'binder_length': binder_length, 'target_length': target_length, 'target_start': target_start,
            'junction_step': target[0] - binder[-1], 'cross_chain_pairs': total,
            'in_window_pairs': inside, 'in_window_share': inside / total,
            'pairs_at_offset_zero': at_zero}

def census(args):
    rows = []
    for binder_length, target_length, target_start in args.arms:
        row = window_census(binder_length, target_length, target_start)
        broken = window_census(binder_length, target_length, target_start)
        # after monomer_chain_break_indices the junction step is MONOMER_CHAIN_GAP + 1 = 50, so the
        # target is renumbered to start 50 past the binder's last residue
        row['after_chain_break'] = window_census(binder_length, target_length, binder_length + 50)
        del broken
        rows.append(row)
    print(json.dumps(rows, indent=2))
    return rows

def build_states(binder_length, target_length, target_start, chains, seed=0):
    import jax, jax.numpy as jnp
    from bindcraft.protein import Protein, ResidueFlags

    def designed(length, salt):
        protein = Protein.from_fasta('>A\n' + deterministic_sequence(length, salt))
        return protein.replace(flags=jnp.full((length,), int(ResidueFlags.DESIGN), dtype=jnp.uint8))

    if chains == 1:
        return {'complex': {'binder': designed(binder_length + target_length, seed)}}
    target = Protein.from_fasta('>B\n' + deterministic_sequence(target_length, 1))
    target = target.replace(residue_index=jnp.arange(target_start, target_start + target_length, dtype=jnp.int32))
    return {'complex': {'binder': designed(binder_length, seed), 'target': target}}

def scalar_metrics(prediction, binder_length):
    import numpy as np
    metrics = prediction.metrics
    plddt = np.asarray(metrics['plddt'], dtype=np.float64)
    out = {'ptm': float(metrics['ptm']), 'plddt': float(plddt.mean()),
           'binder_plddt': float(plddt[:binder_length].mean())}
    if 'iptm' in metrics:
        out['iptm'] = float(metrics['iptm'])
    return out

def run(args):
    import jax, jax.numpy as jnp
    import bindcraft.af2 as af2
    from bindcraft.af2 import AlphaFoldDesignModel
    from bindcraft.loss import DesignLoss, iptm_loss, plddt_loss

    if args.leg == 'predict_identity':
        af2.monomer_chain_break_indices = lambda chain_lengths, residue_index: residue_index

    states = build_states(args.binder, args.target, args.target_start, args.chains, args.seed)
    binder_length = args.binder + args.target if args.chains == 1 else args.binder
    model = AlphaFoldDesignModel(presets='model_1_ptm', data_dir=args.params, num_recycle=args.recycle, dropout=False)
    sequence_parameters = dict(softmax_weight=1.0, one_hot_weight=1.0, temperature=0.01, logit_scale=2.0)

    started = time.time()
    if args.leg == 'gradient':
        loss_function = iptm_loss if args.chains > 1 else plddt_loss
        losses = {'objective': DesignLoss(function=loss_function, weight=1.0, required_states=frozenset({'complex'}),
                                          interface_mask=None)}
        if args.chains == 1:
            losses = {'objective': DesignLoss(function=lambda s, p: plddt_loss(s, p, prediction_state='complex', chain='binder'),
                                              weight=1.0, required_states=frozenset({'complex'}), interface_mask=None)}
        predictions, _, design_loss = model.sequence_gradients(states, losses, **sequence_parameters)
        result = scalar_metrics(predictions['complex'], binder_length)
        result['design_loss'] = float(design_loss)
    else:
        predictions = model.predict(states, **sequence_parameters)
        result = scalar_metrics(predictions['complex'], binder_length)

    result.update({'leg': args.leg, 'binder': args.binder, 'target': args.target,
                   'target_start': args.target_start, 'chains': args.chains, 'recycle': args.recycle, 'seed': args.seed,
                   'seconds': round(time.time() - started, 2), 'host': platform.node(),
                   'commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(af2.__file__),
                                            capture_output=True, text=True).stdout.strip(),
                   'jax': jax.__version__, 'loadavg': os.getloadavg()[0],
                   'threads': os.environ.get('XLA_FLAGS', '')})
    print(json.dumps(result))
    if args.out:
        with open(args.out, 'w') as handle:
            json.dump(result, handle, indent=2)
    return result

def arm(text):
    binder_length, target_length, target_start = (int(value) for value in text.split(','))
    return binder_length, target_length, target_start

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    census_parser = sub.add_parser('census')
    census_parser.add_argument('--arms', type=arm, nargs='+',
                               default=[(32, 96, 1), (32, 160, 1), (32, 288, 1), (64, 192, 1), (32, 96, 4000)])
    census_parser.set_defaults(function=census)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--leg', choices=('predict', 'predict_identity', 'gradient'), required=True)
    run_parser.add_argument('--binder', type=int, default=32)
    run_parser.add_argument('--target', type=int, default=96)
    run_parser.add_argument('--target-start', type=int, default=1)
    run_parser.add_argument('--chains', type=int, default=2, choices=(1, 2))
    run_parser.add_argument('--recycle', type=int, default=1)
    run_parser.add_argument('--seed', type=int, default=0, help='picks the binder sequence; the target is held fixed')
    run_parser.add_argument('--params', default=os.environ.get('BC2_PARAMS', ''))
    run_parser.add_argument('--out', default='')
    run_parser.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)

if __name__ == '__main__':
    main()
