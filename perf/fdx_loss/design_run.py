"""The smallest real BindCraft 2 gradient-design run, so two processes can be compared bit-for-bit.

BindCraft 2's own JAX trunk, on the CPU, with the real AF2 weights: 32-residue target plus a
16-residue designed binder, which BC2's 32-residue length bucket pads to 64 tokens. Each step is
one `AlphaFoldDesignModel.sequence_gradients` call (`bindcraft/af2.py`) followed by one real
optimiser step from `LogitSequenceOptimizer` -- the same pair `run_gradient_design_stage` runs.

    PYTHONPATH=<worktree>:/home/moritz/bcx_shipped/bc2 \
      /home/moritz/bcx_hostcut_venv/bin/python3 design_run.py --steps 2 --seed 0

Never opens a Tenstorrent device: it builds `bindcraft.af2.AlphaFoldDesignModel` directly.

Everything that could vary between two processes is pinned: the only randomness is `--seed`, fed to
the model key and to `Protein.empty`'s binder initialisation, and `dropout=False` removes the
remaining per-step draw (BC2's own `harden` stage runs that way). The target is a literal sequence,
not a library name, so no path, no crop sampling and no structure parse enters the result.
"""
import argparse
import contextlib
import hashlib
import importlib
import json
import sys
import time

import jax
import numpy

from bindcraft.af2 import AlphaFoldDesignModel, pad_design_chains
from bindcraft.loss import build_design_losses
from bindcraft.protein import ATOM_INDEX, Protein
from bindcraft.prediction import collect_shared_chains
from bindcraft.sequence_optimization import LogitSequenceOptimizer
from bindcraft.target_schedule import losses_for_active_states
from bindcraft.trajectory import transfer_binder_sequences

AF2_DATA_DIR = '/home/moritz/.boltz/af2'
TARGET_STATE = 'target_state'
TARGET_CHAIN = f'target_{TARGET_STATE}'
BINDER_CHAIN = 'binder'
# A 32-residue stretch, short enough that 32-residue binder bucket + target = 64 tokens.
TARGET_SEQUENCE = 'SYDLLDNHLAKQMFSGLSYEEIQKLVGRRGEN'
BINDER_LENGTH = 16
LENGTH_BUCKET = 32
# Tryptophan, phenylalanine, tyrosine: what the third A/B arm's added term asks for.
AROMATIC_ACIDS = 'WFY'


def canonical_digest(array) -> str:
    """sha256 over float32 bytes of a C-contiguous copy.

    float32 and not the native dtype because the two arrays hashed here arrive in different native
    widths -- AF2 positions come back as float16 (`af2.py` casts `final_atom_positions`), the binder
    logits come back in whatever width the optimiser left them -- and a digest is only comparable
    between two processes, and between the two arms of a loss A/B, if the byte width is fixed by
    this file rather than by the code path. float32 holds every float16 and bfloat16 value exactly,
    so the cast is lossless: it cannot hide a difference, it only stops a dtype change from faking
    one. `ascontiguousarray` because `.tobytes()` of a strided view would otherwise reorder bytes.
    """
    contiguous = numpy.ascontiguousarray(numpy.asarray(array, dtype=numpy.float32))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def binder_radius(protein) -> float:
    """The binder's CA radius of gyration in Angstrom, float64 in numpy.

    Here rather than in the A/B arm so the trace is recorded whatever loss is installed, and the
    control arm's curve is measured by the same code as the effect arm's.
    """
    points = numpy.asarray(protein.atoms[:, ATOM_INDEX['CA']], dtype=numpy.float64)
    return float(numpy.sqrt(numpy.square(points - points.mean(0)).sum(-1).mean()))


def binder_resolved(prediction, protein_complex) -> float:
    """The binder's mean `experimentally_resolved_ca`, in numpy.

    Recorded for every arm by this one function, so the control's curve and the effect arm's are
    measured by the same code and neither reads back its own objective.
    """
    from bindcraft.loss import chain_residue_slices

    rows = chain_residue_slices(protein_complex)[BINDER_CHAIN]
    values = numpy.asarray(prediction.metrics['experimentally_resolved_ca'], dtype=numpy.float64)
    return float(values[rows].mean())


def binder_aromatic(protein_states) -> float:
    """The binder's mean designed-residue probability of W, F or Y, in numpy.

    The quantity the third A/B arm targets. Read off the design sequence logits rather than off
    anything AF2 returned, and recomputed here for every arm by this one function, so the control
    curve and the effect curve are measured by the same code and neither arm reads back its own
    objective.
    """
    from bindcraft.protein import AMINO_ACID_INDEX, ResidueFlags, has_residue_flag

    binder = protein_states[TARGET_STATE][BINDER_CHAIN]
    logits = numpy.asarray(binder.sequence, dtype=numpy.float64)
    shifted = logits - logits.max(-1, keepdims=True)
    probabilities = numpy.exp(shifted)
    probabilities /= probabilities.sum(-1, keepdims=True)
    columns = [AMINO_ACID_INDEX[amino_acid] for amino_acid in AROMATIC_ACIDS]
    designed = numpy.asarray(has_residue_flag(binder.flags, ResidueFlags.DESIGN), dtype=bool)
    return float(probabilities[designed][:, columns].sum(-1).mean())


def padded_token_count(protein_states) -> int:
    padded = pad_design_chains(protein_states, LENGTH_BUCKET, 0)
    return sum(len(protein) for protein_complex in padded.values() for protein in protein_complex.values())


def loss_hook(module_name, attribute_name):
    if not module_name or not attribute_name:
        return contextlib.nullcontext()
    return getattr(importlib.import_module(module_name), attribute_name)()


def run_design(steps: int, seed: int, hook) -> dict:
    model_key, binder_key = jax.random.split(jax.random.PRNGKey(seed))
    model = AlphaFoldDesignModel(presets='model_1_ptm', data_dir=AF2_DATA_DIR, key=model_key,
                                 num_recycle=1, length_bucket_size=LENGTH_BUCKET, dropout=False)
    protein_states = {TARGET_STATE: {BINDER_CHAIN: Protein.empty(BINDER_LENGTH, binder_key),
                                     TARGET_CHAIN: Protein.from_fasta(TARGET_SEQUENCE)}}
    optimizer = LogitSequenceOptimizer(iterations=steps)
    design_loss, prediction, trace = None, None, []
    with hook:
        # INSIDE the hook, and that is the whole point: a custom loss is installed by patching
        # `bindcraft.loss.build_losses`, which `build_design_losses` reads. Building the dict
        # before entering the hook runs the campaign's default terms under a name that claims
        # otherwise, so the A/B arm would come out identical and read as "no effect".
        losses = losses_for_active_states(
            build_design_losses({}, {TARGET_STATE: 1.0}, LENGTH_BUCKET, seed), protein_states)
        print('terms ' + ','.join(f'{n}={e.weight:+.3f}' for n, e in sorted(losses.items())),
              file=sys.stderr, flush=True)
        for step in range(steps):
            started = time.perf_counter()
            softmax_weight, one_hot_weight, temperature, logit_scale = optimizer.sequence_parameters()
            # model= is named so the call never draws from (and so never advances) the model key.
            predictions, chain_gradients, design_loss = model.sequence_gradients(
                protein_states, losses, model='model_1_ptm', softmax_weight=softmax_weight,
                one_hot_weight=one_hot_weight, temperature=temperature, logit_scale=logit_scale)
            prediction = predictions[TARGET_STATE]
            updated = optimizer.update_sequence(protein_states, {name: [gradient] for name, gradient in chain_gradients.items()})
            protein_states = transfer_binder_sequences(protein_states, updated)
            # An observation, not a feedback: computed in numpy off the prediction, consumes no
            # randomness and enters no graph, so it cannot move the digest.
            radius = binder_radius(prediction.protein_complex[BINDER_CHAIN])
            resolved = binder_resolved(prediction, protein_states[TARGET_STATE])
            aromatic = binder_aromatic(protein_states)
            trace.append({'step': step + 1, 'loss': float(design_loss), 'binder_rg': radius,
                          'binder_resolved': resolved, 'binder_aromatic': aromatic})
            print(f'step {step + 1}/{steps} loss={float(design_loss):.6f} binder_rg={radius:.4f} '
                  f'binder_resolved={resolved:.6f} binder_aromatic={aromatic:.6f} '
                  f'{time.perf_counter() - started:.1f}s',
                  file=sys.stderr, flush=True)
    binder = prediction.protein_complex[BINDER_CHAIN]
    return {'tokens': padded_token_count(protein_states),
            'steps': steps,
            'final_loss': float(design_loss),
            # The loss names only: `prediction.metrics` also carries plddt/ptm/iptm, which are not terms.
            'terms': {name: float(prediction.metrics[name]) for name in losses if name in prediction.metrics},
            'trace': trace,
            'weights': {name: float(entry.weight) for name, entry in sorted(losses.items())},
            'coord_sha256': canonical_digest(binder.atoms[:, ATOM_INDEX['CA']]),
            # The CA coordinates themselves, so an A/B can measure a geometric quantity on the
            # output instead of reading back the objective that was optimised.
            'binder_ca': numpy.asarray(binder.atoms[:, ATOM_INDEX['CA']], dtype=float).tolist(),
            # The logits after the last optimiser step: the run's design output, not its input.
            'seq_sha256': canonical_digest(collect_shared_chains(protein_states)[1][BINDER_CHAIN].sequence)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loss-module', default='')
    parser.add_argument('--loss-attr', default='')
    parser.add_argument('--json', dest='json_path', default='')
    arguments = parser.parse_args()
    if arguments.steps < 1:
        parser.error('--steps must be at least 1')
    result = run_design(arguments.steps, arguments.seed, loss_hook(arguments.loss_module, arguments.loss_attr))
    print(f'TOKENS {result["tokens"]}')
    print(f'STEPS {result["steps"]}')
    print(f'FINAL_LOSS {result["final_loss"]:.10f}')
    print('WEIGHTS ' + ','.join(f'{n}={w:+.3f}' for n, w in sorted(result['weights'].items())))
    print('TERMS ' + ','.join(f'{name}={value:.6f}' for name, value in sorted(result['terms'].items())))
    print(f'COORD_SHA256 {result["coord_sha256"]}')
    print(f'SEQ_SHA256 {result["seq_sha256"]}')
    if arguments.json_path:
        with open(arguments.json_path, 'w') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
