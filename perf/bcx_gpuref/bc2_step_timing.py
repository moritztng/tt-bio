"""Time BindCraft 2's gradient step on a GPU, with compile and compute measured separately.

Run it exactly as you would run BindCraft, with `bindcraft` replaced by this file:

    python bc2_step_timing.py --jsonl steps.jsonl -- design examples/pdl1.json --core benchmark \
        --set 'binder_lengths=[96]' --set 'project_folder=results/pdl1'

One gradient step in BindCraft 2 is one call to `AlphaFoldDesignModel.sequence_gradients`
(`bindcraft/trajectory.py:131`). With `design_recycles: 1` that call is, from BC2's own code:

  * `bindcraft/af2.py:136-143` `recycled_alphafold_outputs` splits the key into `num_recycle + 1`,
    runs `alphafold_runner.apply` once per key bar the last under `jax.lax.stop_gradient`, and
    returns the un-stopped `apply` on the last key -- so one stop-gradient forward and one
    differentiated forward;
  * `bindcraft/af2.py:376` wraps that in `jax.value_and_grad`, so one backward over the second
    forward only.

Three passes, which is the same unit tt-bio's 16.554 s is the sum of.

The split between compile and compute is measured, not estimated. `sequence_gradients` asks
`_compiled_sequence_gradients` for a jitted function and then calls `.lower(...).compile()` on it
before every invocation (`af2.py:400-406`). This file wraps that jitted function so the
`lower().compile()` path and the call itself are timed on their own clocks: no per-iteration rate
is ever extrapolated over a stage to price compilation by subtraction.

Every timing blocks on the result. A JAX call is asynchronous, so an unblocked timing measures
dispatch and charges the real work to whatever line reads the array next.
"""
import argparse
import json
import os
import sys
import threading
import time

RECORD = threading.local()


def _current():
    return getattr(RECORD, 'record', None)


class TimedCompiled:
    """Wraps the jitted gradient function so compile and execution are timed apart.

    Both `lower(...).compile()` and `__call__` are on the hot path of every step: the first is a
    cache hit after the shape has been seen once, the second is the step. Timing them separately
    is what makes "compile excluded" a measurement rather than a subtraction.
    """

    def __init__(self, inner, block_until_ready):
        self._inner = inner
        self._block = block_until_ready

    def lower(self, *arguments, **keywords):
        started = time.perf_counter()
        lowered = self._inner.lower(*arguments, **keywords)
        record = _current()

        class TimedLowered:
            def compile(self, *compile_arguments, **compile_keywords):
                compiled = lowered.compile(*compile_arguments, **compile_keywords)
                if record is not None:
                    record['compile_s'] = record.get('compile_s', 0.0) + time.perf_counter() - started
                return compiled

        return TimedLowered()

    def __call__(self, *arguments, **keywords):
        started = time.perf_counter()
        outputs = self._inner(*arguments, **keywords)
        self._block(outputs)
        elapsed = time.perf_counter() - started
        record = _current()
        if record is not None:
            record['exec_s'] = record.get('exec_s', 0.0) + elapsed
        return outputs

    def __getattr__(self, name):
        return getattr(self._inner, name)


def padded_step_length(model, protein_states):
    """The residue count the step actually folds, which is the padded one.

    BindCraft 2 rounds every chain up to a multiple of `length_bucket_size`, 32 by default
    (`af2.py:47`), so a step's cost belongs to its bucket and not to the binder length drawn.
    """
    from bindcraft.af2 import pad_design_chains

    padded = pad_design_chains(protein_states, model.length_bucket_size, model.target_pad_length)
    return {name: sum(len(protein) for protein in complex_.values()) for name, complex_ in padded.items()}


def wrap(model_class, block_until_ready, handle):
    """Patch the design model in place; returns nothing, the patch is the effect.

    Kept apart from `install` so the plumbing can be exercised against a stub on a machine with
    neither jax nor a card, which is where this harness gets written.
    """
    lock = threading.Lock()
    counter = {'n': 0}
    compiled_inner = model_class._compiled_sequence_gradients
    gradients_inner = model_class.sequence_gradients

    def compiled_sequence_gradients(self, *arguments, **keywords):
        return TimedCompiled(compiled_inner(self, *arguments, **keywords), block_until_ready)

    def sequence_gradients(self, protein_states, losses, *arguments, **keywords):
        record = {'compile_s': 0.0, 'exec_s': 0.0}
        RECORD.record = record
        started = time.perf_counter()
        try:
            result = gradients_inner(self, protein_states, losses, *arguments, **keywords)
        finally:
            RECORD.record = None
        record['call_s'] = time.perf_counter() - started
        with lock:
            counter['n'] += 1
            record['step'] = counter['n']
        record['compile_only'] = bool(keywords.get('compile_only'))
        record['background'] = threading.current_thread() is not threading.main_thread()
        record['model'] = keywords.get('model') or getattr(self, 'models', ('?',))[0]
        record['num_recycle'] = self.num_recycle
        record['true_residues'] = {name: sum(len(p) for p in c.values()) for name, c in protein_states.items()}
        try:
            record['padded_residues'] = padded_step_length(self, protein_states)
        except Exception as failure:  # a shape this helper cannot pad must not end the campaign
            record['padded_residues'] = f'unavailable: {failure}'
        record['t_unix'] = time.time()
        handle.write(json.dumps(record) + '\n')
        return result

    model_class._compiled_sequence_gradients = compiled_sequence_gradients
    model_class.sequence_gradients = sequence_gradients


def install(jsonl_path):
    import jax
    from bindcraft.af2 import AlphaFoldDesignModel

    handle = open(jsonl_path, 'a', buffering=1)
    wrap(AlphaFoldDesignModel, jax.block_until_ready, handle)
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jsonl', required=True, help='where one line per gradient step is written')
    parser.add_argument('rest', nargs=argparse.REMAINDER, help='the bindcraft command line, after --')
    arguments = parser.parse_args()
    command = arguments.rest[1:] if arguments.rest[:1] == ['--'] else arguments.rest
    if not command:
        parser.error('nothing to run: put the bindcraft command line after --')
    install(os.path.abspath(arguments.jsonl))
    from bindcraft.cli import main as bindcraft_main

    bindcraft_main(command)


if __name__ == '__main__':
    sys.exit(main())
