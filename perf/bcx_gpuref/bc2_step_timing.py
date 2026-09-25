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

# One counter for the whole process, because the phase wrappers below read it to say how many
# gradient steps a stage contained. A counter private to `wrap` could not answer that.
STEPS = {'n': 0}
STEPS_LOCK = threading.Lock()


def _current():
    return getattr(RECORD, 'record', None)


def next_step_number():
    with STEPS_LOCK:
        STEPS['n'] += 1
        return STEPS['n']


def steps_so_far():
    with STEPS_LOCK:
        return STEPS['n']


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
        record['step'] = next_step_number()
        record['compile_only'] = bool(keywords.get('compile_only'))
        record['background'] = threading.current_thread() is not threading.main_thread()
        record['model'] = keywords.get('model') or getattr(self, 'models', ('?',))[0]
        record['num_recycle'] = self.num_recycle
        record['true_residues'] = {name: sum(len(p) for p in c.values()) for name, c in protein_states.items()}
        try:
            record['padded_residues'] = padded_step_length(self, protein_states)
        except Exception as failure:  # a shape this helper cannot pad must not end the campaign
            record['padded_residues'] = f'unavailable: {failure}'
        record['record'] = 'step'
        record['t_unix'] = time.time()
        handle.write(json.dumps(record) + '\n')
        return result

    model_class._compiled_sequence_gradients = compiled_sequence_gradients
    model_class.sequence_gradients = sequence_gradients




# The four phases a BindCraft 2 trajectory is made of, and where each one is called from. All four
# names are module attributes at their call site, so rebinding them here is enough: `run_trajectory`
# and `redesign_and_validate_binders` are imported into `bindcraft.campaign` (campaign.py:21 and
# :10) and called there, and the two stage functions are called inside `bindcraft.trajectory`.
#
# This split is the whole point of the phase arm. `8,069.9 chip-s` on our side is the 125-step
# GRADIENT PHASE; BindCraft 2's published 90.5 s/trajectory on a GH200 is the WHOLE CYCLE, gradient
# design plus ProteinMPNN redesign plus validation. Comparing the two compares our part to their
# whole, so both have to be timed here on the same card in the same run.
PHASES = (
    ('bindcraft.trajectory', 'run_gradient_design_stage', 'gradient_stage'),
    ('bindcraft.trajectory', 'run_sequence_mutation_stage', 'mutate'),
    ('bindcraft.campaign', 'run_trajectory', 'design'),
    ('bindcraft.campaign', 'redesign_and_validate_binders', 'mpnn_validation'),
)

DEPTH = threading.local()


def settle(value, is_array, depth=6):
    """Wait for every device array reachable from `value`, so a phase wall has no async tail.

    `jax.block_until_ready` walks pytrees, and BindCraft 2's `Protein` and `StructurePrediction`
    are plain classes rather than registered nodes, so the arrays inside them are invisible to it.
    A phase that returned without settling would charge its own tail to whichever later line first
    read the array, which is exactly the accounting error this harness exists to avoid.
    """
    if depth < 0:
        return
    if is_array(value):
        try:
            value.block_until_ready()
        except Exception:  # a deleted or committed array is already settled
            pass
        return
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    else:
        contents = getattr(value, '__dict__', None)
        children = contents.values() if isinstance(contents, dict) else ()
    for child in children:
        settle(child, is_array, depth - 1)


def wrap_phases(namespaces, handle, is_array, phases=PHASES):
    """Time each phase, in place. `namespaces` maps a module name to the module object.

    Nesting is recorded rather than unwound: `design` encloses the four `gradient_stage` calls and
    `mutate`, so a reader that summed every phase would count the same seconds twice. `depth` is
    what lets the analysis add up only the phases at one level.
    """
    for module_name, attribute, phase in phases:
        module = namespaces.get(module_name)
        if module is None or not hasattr(module, attribute):
            continue
        inner = getattr(module, attribute)

        def timed(*arguments, _inner=inner, _phase=phase, **keywords):
            depth = getattr(DEPTH, 'n', 0)
            DEPTH.n = depth + 1
            first_step, started = steps_so_far() + 1, time.perf_counter()
            try:
                result = _inner(*arguments, **keywords)
                settle(result, is_array)
                return result
            finally:
                DEPTH.n = depth
                last_step = steps_so_far()
                handle.write(json.dumps({'record': 'phase',
                                         'phase': _phase,
                                         'wall_s': time.perf_counter() - started,
                                         'depth': depth,
                                         'gradient_steps': max(0, last_step - first_step + 1),
                                         'first_step': first_step,
                                         'last_step': last_step,
                                         'background': threading.current_thread() is not threading.main_thread(),
                                         't_unix': time.time()}) + '\n')

        setattr(module, attribute, timed)


def install(jsonl_path):
    import jax
    import bindcraft.campaign
    import bindcraft.trajectory
    from bindcraft.af2 import AlphaFoldDesignModel

    handle = open(jsonl_path, 'a', buffering=1)
    wrap(AlphaFoldDesignModel, jax.block_until_ready, handle)
    wrap_phases({'bindcraft.campaign': bindcraft.campaign,
                 'bindcraft.trajectory': bindcraft.trajectory},
                handle, lambda value: isinstance(value, jax.Array))
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
