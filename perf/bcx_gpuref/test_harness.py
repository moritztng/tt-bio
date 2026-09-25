"""The harness, exercised against a stub design model and synthetic structures.

    python3 test_harness.py

It needs neither jax, nor BindCraft 2, nor a card, so the measurement code is known to work before
an hour of GPU is spent finding out. What it cannot check is that the stub's shape matches
BindCraft 2's: `sequence_gradients` calling `_compiled_sequence_gradients(...)` and then
`.lower(...).compile()` before invoking it is asserted here and read from `bindcraft/af2.py:400-406`
at commit 7a2dfdb8a285232a6f881899fe135c6dc48679f1, the commit the device arms run.
"""
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_steps
import bc2_step_timing
import validate_designs


def install_stub_bindcraft():
    """Stand in for `bindcraft.af2.pad_design_chains` so the padding path is exercised here too.

    The harness asks BindCraft 2 for the padded length rather than rounding it itself, so that it
    can never disagree with what the card actually folds. That import is the one thing in the
    plumbing a machine without BindCraft 2 cannot reach, so it gets a stub with the same contract:
    every chain up to the next multiple of the bucket.
    """
    import types

    module = types.ModuleType('bindcraft.af2')

    def pad_design_chains(protein_states, bucket_size=32, target_pad_length=0):
        def padded(length):
            return -(-length // bucket_size) * bucket_size
        return {state: {name: StubProtein(padded(len(protein))) for name, protein in complex_.items()}
                for state, complex_ in protein_states.items()}

    module.pad_design_chains = pad_design_chains
    package = types.ModuleType('bindcraft')
    package.af2 = module
    sys.modules.setdefault('bindcraft', package)
    sys.modules['bindcraft.af2'] = module

COMPILE_S = 0.05
EXEC_S = 0.02


class StubLowered:
    def __init__(self, cache, key):
        self.cache, self.key = cache, key

    def compile(self):
        if self.key not in self.cache:
            time.sleep(COMPILE_S)
            self.cache.add(self.key)
        return self


class StubCompiled:
    """Stands in for the jitted gradient function, and compiles once per shape like the real one."""

    def __init__(self, cache):
        self.cache = cache

    def lower(self, *arguments):
        return StubLowered(self.cache, arguments[0])

    def __call__(self, *arguments):
        time.sleep(EXEC_S)
        return ('loss', {'B': 'gradient'})


class StubProtein:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class StubModel:
    """The three attributes and two methods the harness touches on `AlphaFoldDesignModel`."""

    length_bucket_size = 32
    target_pad_length = 0
    num_recycle = 1
    models = ('model_1_multimer_v3',)

    def __init__(self):
        self.cache = set()

    def _compiled_sequence_gradients(self, key):
        return StubCompiled(self.cache)

    def sequence_gradients(self, protein_states, losses, model=None, compile_only=False):
        compiled = self._compiled_sequence_gradients(tuple(sorted(protein_states)))
        key = max(sum(len(p) for p in c.values()) for c in protein_states.values())
        compiled.lower(key).compile()
        if compile_only:
            return {}, {}, 0.0
        return compiled(key)


def states(binder_length, target_length=115):
    return {'complex': {'A': StubProtein(target_length), 'B': StubProtein(binder_length)}}


class TimingHarness(unittest.TestCase):
    def setUp(self):
        install_stub_bindcraft()
        self.handle = io.StringIO()
        # The step counter is process-wide, because the phase wrappers read it to say how many
        # gradient steps a stage contained. One campaign per process, so one counter -- but a test
        # file runs many, and each starts at step 1.
        bc2_step_timing.STEPS['n'] = 0
        self.model_class = type('Stub', (StubModel,), {})
        bc2_step_timing.wrap(self.model_class, lambda outputs: None, self.handle)
        self.model = self.model_class()

    def records(self):
        return [json.loads(line) for line in self.handle.getvalue().splitlines() if line.strip()]

    def test_a_step_is_logged_with_compile_and_execution_apart(self):
        self.model.sequence_gradients(states(96), {})
        record, = self.records()
        self.assertGreaterEqual(record['compile_s'], COMPILE_S)
        self.assertGreaterEqual(record['exec_s'], EXEC_S)
        self.assertGreaterEqual(record['call_s'], record['compile_s'] + record['exec_s'])
        self.assertEqual(record['num_recycle'], 1)
        self.assertEqual(record['step'], 1)

    def test_the_second_step_at_a_shape_carries_no_compilation(self):
        for _ in range(2):
            self.model.sequence_gradients(states(96), {})
        cold, warm = self.records()
        self.assertGreaterEqual(cold['compile_s'], COMPILE_S)
        self.assertLess(warm['compile_s'], COMPILE_S / 2)
        self.assertEqual(analyze_steps.classify(cold), 'cold')
        self.assertEqual(analyze_steps.classify(warm), 'warm')

    def test_a_new_shape_compiles_again(self):
        self.model.sequence_gradients(states(96), {})
        self.model.sequence_gradients(states(96), {})
        self.model.sequence_gradients(states(160), {})
        self.assertEqual([analyze_steps.classify(r) for r in self.records()], ['cold', 'warm', 'cold'])

    def test_the_background_precompile_is_not_a_step(self):
        self.model.sequence_gradients(states(96), {}, compile_only=True)
        record, = self.records()
        self.assertTrue(record['compile_only'])
        self.assertEqual(analyze_steps.classify(record), 'precompile')

    def test_a_step_off_the_main_thread_is_marked(self):
        thread = threading.Thread(target=lambda: self.model.sequence_gradients(states(96), {}))
        thread.start()
        thread.join()
        record, = self.records()
        self.assertTrue(record['background'])

    def test_concurrent_steps_do_not_share_a_record(self):
        threads = [threading.Thread(target=lambda: self.model.sequence_gradients(states(96), {})) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        records = self.records()
        self.assertEqual(len(records), 4)
        self.assertEqual(sorted(r['step'] for r in records), [1, 2, 3, 4])
        for record in records:
            self.assertGreaterEqual(record['exec_s'], EXEC_S)

    def test_the_denominator_excludes_compilation(self):
        for _ in range(5):
            self.model.sequence_gradients(states(96), {})
        report = analyze_steps.report(self.records())
        self.assertEqual(report['warm']['call_s']['n'], 4)
        self.assertEqual(report['cold']['call_s']['n'], 1)
        self.assertLess(report['warm']['call_s']['p50'], report['cold']['call_s']['p50'])
        # a 96-residue binder against the 115-residue target folds as 96 + 128 padded residues
        self.assertEqual(list(report['warm_by_padded_residues']), ['224'])

    def test_the_padded_length_is_the_one_the_card_folds(self):
        self.model.sequence_gradients(states(100), {})
        record, = self.records()
        self.assertEqual(record['true_residues'], {'complex': 215})
        self.assertEqual(record['padded_residues'], {'complex': 256})


PDB = """ATOM      1  CA  MET A   1      11.000  11.000  11.000  1.00 90.00           C
ATOM      2  CA  ALA A   2      12.000  11.000  11.000  1.00 90.00           C
ATOM      3  CA  GLY B   1      13.000  11.000  11.000  1.00 90.00           C
ATOM      4  CA  SER B   2      14.000  11.000  11.000  1.00 90.00           C
ATOM      5  CA  LEU B   3      15.000  11.000  11.000  1.00 90.00           C
END
"""

CIF = """data_design
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 CA MET A 1 A 1 11.0 11.0 11.0
ATOM 2 N  MET A 1 A 1 11.5 11.0 11.0
ATOM 3 CA ALA A 2 A 2 12.0 11.0 11.0
ATOM 4 CA GLY B 1 B 1 13.0 11.0 11.0
ATOM 5 CA SER B 2 B 2 14.0 11.0 11.0
ATOM 6 CA LEU B 3 B 3 15.0 11.0 11.0
#
"""


class DesignValidation(unittest.TestCase):
    def write(self, name, text):
        path = os.path.join(self.directory.name, name)
        with open(path, 'w') as handle:
            handle.write(text)
        return path

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_pdb_and_cif_read_the_same_chains(self):
        self.assertEqual(validate_designs.read_chains(self.write('d.pdb', PDB)),
                         validate_designs.read_chains(self.write('d.cif', CIF)))

    def test_a_good_design_passes(self):
        result = validate_designs.check(self.write('d.cif', CIF), binder_length=3, target_length=2)
        self.assertTrue(result['ok'], result['problems'])
        self.assertEqual(result['binder_chain'], 'B')

    def test_the_wrong_binder_length_fails(self):
        self.assertFalse(validate_designs.check(self.write('d.cif', CIF), binder_length=4)['ok'])

    def test_a_third_chain_fails(self):
        extra = CIF.replace('#\n', '', 1).rstrip()[:-1] + 'ATOM 7 CA GLY C 1 C 1 16.0 11.0 11.0\n#\n'
        result = validate_designs.check(self.write('t.cif', extra), binder_length=3)
        self.assertFalse(result['ok'])
        self.assertIn('3 chains', result['problems'][0])

    def test_a_degenerate_binder_fails(self):
        poly = '\n'.join(f'ATOM {i} CA ALA B {i} B {i} {i}.0 11.0 11.0' for i in range(1, 13))
        text = CIF.replace('ATOM 4 CA GLY B 1 B 1 13.0 11.0 11.0\nATOM 5 CA SER B 2 B 2 14.0 11.0 11.0\nATOM 6 CA LEU B 3 B 3 15.0 11.0 11.0', poly)
        result = validate_designs.check(self.write('p.cif', text), binder_length=12, max_run=8)
        self.assertFalse(result['ok'])
        self.assertIn('degenerate', result['problems'][0])

    def test_a_run_at_the_limit_passes(self):
        poly = '\n'.join(f'ATOM {i} CA ALA B {i} B {i} {i}.0 11.0 11.0' for i in range(1, 9))
        text = CIF.replace('ATOM 4 CA GLY B 1 B 1 13.0 11.0 11.0\nATOM 5 CA SER B 2 B 2 14.0 11.0 11.0\nATOM 6 CA LEU B 3 B 3 15.0 11.0 11.0', poly)
        self.assertTrue(validate_designs.check(self.write('e.cif', text), binder_length=8, max_run=8)['ok'])


class StubArray:
    """Stands in for a jax.Array: it knows whether anything waited on it."""

    def __init__(self):
        self.blocked = 0

    def block_until_ready(self):
        self.blocked += 1
        return self


class StubProteinHolder:
    """A plain object holding an array, which is what `jax.block_until_ready` walks straight past."""

    def __init__(self, array):
        self.coordinates = array


class PhaseTimingTest(unittest.TestCase):
    """The phase arm: whole cycle against gradient phase, on the same run.

    Our 8,069.9 chip-s is the 125-step gradient phase and BindCraft 2's published 90.5 s is a whole
    cycle, so a phase-matched ratio needs both timed here. These tests fix the shape of that
    accounting against a stub of BindCraft 2's own call nesting.
    """

    def setUp(self):
        import types
        self.handle = io.StringIO()
        bc2_step_timing.STEPS['n'] = 0
        self.trajectory = types.ModuleType('t')
        self.campaign = types.ModuleType('c')

        def gradient_stage(steps):
            for _ in range(steps):
                bc2_step_timing.next_step_number()
            return ('states', 'predictions')

        def mutation_stage():
            return ('states', 'predictions')

        def run_trajectory(rounds=(50, 25, 45, 5), mutate=True):
            for count in rounds:
                self.trajectory.run_gradient_design_stage(count)
            if mutate:
                self.trajectory.run_sequence_mutation_stage()
            return 'trajectory'

        self.trajectory.run_gradient_design_stage = gradient_stage
        self.trajectory.run_sequence_mutation_stage = mutation_stage
        self.campaign.run_trajectory = run_trajectory
        self.campaign.redesign_and_validate_binders = lambda: ['binder']
        bc2_step_timing.wrap_phases({'bindcraft.trajectory': self.trajectory,
                                     'bindcraft.campaign': self.campaign},
                                    self.handle, lambda value: isinstance(value, StubArray))

    def records(self):
        return [json.loads(line) for line in self.handle.getvalue().splitlines() if line.strip()]

    def test_the_gradient_phase_is_the_four_stages_and_not_the_mutate_stage(self):
        self.campaign.run_trajectory()
        self.campaign.redesign_and_validate_binders()
        rows = analyze_steps.trajectories([r for r in self.records() if r.get('record') == 'phase'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['gradient_steps'], 125)
        self.assertEqual(len(rows[0]['gradient_stages']), 4)
        self.assertTrue(rows[0]['reached_mpnn'])

    def test_the_whole_cycle_exceeds_the_design_wall_by_the_mpnn_stage(self):
        self.campaign.run_trajectory()
        self.campaign.redesign_and_validate_binders()
        row = analyze_steps.trajectories([r for r in self.records() if r.get('record') == 'phase'])[0]
        self.assertAlmostEqual(row['whole_cycle_s'], row['design_s'] + row['mpnn_validation_s'], places=9)
        self.assertGreater(row['whole_cycle_s'], row['design_s'])

    def test_the_gradient_phase_does_not_double_count_inside_the_design_wall(self):
        """`design` encloses the stages, so `depth` is what keeps a reader from adding both."""
        self.campaign.run_trajectory()
        phases = [r for r in self.records() if r.get('record') == 'phase']
        depths = {r['phase']: r['depth'] for r in phases}
        self.assertEqual(depths['design'], 0)
        self.assertEqual(depths['gradient_stage'], 1)
        self.assertEqual(depths['mutate'], 1)

    def test_a_trajectory_rejected_before_mpnn_reports_no_whole_cycle(self):
        self.campaign.run_trajectory(rounds=(50, 25), mutate=False)
        row = analyze_steps.trajectories([r for r in self.records() if r.get('record') == 'phase'])[0]
        self.assertEqual(row['gradient_steps'], 75)
        self.assertFalse(row['reached_mpnn'])
        self.assertEqual(row['mutate_s'], 0.0)
        summary = analyze_steps.report(self.records())
        self.assertEqual(summary['phase_split']['trajectories_through_mpnn'], 0)
        self.assertEqual(summary['phase_split']['whole_cycle_s'], {'n': 0})

    def test_two_trajectories_are_two_rows(self):
        for _ in range(2):
            self.campaign.run_trajectory()
            self.campaign.redesign_and_validate_binders()
        rows = analyze_steps.trajectories([r for r in self.records() if r.get('record') == 'phase'])
        self.assertEqual([r['gradient_steps'] for r in rows], [125, 125])
        self.assertEqual(rows[1]['first_step'], 126)

    def test_a_phase_settles_arrays_a_pytree_walk_would_miss(self):
        """BindCraft 2's `Protein` is not a registered pytree node, so its arrays are invisible to
        `jax.block_until_ready`; an unsettled phase would charge its tail to a later line."""
        array = StubArray()
        self.campaign.redesign_and_validate_binders = lambda: {'a': [StubProteinHolder(array)]}
        bc2_step_timing.wrap_phases({'bindcraft.campaign': self.campaign}, self.handle,
                                    lambda value: isinstance(value, StubArray),
                                    phases=(('bindcraft.campaign', 'redesign_and_validate_binders', 'mpnn_validation'),))
        self.campaign.redesign_and_validate_binders()
        self.assertEqual(array.blocked, 1)

    def test_settle_is_bounded_so_a_cycle_cannot_hang_it(self):
        array = StubArray()
        holder = StubProteinHolder(array)
        holder.self_reference = holder
        bc2_step_timing.settle(holder, lambda value: isinstance(value, StubArray))
        self.assertGreaterEqual(array.blocked, 1)

    def test_a_missing_attribute_is_skipped_rather_than_raising(self):
        """A BindCraft 2 that renames a phase function must not end the campaign in the wrapper."""
        import types
        empty = types.ModuleType('empty')
        bc2_step_timing.wrap_phases({'bindcraft.campaign': empty}, self.handle,
                                    lambda value: False)
        self.assertEqual(self.records(), [])

    def test_a_background_precompile_is_not_a_trajectory(self):
        phases = [{'record': 'phase', 'phase': 'design', 'wall_s': 1.0, 'depth': 0,
                   'gradient_steps': 5, 'first_step': 1, 'last_step': 5, 'background': True}]
        self.assertEqual(analyze_steps.trajectories(phases), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
