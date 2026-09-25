"""Check the harness's assumptions against BindCraft 2's real source, before the card is rented.

    BC2_SRC=/path/to/BindCraft2 python3 test_bc2_contract.py

`test_harness.py` exercises the plumbing against a stub, which cannot notice that BindCraft 2 has
renamed a method or stopped calling `.lower(...).compile()`. This reads the real `bindcraft/af2.py`
with `ast` and asserts the four things the harness patches into, plus the two facts the state doc
quotes. It needs no jax, no card and no AlphaFold parameters, only a clone -- so the failure mode
it catches is found on pc for nothing instead of on a billing box.

The clone is not vendored here: BindCraft 2 is under the UZH Source-Available licence, so this
points at a clone rather than carrying one. Skips cleanly when `BC2_SRC` is unset.
"""
import ast
import json
import os
import sys
import unittest

# The commit the campaign's DEVICE arms run on qb1 and qb2 (`state/bcx/UPSTREAM.md`). A GPU
# reference measured against a different BC2 than the arm it is the denominator for would
# repeat this row's founding mistake one level up.
PINNED = '7a2dfdb8a285232a6f881899fe135c6dc48679f1'
SOURCE = os.environ.get('BC2_SRC')


def text(name):
    with open(os.path.join(SOURCE, 'bindcraft', name)) as handle:
        return handle.read()


def module(name):
    return ast.parse(text(name))


def imported_from(tree):
    """`{name: module}` for every module-level `from X import name`.

    This is what makes the phase timing possible at all: a name imported into a module and called
    there is an attribute of that module, so rebinding the attribute reaches the call site. A BC2
    that switched to `import bindcraft.trajectory` and called `trajectory.run_trajectory(...)`
    would leave the wrapper installed and measuring nothing.
    """
    names = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names[alias.asname or alias.name] = node.module
    return names


def function(tree, name, inside=None):
    scope = tree
    if inside:
        scope = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == inside)
    return next((node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name == name), None)


def calls(node):
    """Every call in `node`, as the dotted name being called."""
    names = []
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            target = inner.func
            parts = []
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name):
                parts.append(target.id)
            names.append('.'.join(reversed(parts)))
    return names


@unittest.skipUnless(SOURCE, 'set BC2_SRC to a BindCraft 2 clone')
class DesignModelContract(unittest.TestCase):
    """What `bc2_step_timing.wrap` patches. If any of this moves, the harness silently measures nothing."""

    @classmethod
    def setUpClass(cls):
        cls.af2 = module('af2.py')

    def test_the_class_the_harness_patches_exists(self):
        self.assertTrue(any(isinstance(node, ast.ClassDef) and node.name == 'AlphaFoldDesignModel'
                            for node in self.af2.body))

    def test_both_patched_methods_exist(self):
        for name in ('sequence_gradients', '_compiled_sequence_gradients'):
            self.assertIsNotNone(function(self.af2, name, 'AlphaFoldDesignModel'), name)

    def test_sequence_gradients_goes_through_the_compiled_function(self):
        body = function(self.af2, 'sequence_gradients', 'AlphaFoldDesignModel')
        self.assertIn('self._compiled_sequence_gradients', calls(body))

    def test_sequence_gradients_lowers_and_compiles_before_calling(self):
        """The harness times `lower(...).compile()` apart from the call; both must still be there.

        Asserted on the AST shape rather than on a dotted name, because `x.lower(...).compile()` is
        a call whose callee is an attribute of another call, and flattening it to a name loses
        exactly the chaining this checks.
        """
        body = function(self.af2, 'sequence_gradients', 'AlphaFoldDesignModel')
        chained = [node for node in ast.walk(body)
                   if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'compile'
                   and isinstance(node.func.value, ast.Call)
                   and isinstance(node.func.value.func, ast.Attribute) and node.func.value.func.attr == 'lower']
        self.assertEqual(len(chained), 1, 'expected exactly one lower(...).compile() on the step path')
        self.assertIn('compiled_sequence_gradients', calls(body), 'the compiled function is never called')

    def test_the_compile_only_shortcut_still_exists(self):
        body = function(self.af2, 'sequence_gradients', 'AlphaFoldDesignModel')
        self.assertIn('compile_only', [argument.arg for argument in body.args.kwonlyargs + body.args.args])

    def test_pad_design_chains_takes_what_the_harness_passes(self):
        padder = function(self.af2, 'pad_design_chains')
        self.assertIsNotNone(padder)
        self.assertEqual([argument.arg for argument in padder.args.args],
                         ['protein_states', 'bucket_size', 'target_pad_length'])


@unittest.skipUnless(SOURCE, 'set BC2_SRC to a BindCraft 2 clone')
class StepShape(unittest.TestCase):
    """The two facts the state doc quotes about what one gradient step contains."""

    @classmethod
    def setUpClass(cls):
        cls.af2 = module('af2.py')
        cls.campaign = module('campaign.py')

    def test_a_step_is_two_forwards_with_the_gradient_over_the_last(self):
        recycled = function(self.af2, 'recycled_alphafold_outputs')
        self.assertIsNotNone(recycled)
        loop, = [node for node in recycled.body if isinstance(node, ast.For)]
        self.assertIn('jax.lax.stop_gradient', calls(loop))
        returned, = [node for node in recycled.body if isinstance(node, ast.Return)]
        self.assertNotIn('jax.lax.stop_gradient', calls(returned))
        self.assertIn('alphafold_runner.apply', calls(returned))

    def test_the_backward_is_one_value_and_grad(self):
        self.assertIn('jax.value_and_grad', calls(self.af2))

    def test_the_length_bucket_is_thirty_two(self):
        assignment, = [node for node in self.af2.body if isinstance(node, ast.Assign)
                       and any(getattr(target, 'id', None) == 'DEFAULT_LENGTH_BUCKET' for target in node.targets)]
        self.assertEqual(ast.literal_eval(assignment.value), 32)

    def test_the_next_bucket_is_precompiled_on_a_background_thread(self):
        precompile = function(self.campaign, 'compile_next_length_bucket')
        self.assertIsNotNone(precompile)
        self.assertIn('threading.Thread', calls(precompile))
        self.assertIn('alphafold_model.sequence_gradients', calls(precompile))


@unittest.skipUnless(SOURCE, 'set BC2_SRC to a BindCraft 2 clone')
class StepAccounting(unittest.TestCase):
    """How many gradient steps a trajectory spends, which is what turns a per-step figure into a wall.

    The campaign quotes "125 backprop iterations per trajectory" for BindCraft 2 from the shape of
    FreeBindCraft's schedule. It happens to be right, and these read it out of BC2 instead.
    """

    @classmethod
    def setUpClass(cls):
        cls.settings = module('settings.py')
        cls.trajectory = module('trajectory.py')
        cls.schedule = module('target_schedule.py')
        cls.af2 = module('af2.py')
        with open(os.path.join(SOURCE, 'settings', 'core', 'default.json')) as handle:
            cls.defaults = json.load(handle)
        with open(os.path.join(SOURCE, 'settings', 'core', 'benchmark.json')) as handle:
            cls.benchmark = json.load(handle)

    def rounds(self):
        assignment, = [node for node in self.settings.body if isinstance(node, ast.Assign)
                       and any(getattr(target, 'id', None) == 'DESIGN_STAGE_DEFAULT_ROUNDS' for target in node.targets)]
        return ast.literal_eval(assignment.value)

    def test_the_four_gradient_stages_sum_to_125_rounds(self):
        rounds = self.rounds()
        self.assertEqual(rounds, {'screen': 50, 'refine': 25, 'anneal': 45, 'harden': 5, 'mutate': 15})
        gradient = {stage: count for stage, count in rounds.items() if stage != 'mutate'}
        self.assertEqual(sum(gradient.values()), 125)

    def test_mutate_is_the_forward_only_stage_and_is_left_out(self):
        """`gradient_stage_rounds` drops exactly `mutate`, so its 15 rounds carry no backward."""
        body = function(self.settings, 'gradient_stage_rounds')
        self.assertIn("'mutate'", ast.unparse(body))
        mutation_stage = function(self.trajectory, 'run_sequence_mutation_stage')
        self.assertIsNotNone(mutation_stage)
        self.assertNotIn('design_model.sequence_gradients', calls(mutation_stage))
        self.assertIn('design_model.predict', calls(mutation_stage))

    def test_one_round_is_one_gradient_call(self):
        """The gradient stage calls `sequence_gradients` once per loop turn and the loop turns once
        per sequence update, so rounds and gradient calls are the same number for one target."""
        stage = function(self.trajectory, 'run_gradient_design_stage')
        loop, = [node for node in stage.body if isinstance(node, ast.While)]
        self.assertEqual(calls(loop).count('design_model.sequence_gradients'), 1)
        updater = function(self.schedule, 'should_update_sequence', 'IterationLimitedDesignSchedule')
        returned, = [node for node in updater.body if isinstance(node, ast.Return)]
        self.assertIs(ast.literal_eval(returned.value), True)

    def test_a_gradient_call_folds_one_sampled_model_not_the_pool(self):
        """`model=None` resolves to one sampled design model, so a round is one step and not five."""
        resolver = function(self.af2, '_resolve_model_name', 'AlphaFoldDesignModel')
        self.assertIn('self._sample_design_model', calls(resolver))

    def test_a_second_binding_target_doubles_the_budget(self):
        """Not our case -- hPDL1 is one target -- but it is the one setting that moves 125."""
        budget = function(self.settings, 'rounds_per_binding_target')
        self.assertIn('2 if', ast.unparse(budget))

    def test_the_betasheet_reopt_extras_are_zero_by_default(self):
        self.assertEqual(self.defaults['betasheet_reopt_extra_refine_steps'], 0)
        self.assertEqual(self.defaults['betasheet_reopt_extra_anneal_steps'], 0)
        self.assertEqual(self.defaults['design_recycles'], 1)

    def test_benchmark_core_turns_off_what_would_move_the_step_count(self):
        """`--core benchmark` is what makes the reference run reproducible: autotune and
        desperation both rewrite the stage budget mid-campaign."""
        self.assertEqual(self.benchmark['campaign_seed'], 0)
        self.assertIs(self.benchmark['autotune'], False)
        self.assertIs(self.benchmark['desperation'], False)


@unittest.skipUnless(SOURCE, 'set BC2_SRC to a BindCraft 2 clone')
class PhaseContract(unittest.TestCase):
    """What `bc2_step_timing.wrap_phases` patches, and why patching there reaches the call site.

    The phase arm exists because `8,069.9 chip-s` on our side is the 125-step gradient phase while
    BindCraft 2's published 90.5 s/trajectory is the whole cycle. If any of these four hooks moves,
    the harness logs a whole cycle with no gradient phase inside it, or no phases at all, and the
    ratio silently goes back to comparing our part against their whole.
    """

    @classmethod
    def setUpClass(cls):
        cls.trajectory, cls.campaign = module('trajectory.py'), module('campaign.py')
        cls.workers_text = text('design_workers.py')

    def test_the_four_phase_functions_exist_where_the_harness_looks(self):
        self.assertIsNotNone(function(self.trajectory, 'run_gradient_design_stage'))
        self.assertIsNotNone(function(self.trajectory, 'run_sequence_mutation_stage'))
        self.assertIsNotNone(function(self.trajectory, 'run_trajectory'))

    def test_the_campaign_imports_both_phase_names_it_calls(self):
        imports = imported_from(self.campaign)
        self.assertEqual(imports.get('run_trajectory'), 'bindcraft.trajectory')
        self.assertEqual(imports.get('redesign_and_validate_binders'), 'bindcraft.MPNN_stage')

    def test_the_campaign_calls_them_by_bare_name(self):
        """A dotted call would read through to the defining module and miss the rebinding."""
        called = calls(function(self.campaign, 'run_campaign_arm'))
        self.assertIn('run_trajectory', called)
        self.assertIn('redesign_and_validate_binders', called)

    def test_the_gradient_stage_and_the_mutate_stage_are_called_by_bare_name_too(self):
        self.assertIn('run_gradient_design_stage', calls(function(self.trajectory, 'run_trajectory')))
        self.assertIn('run_mutation_polish', calls(function(self.trajectory, 'run_trajectory')))
        self.assertIn('run_sequence_mutation_stage', calls(function(self.trajectory, 'run_mutation_polish')))

    def test_the_mutate_stage_is_still_outside_the_gradient_phase(self):
        """Forward-only: `run_sequence_mutation_stage` never reaches `sequence_gradients`, so its
        15 rounds belong to the cycle and not to the 125-step gradient phase."""
        called = calls(function(self.trajectory, 'run_sequence_mutation_stage'))
        self.assertNotIn('sequence_gradients', [name.rsplit('.', 1)[-1] for name in called])

    def test_bc2_times_its_own_design_wall_without_the_mpnn_stage(self):
        """`design_started` is set immediately before `run_trajectory`, and `timing_stamp` reads the
        clock right after it, so BC2's own `Timing` column corroborates our `design` phase rather
        than measuring something else. The MPNN stage runs after that line."""
        assignments = [node for node in ast.walk(function(self.campaign, 'run_campaign_arm'))
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'design_started' for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertIn('time.time', calls(assignments[0]))

    def test_one_process_is_reachable_by_turning_auto_multi_gpu_off(self):
        """The wrapper only instruments the process it lives in, so the measured run must not fan
        out into subprocess workers. `auto_multi_gpu=false` is the documented way back."""
        dispatch = self.workers_text[self.workers_text.index('def dispatch_design_workers'):]
        guard = dispatch[:dispatch.index('plan = ')]
        self.assertIn('auto_multi_gpu', guard)
        self.assertIn('return None', guard)


if __name__ == '__main__':
    if SOURCE:
        print(f'reading {SOURCE} (pinned {PINNED})')
    unittest.main(verbosity=2)
