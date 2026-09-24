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
import os
import sys
import unittest

PINNED = '5342aefa18dedad653f7a5f6dbee1e566ca24d8f'
SOURCE = os.environ.get('BC2_SRC')


def module(name):
    with open(os.path.join(SOURCE, 'bindcraft', name)) as handle:
        return ast.parse(handle.read())


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


if __name__ == '__main__':
    if SOURCE:
        print(f'reading {SOURCE} (pinned {PINNED})')
    unittest.main(verbosity=2)
