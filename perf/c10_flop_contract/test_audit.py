"""Known answers and fail-closed controls; stdlib only, no device dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit


def node(counter, kind, name=None, **kw):
    return dict(counter=counter, node_type=kind, params={"name": name}, **kw)


def matrix_graph(a, b, out):
    def tensor(i, dims):
        return dict(counter=i, node_type="tensor", connections=[1],
                    params={"shape": f"Shape({list(dims)})", "address": i})
    return [node(0, "function_start", "ttnn.matmul"),
            node(1, "function_start", "MatmulDeviceOperation", input_tensors=[2, 3]),
            tensor(2, a), tensor(3, b),
            node(4, "function_start", "tt::tt_metal::create_device_tensor",
                 arguments=[f"Shape({list(out)})"]),
            node(5, "function_end", "tt::tt_metal::create_device_tensor"),
            node(6, "function_end", "MatmulDeviceOperation"),
            node(7, "function_end", "ttnn.matmul")]


class ArithmeticTests(unittest.TestCase):
    def test_dense_rectangular_batched_broadcast_and_padding(self):
        cases = [
            ((8192, 8192), (8192, 8192), (8192, 8192), 1099511627776, 1099511627776),
            ((64, 32), (32, 16), (64, 16), 65536, 131072),
            ((32, 32), (32, 64), (32, 64), 131072, 131072),
            ((2, 3, 5), (2, 5, 7), (2, 3, 7), 420, 131072),
            ((1, 3, 5), (7, 5, 11), (7, 3, 11), 2310, 458752),
            ((2, 1, 3, 5), (1, 4, 5, 7), (2, 4, 3, 7), 1680, 524288),
            ((1, 1), (1, 256), (1, 256), 512, 524288),
            ((33, 17), (17, 65), (33, 65), 72930, 393216),
        ]
        for a, b, out, logical, tiled in cases:
            with self.subTest(a=a, b=b):
                got = audit.matrix_work(a, b, out)
                self.assertEqual((got["shape_flops"], got["tile32_flops"]), (logical, tiled))

    def test_scalar_enumeration_independent_oracle(self):
        # Count every product and accumulation explicitly for small rectangular batches.
        expected = sum(2 for batch in range(7) for m in range(3)
                       for n in range(11) for k in range(5))
        self.assertEqual(audit.matrix_work((1, 3, 5), (7, 5, 11))["shape_flops"], expected)

    def test_transpose_and_repeated_buffer(self):
        self.assertEqual(audit.matrix_work((1, 4480, 768), (1, 4480, 512),
                         transpose_a=True)["shape_flops"], 3523215360)
        self.assertEqual(audit.matrix_work((3, 5), (7, 5), transpose_b=True)["shape_flops"], 210)
        dev = {"input_tensors": [1, 1]}
        got = audit.matrix_candidates(dev, {1: {"params": {"shape": "Shape([32, 32])"}}}, [(32, 32)])
        self.assertEqual(got["shape_flops"], 65536)

    def test_unsupported_shapes(self):
        for a, b, out in [((3,), (3, 2), None), ((3, 5), (6, 7), None),
                          ((2, 3, 5), (4, 5, 7), None), ((3, 5), (5, 7), (3, 8)),
                          ((0, 5), (5, 7), None), ((3, -5), (5, 7), None),
                          ((3.0, 5), (5, 7), None)]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                audit.matrix_work(a, b, out)

    def test_fused_full_weight_width_and_two_distinct_projections(self):
        # qkv + gate + physically padded bias: N=544, not 3*128 or 4*128.
        got = audit.known_fused_matrix("triatt_qkv", [((262144, 128), (128, 544))])
        self.assertEqual(got["matrix_shape_flops"], 36507222016)
        got = audit.known_fused_matrix("trimul_tail", [((262144, 256), (256, 256))] * 2)
        self.assertEqual(got["matrix_shape_flops"], 68719476736)
        self.assertEqual(got["remaining"], ["sigmoid and gate multiply"])
        self.assertIsNone(got["issued_flops"])

    def test_fused_head_major_requires_effective_matrix(self):
        got = audit.known_fused_matrix("mm_generic", [((262144, 256), (256, 128))])
        self.assertEqual(got["matrix_shape_flops"], 17179869184)
        with self.assertRaises(ValueError):
            audit.known_fused_matrix("mm_generic", [((512, 8, 512, 32), (256, 128))])

    def test_fused_identity_cannot_be_guessed(self):
        for kind, pairs in [("unknown", [((32, 32), (32, 32))]),
                            ("trimul_tail", [((32, 32), (32, 32))]),
                            ("trimul_tail", [((32, 32), (32, 32)), ((64, 32), (32, 32))])]:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit.known_fused_matrix(kind, pairs)

    def test_attention_two_matrix_terms_rectangular_different_value_width(self):
        got = audit.dense_attention_matrix((2, 3, 5, 7), (2, 3, 11, 7), (2, 3, 11, 13))
        self.assertEqual(got["matrix_shape_flops"], 13200)
        self.assertTrue(got["remaining"])
        self.assertIsNone(got["issued_flops"])
        self.assertEqual(audit.dense_attention_matrix(*([(1, 2, 32, 64)] * 3))[
            "matrix_shape_flops"], 524288)


class CaptureTests(unittest.TestCase):
    def replay(self, nodes):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "capture.json"
            path.write_text(json.dumps(nodes))
            return audit.audit_capture(path)

    def test_ordered_inputs_fix_square_and_broadcast_defects(self):
        for a, b, out, expected in [((32, 32), (32, 64), (32, 64), 131072),
                                    ((1, 3, 5), (7, 5, 11), (7, 3, 11), 2310)]:
            r = self.replay(matrix_graph(a, b, out))
            self.assertEqual(r["rows"][0]["matrix"]["shape_flops"], expected)
            self.assertFalse(r["rows"][0]["legacy_matrix_agrees"])
            self.assertFalse(r["exact_total_allowed"])

    def test_unknown_and_generic_never_count_as_zero(self):
        for name in ("ttnn.unknown_op", "ttnn.generic_op"):
            r = self.replay([node(0, "function_start", name), node(1, "function_end", name)])
            self.assertEqual(r["rows"][0]["status"], "uncounted")
            self.assertIsNone(r["rows"][0]["matrix"])
            self.assertIsNone(r["exact_total_flops"])
            self.assertFalse(r["exact_total_allowed"])

    def test_opaque_rank2_is_not_a_matrix_identity(self):
        r = audit.classify("ttnn.generic_op", {(0, (32, 32)), (1, (32, 32))}, [], [], {})
        self.assertIsNone(r["matrix"])
        self.assertEqual(r["status"], "uncounted")

    def test_reductions_transcendentals_and_inplace_not_one_flop(self):
        for name in ("ttnn.layer_norm", "ttnn.softmax", "ttnn.cos", "ttnn.multiply_"):
            r = self.replay([node(0, "function_start", name), node(1, "function_end", name)])
            self.assertEqual(r["rows"][0]["status"], "symbolic")
            self.assertFalse(r["exact_total_allowed"])

    def test_nested_calls_not_counted_twice(self):
        r = self.replay([node(0, "function_start", "ttnn.outer"),
                         node(1, "function_start", "ttnn.inner"),
                         node(2, "function_end", "ttnn.inner"),
                         node(3, "function_end", "ttnn.outer")])
        self.assertEqual(r["operations"], 1)
        self.assertEqual(list(r["classes"]), ["ttnn.outer"])

    def test_arithmetic_child_cannot_hide_under_free_parent(self):
        r = self.replay([node(0, "function_start", "ttnn.reshape"),
                         node(1, "function_start", "ttnn.generic_op"),
                         node(2, "function_end", "ttnn.generic_op"),
                         node(3, "function_end", "ttnn.reshape")])
        self.assertEqual(r["operations"], 1)
        self.assertFalse(r["exact_total_allowed"])

    def test_missing_and_misnamed_ends_refuse_exactness(self):
        for nodes in ([node(0, "function_start", "ttnn.reshape")],
                      [node(0, "function_start", "ttnn.reshape"),
                       node(1, "function_end", "ttnn.wrong")]):
            r = self.replay(nodes)
            self.assertEqual(r["ownership"], "unverified_legacy_spans")
            self.assertFalse(r["exact_total_allowed"])

    def test_empty_and_duplicate_graphs_rejected(self):
        for nodes in ([], [node(0, "function_start", "ttnn.reshape")] * 2):
            with self.assertRaises(ValueError):
                self.replay(nodes)
        with self.assertRaises(ValueError):
            audit.build_report([Path(__file__), Path(__file__)])

    def test_missing_inputs_and_ambiguous_outputs(self):
        graph = matrix_graph((3, 5), (5, 7), (3, 7))
        graph[1]["input_tensors"] = []
        r = self.replay(graph)
        self.assertEqual(r["rows"][0]["status"], "uncounted")
        by_id = {1: {"params": {"shape": "Shape([3, 5])"}},
                 2: {"params": {"shape": "Shape([5, 3])"}}}
        with self.assertRaises(ValueError):
            audit.matrix_candidates({"input_tensors": [1, 2]}, by_id, [(3, 3), (5, 5)])

    def test_zero_is_only_allowed_for_named_non_arithmetic_convention(self):
        r = self.replay([node(0, "function_start", "ttnn.reshape"),
                         node(1, "function_end", "ttnn.reshape")])
        self.assertTrue(r["exact_total_allowed"])
        self.assertEqual(r["exact_total_flops"], 0)
        self.assertIsNone(r["rows"][0]["machine_instructions"])

    def test_real_dense_controls(self):
        for p in ["perf/roof_budget/control_matmul_8192.json", "perf/c10_flop_contract/control_8192.json"]:
            r = audit.audit_capture(audit.ROOT / p)
            self.assertEqual(r["operations"], 1)
            self.assertEqual(r["rows"][0]["matrix"]["shape_flops"], 1099511627776)
            self.assertTrue(r["rows"][0]["legacy_matrix_agrees"])

    def test_real_missing_transposed_and_scalar_matmul(self):
        r = audit.audit_capture(audit.ROOT / "perf/roof_budget/captures/cap_DiffusionModule__.json.gz")
        rows = {x["counter"]: x for x in r["rows"]}
        self.assertEqual(rows[2464]["matrix"]["shape_flops"], 3523215360)
        self.assertEqual(rows[2510]["matrix"]["shape_flops"], 512)
        self.assertEqual(rows[2510]["legacy"]["logical"], 131072)
        self.assertFalse(r["exact_total_allowed"])

    def test_legacy_negative_controls_are_reported_as_stop(self):
        cases = audit.legacy_controls()
        self.assertEqual([x["passed"] for x in cases], [True, False, False, False, False])
        for c in cases[1:]:
            self.assertNotEqual(c["expected_matrix_shape_flops"], c["legacy_shape_flops"])

    def test_cli_refuses_exact_total(self):
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run([sys.executable, str(Path(audit.__file__)),
                str(audit.ROOT / "perf/c10_flop_contract/control_8192.json"),
                "--require-exact-total", "--out", str(Path(td) / "audit.json")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("REFUSED", result.stderr)


if __name__ == "__main__":
    unittest.main()
