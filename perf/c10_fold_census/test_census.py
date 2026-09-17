#!/usr/bin/env python3
"""CPU controls for the census arm builder and its two counters. No device, no clock, no timing."""
from __future__ import annotations

import json
import sys
import unittest
from math import prod
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PERF = HERE.parent

import census as C                                                            # noqa: E402


class KnownAnswer(unittest.TestCase):
    def test_cube_flops_exact(self):
        self.assertEqual(C.matmul_flops([8192, 8192], 8192), 1_099_511_627_776)

    def test_cube_bytes_exact(self):
        self.assertEqual(3 * C.tensor_bytes([8192, 8192]), 402_653_184)

    def test_cube_control_passes(self):
        c = C.cube_control()
        self.assertTrue(c["flops_pass"] and c["bytes_pass"])

    def test_a_counter_off_by_one_output_tensor_fails(self):
        """roof-reset's exact defect: 4 tensors instead of 3, 536,870,912 against 402,653,184."""
        bad = 4 * C.tensor_bytes([8192, 8192])
        self.assertEqual(bad, 536_870_912)
        self.assertNotEqual(bad, C.CUBE_BYTES)

    def test_fp32_dtype_would_fail_the_byte_control(self):
        self.assertNotEqual(3 * 8192 * 8192 * 4, C.CUBE_BYTES)

    def test_flop_counter_rejects_the_one_sided_convention(self):
        self.assertNotEqual(prod([8192, 8192]) * 8192, C.CUBE_FLOPS)


class Tiles(unittest.TestCase):
    def test_matches_census_convention_on_pair_tensor(self):
        self.assertEqual(C.tiles([1, 512, 512, 128]), 512 * 16 * 4)

    def test_rank_one_rounds_up(self):
        self.assertEqual(C.tiles([768]), 24)
        self.assertEqual(C.tiles([3]), 1)

    def test_empty_is_zero(self):
        self.assertEqual(C.tiles([]), 0)

    def test_unaligned_rows_round_up_not_down(self):
        self.assertEqual(C.tiles([1, 129, 32, 128]), 129 * 1 * 4)


class Operands(unittest.TestCase):
    """Every derived operand set must equal what the census actually recorded for that shape."""

    @classmethod
    def setUpClass(cls):
        cls.shapes, cls.census = C.load(PERF)

    def _recorded(self, name, out):
        want = "x".join(str(d) for d in out)
        hits = {}
        for key, e in self.census["top_shapes"].items():
            op, o, i = key.split("|")
            if op == "ttnn." + name and o == "out=" + want:
                hits[i[len("in="):]] = e
        return hits

    def test_linear_operands_match_a_recorded_signature(self):
        checked = 0
        for row in self.shapes:
            if row["arm"] != "linear" or not row["K"]:
                continue
            ops = C._operands("linear", row["shape"], row["K"])
            derived = {"x".join(str(d) for d in s) for _n, s in ops}
            # only a recorded signature that actually contracts over this key's K is comparable;
            # two keys can share an output shape and differ only in K, and the census kept the
            # top 60 signatures by calls, so the other K need not appear at all.
            hits = [sig for sig in self._recorded("linear", row["shape"])
                    if derived <= set(sig.split(","))]
            if not hits:
                self.assertFalse(
                    any(set(sig.split(",")) >= {"x".join(str(d) for d in ops[1][1])}
                        for sig in self._recorded("linear", row["shape"])),
                    "weight shape recorded but activation derived wrong for %s" % (row,))
                continue
            checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_inplace_operands_match_a_recorded_signature(self):
        checked = 0
        for row in self.shapes:
            if row["arm"] not in C.INPLACE_ARMS:
                continue
            hits = self._recorded(row["arm"], [])
            want = "x".join(str(d) for d in row["shape"])
            if not any(sig == want + "," + want for sig in hits):
                continue
            ops = C._operands(row["arm"], row["shape"], None)
            self.assertEqual([s for _n, s in ops], [row["shape"], row["shape"]])
            checked += 1
        self.assertGreaterEqual(checked, 5)

    def test_matmul_inner_dim_contracts(self):
        ops = C._operands("matmul", [1, 128, 512, 512], 512)
        (_, a), (_, b) = ops
        self.assertEqual(a[-1], b[-2])
        self.assertEqual([a[0], a[1], a[-2], b[-1]], [1, 128, 512, 512])

    def test_inplace_bytes_count_the_destination_twice(self):
        n = C.tensor_bytes([1, 512, 768])
        self.assertEqual(C._bytes("add_", [1, 512, 768],
                                  C._operands("add_", [1, 512, 768], None)), 3 * n)

    def test_linear_bytes_count_activation_weight_and_output(self):
        out, k = [1, 512, 1536], 768
        ops = C._operands("linear", out, k)
        self.assertEqual(C._bytes("linear", out, ops),
                         2 * (512 * 768 + 768 * 1536 + 512 * 1536))

    def test_build_refuses_a_matmul_key_with_no_K(self):
        arms, refused = C.build_arms([{"arm": "matmul", "shape": [1, 768, 512], "K": None,
                                       "calls": 200.0}])
        self.assertEqual(arms, [])
        self.assertIn("no recorded K", refused[0]["reason"])

    def test_build_refuses_an_unmodelled_class(self):
        arms, refused = C.build_arms([{"arm": "permute", "shape": [1, 1, 512, 768], "K": None,
                                       "calls": 5000.0}])
        self.assertEqual(arms, [])
        self.assertIn("no operand model", refused[0]["reason"])

    def test_matmul_class_flops_are_reproducible_from_the_keys(self):
        arms, _ = C.build_arms(self.shapes, include=C.MATMUL_ARMS)
        total = sum(a["flops_per_call"] * a["calls"] for a in arms)
        self.assertAlmostEqual(total / 1e12, 131.269, places=2)
        self.assertEqual(sum(a["calls"] for a in arms), 110040)


class ByteIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _shapes, cls.census = C.load(PERF)
        cls.id = C.byte_identity(cls.census)

    def test_median_ratio_is_exactly_one(self):
        self.assertEqual(round(self.id["median_ratio"], 6), 1.0)

    def test_the_two_zero_byte_holes_are_found(self):
        keys = [h["key"] for h in self.id["holes"]]
        self.assertIn("ttnn.multiply_|out=|in=1x16x512x512,1x16x512x512", keys)
        self.assertIn("ttnn.layer_norm|out=1x140x32x128|in=1x140x32x128", keys)

    def test_real_hole_size_reproduces_the_parents_figure(self):
        self.assertAlmostEqual(self.id["real_hole_bytes"] / 1e9, 155.8, places=1)
        self.assertAlmostEqual(self.id["real_hole_pct_of_recorded"], 5.45, places=2)

    def test_exactly_seven_holes_two_of_them_real(self):
        self.assertEqual(len(self.id["holes"]), 7)
        self.assertEqual(len(self.id["real_holes"]), 2)

    def test_an_allocation_is_classified_semantic_not_a_hole(self):
        alloc = [h for h in self.id["holes"]
                 if h["key"].startswith("ttnn.allocate_tensor_on_device")]
        self.assertEqual(len(alloc), 1)
        self.assertTrue(alloc[0]["semantic"])
        self.assertGreater(alloc[0]["implied_bytes"], self.id["real_hole_bytes"])

    def test_a_corrupted_census_breaks_the_identity(self):
        bad = json.loads(json.dumps(self.census))
        for e in bad["top_shapes"].values():
            e["B"] = e["B"] * 1.3333
        r = C.byte_identity(bad)
        self.assertLess(r["median_ratio"], 0.99)

    def test_a_semantic_zero_promoted_to_real_would_be_caught(self):
        bad = json.loads(json.dumps(self.census))
        bad["top_shapes"]["ttnn.multiply|out=|in=1x2x3,1x2x3"] = {
            "calls": 10.0, "B": 0.0, "s_floor": 0.0, "in_tiles": 4.0, "out_tiles": 0.0}
        r = C.byte_identity(bad)
        self.assertEqual(len(r["real_holes"]), 3)

    def test_a_census_with_no_holes_reports_none(self):
        bad = json.loads(json.dumps(self.census))
        for e in bad["top_shapes"].values():
            if e["B"] == 0 and e["in_tiles"] + e["out_tiles"]:
                e["B"] = e["calls"] * (e["in_tiles"] + e["out_tiles"]) * C.TILE_BYTES
        self.assertEqual(C.byte_identity(bad)["holes"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
