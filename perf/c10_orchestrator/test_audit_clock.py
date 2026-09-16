"""CPU regressions for evidence rejection and the published model arithmetic."""
import copy
import json
import unittest

import audit_clock


class ClockAuditTests(unittest.TestCase):
    def setUp(self):
        self.baseline = audit_clock.HERE / "pinned_baseline.json"
        self.data = json.loads(self.baseline.read_text())

    def test_recorded_baseline_and_model_targets(self):
        result = audit_clock.audit(
            self.baseline,
            audit_clock.REPO / "perf/b2z2_cell_recheck/out/cell_main_qb2c1_clock.json",
            1350, 10,
        )
        forced = result["baseline"]["arms"]["force"]
        self.assertEqual(forced["n"], 8)
        self.assertEqual(forced["foreign_holder_folds"], 0)
        self.assertAlmostEqual(forced["median_fold_s"], 14.554)
        self.assertEqual(forced["sampled_clock_min_mhz"], 1350)
        self.assertTrue(result["baseline"]["matches_reference_bytes"])
        self.assertAlmostEqual(result["historical_fit"]["two_clock_extremes"]["intercept_s"], 2.900554787)
        self.assertAlmostEqual(result["conditional_targets"]["scenarios"][0]["projected_work_mcycles"], 9583.65)

    def test_rejects_missing_clock_sag_holder_and_empty_samples(self):
        for key, value in (("aiclk_mean", None), ("aiclk_min", 800),
                           ("foreign_tt", [{"pid": 1}]), ("aiclk_n", 0)):
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                row = next(r for r in data["runs"] if not r["warmup"] and r["arm"] == "force")
                if value is None:
                    del row[key]
                else:
                    row[key] = value
                with self.assertRaises(ValueError):
                    audit_clock.validate_baseline(data)

    def test_rejects_clamped_control_arm(self):
        row = next(r for r in self.data["runs"] if not r["warmup"] and r["arm"] == "base")
        row["aiclk_min"] = 800
        with self.assertRaisesRegex(ValueError, "clock artifact"):
            audit_clock.validate_baseline(self.data)


if __name__ == "__main__":
    unittest.main()
