"""Known-answer timeline controls, with no device import or hardware measurements."""
import copy
from collections import Counter
import random
import unittest

from reconcile_cycles import reconcile


def capture(*spans):
    return {
        "device": "synthetic-device",
        "timebase": "synthetic-synchronized-ticks",
        "start_tick": 0,
        "end_tick": 100,
        "programs": [
            {"id": str(i), "op_class": kind, "start_tick": start, "end_tick": end,
             "device": "synthetic-device", "timebase": "synthetic-synchronized-ticks"}
            for i, (kind, start, end) in enumerate(spans)
        ],
    }


class TimelineTests(unittest.TestCase):
    def test_serial_classes_and_exposed_gaps_close(self):
        result = reconcile(capture(("matmul", 10, 40), ("copy", 40, 70)))
        self.assertEqual(result["exclusive_ticks_by_class"], {"copy": 30, "matmul": 30})
        self.assertEqual(result["idle_or_unobserved_ticks"], 40)
        self.assertEqual(result["overlap_ticks"], 0)
        self.assertEqual(result["closure_ticks"], 0)

    def test_overlap_does_not_become_fold_latency(self):
        result = reconcile(capture(("matmul", 10, 60), ("copy", 40, 80)))
        self.assertEqual(result["sum_program_ticks"], 90)
        self.assertEqual(result["covered_ticks"], 70)
        self.assertEqual(result["overlap_ticks"], 20)
        self.assertEqual(result["exclusive_ticks_by_class"], {"copy": 20, "matmul": 30})
        self.assertEqual(result["idle_or_unobserved_ticks"], 30)

    def test_same_class_overlap_is_still_overlap(self):
        result = reconcile(capture(("matmul", 0, 100), ("matmul", 20, 80)))
        self.assertEqual(result["sum_program_ticks"], 160)
        self.assertEqual(result["covered_ticks"], 100)
        self.assertEqual(result["overlap_ticks"], 60)
        self.assertEqual(result["exclusive_ticks_by_class"], {"matmul": 40})
        self.assertEqual(result["peak_concurrent_programs"], 2)

    def test_empty_capture_is_all_unobserved(self):
        result = reconcile(capture())
        self.assertEqual(result["covered_ticks"], 0)
        self.assertEqual(result["idle_or_unobserved_ticks"], 100)

    def test_matches_independent_tick_by_tick_oracle(self):
        rng = random.Random(0)
        for _ in range(100):
            spans = []
            for _ in range(rng.randrange(12)):
                start, end = sorted(rng.sample(range(101), 2))
                spans.append((rng.choice(("copy", "matmul")), start, end))
            exclusive, overlap, gaps = Counter(), 0, 0
            for t in range(100):
                active = [kind for kind, a, b in spans if a <= t < b]
                if len(active) == 1:
                    exclusive[active[0]] += 1
                elif active:
                    overlap += 1
                else:
                    gaps += 1
            result = reconcile(capture(*spans))
            self.assertEqual(result["exclusive_ticks_by_class"], dict(exclusive))
            self.assertEqual(result["overlap_ticks"], overlap)
            self.assertEqual(result["idle_or_unobserved_ticks"], gaps)

    def test_large_timestamp_offsets_preserve_integer_cycles(self):
        data = capture(("copy", 1, 3))
        for row in [data, *data["programs"]]:
            row["start_tick"] += 2**60
            row["end_tick"] += 2**60
        self.assertEqual(reconcile(data)["exclusive_ticks_by_class"], {"copy": 2})

    def test_rejects_invalid_or_mixed_evidence(self):
        original = capture(("matmul", 10, 40), ("copy", 40, 70))
        changes = [("id", "0"), ("device", "other-device"), ("timebase", "other-clock"),
                   ("start_tick", -1), ("end_tick", 101), ("end_tick", 40),
                   ("start_tick", 40.5), ("start_tick", True), ("op_class", "")]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(original)
                data["programs"][1][key] = value
                with self.assertRaises(ValueError):
                    reconcile(data)


if __name__ == "__main__":
    unittest.main()
