"""Synthetic controls for identity joins and exact raw-cycle accounting."""
import unittest

from import_profiler_cycles import convert, START, END


def row(call, a=0, b=1, *, trace="", replay="", device="0", label="Matmul"):
    return {"DEVICE ID": device, "GLOBAL CALL COUNT": str(call),
            "METAL TRACE ID": trace, "METAL TRACE REPLAY SESSION ID": replay,
            "OP CODE": label, START: str(a), END: str(b),
            "DEVICE KERNEL DURATION [ns]": "deliberately unused"}


def run(ops, device=None, start=100, end=200):
    return convert(ops, ops if device is None else device, device=0,
                   timebase="synthetic-control", start_tick=start, end_tick=end)


class ProfilerImportTest(unittest.TestCase):
    def test_exact_union_join_and_explicit_boundaries(self):
        ops = [row(1, 110, 150), row(2, 140, 180, label="Unary")]
        result = run(ops, list(reversed(ops)))
        part = result["partition"]
        self.assertEqual(part["exclusive_ticks_by_class"], {"Matmul": 30, "Unary": 30})
        self.assertEqual(part["overlap_ticks"], 10)
        self.assertEqual(part["idle_or_unobserved_ticks"], 30)
        self.assertEqual(part["sum_program_ticks"], 80)
        self.assertEqual(part["closure_ticks"], 0)

    def test_replay_identity_is_not_just_call_count(self):
        result = run([row(1, 110, 120, trace="7", replay="1"),
                      row(1, 130, 140, trace="7", replay="2")])
        self.assertEqual(result["partition"]["program_count"], 2)
        self.assertEqual(result["partition"]["sum_program_ticks"], 20)

    def test_large_ticks_never_round_through_float(self):
        start = 2**60
        result = run([row(1, start+1, start+4)], start=start, end=start+5)
        self.assertEqual(result["partition"]["covered_ticks"], 3)

    def test_warmup_is_counted_but_excluded(self):
        ops = [row(1, 50, 100), row(2, 110, 150), row(3, 200, 250)]
        result = run([{"DEVICE ID": "", "OP CODE": "host"}] + ops, ops)
        self.assertEqual(result["join"]["host_only_label_rows"], 1)
        self.assertEqual(result["join"]["device_executions_outside_interval"], 2)
        self.assertEqual(result["partition"]["covered_ticks"], 40)

    def test_rejects_missing_or_ambiguous_evidence(self):
        good = row(1, 110, 150)
        cases = [([good, good], [good]), ([good], [good, good]),
                 ([good], []), ([], [good]),
                 ([dict(good, **{"OP CODE": ""})], [good]),
                 ([dict(good, **{"DEVICE ID": "1"})], [good]),
                 ([good], [dict(good, **{"DEVICE ID": "1"})]),
                 ([dict(good, **{"METAL TRACE ID": "5"})], [good]),
                 ([good], [dict(good, **{START: "1e2"})]),
                 ([good], [dict(good, **{END: "110"})]),
                 ([good], [dict(good, **{START: "99"})]),
                 ([good], [dict(good, **{END: "201"})]),
                 ([good], [dict(good, **{START: "0", END: "1"})])]
        for ops, devices in cases:
            with self.subTest(ops=ops, devices=devices), self.assertRaises(ValueError):
                run(ops, devices)


if __name__ == "__main__":
    unittest.main()
