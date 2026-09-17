#!/usr/bin/env python3
"""Controls for the node-parameterised validator. No device is opened and no clock is read."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import control                                                                # noqa: E402
import node_control as N                                                      # noqa: E402

GOOD = {"containment": "active", "module_srcversion": "A10759A24565BC5BBE903C5",
        "holders": [], "own_nodes": []}


def snap(**kw):
    return {**GOOD, **kw}


FAKE = {100: 1, 200: 100, 300: 1, 400: 300}   # 200 is a fork of 100; 400 descends from 300


def fake_stat(pid):
    if pid not in FAKE:
        raise OSError(2, "no such pid")
    return "%d (python3) S %d 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" % (pid, FAKE[pid])


class Ancestry(unittest.TestCase):
    def test_a_fork_of_me_resolves_to_me(self):
        self.assertIn(100, N.ancestry(200, read=fake_stat))

    def test_a_foreign_process_does_not(self):
        self.assertNotIn(100, N.ancestry(400, read=fake_stat))

    def test_a_cycle_or_missing_parent_terminates(self):
        self.assertEqual(N.ancestry(999, read=fake_stat), [999])

    def test_my_own_pid_is_in_my_ancestry(self):
        self.assertIn(os.getpid(), N.ancestry(os.getpid()))

    def test_foreign_filters_the_fork_and_keeps_the_co_tenant(self):
        rows = [{"pid": 200, "nodes": ["/dev/tenstorrent/1"]},
                {"pid": 400, "nodes": ["/dev/tenstorrent/1"]},
                {"pid": 400, "nodes": ["/dev/tenstorrent/3"]}]
        got = N.foreign(rows, 1, 100, read=fake_stat)
        self.assertEqual([g["pid"] for g in got], [400])


class Retry(unittest.TestCase):
    def test_a_transient_foreign_holder_is_not_a_refusal(self):
        first = snap(holders=[{"pid": os.getpid() + 1, "nodes": ["/dev/tenstorrent/1"]}])
        N.validate(first, 1, resample=lambda: snap(), pause=0.0)

    def test_a_persistent_foreign_holder_is_a_refusal(self):
        busy = snap(holders=[{"pid": 1, "nodes": ["/dev/tenstorrent/1"]}])
        with self.assertRaises(RuntimeError):
            N.validate(busy, 1, resample=lambda: busy, pause=0.0)

    def test_the_resample_is_actually_taken(self):
        calls = []

        def again():
            calls.append(1)
            return snap(holders=[{"pid": 1, "nodes": ["/dev/tenstorrent/1"]}])
        with self.assertRaises(RuntimeError):
            N.validate(snap(holders=[{"pid": 1, "nodes": ["/dev/tenstorrent/1"]}]), 1,
                       resample=again, pause=0.0)
        self.assertEqual(len(calls), 2)


class Validate(unittest.TestCase):
    def test_agrees_with_the_pinned_helper_on_node_zero(self):
        cases = [snap(),
                 snap(containment="inactive"),
                 snap(module_srcversion="28CFF5A6678E4F2D87F6383"),
                 snap(holders=[{"pid": os.getpid() + 1, "nodes": ["/dev/tenstorrent/0"]}]),
                 snap(holders=[{"pid": os.getpid(), "nodes": ["/dev/tenstorrent/0"]}])]
        for s in cases:
            for opened in (False, True):
                if opened:
                    s = {**s, "own_nodes": ["/dev/tenstorrent/0"]}
                a = b = None
                try:
                    control.validate_snapshot(s, opened=opened)
                except RuntimeError as e:
                    a = type(e)
                try:
                    N.validate(s, 0, opened=opened, resample=lambda s=s: s, pause=0.0)
                except RuntimeError as e:
                    b = type(e)
                self.assertEqual(a, b, "disagreement on %s opened=%s" % (s, opened))

    def test_node_one_accepts_a_node_zero_holder(self):
        s = snap(holders=[{"pid": 1, "nodes": ["/dev/tenstorrent/0"]}])
        N.validate(s, 1, resample=lambda: s, pause=0.0)
        with self.assertRaises(RuntimeError):
            control.validate_snapshot(s)

    def test_node_one_rejects_a_node_one_holder(self):
        s = snap(holders=[{"pid": 1, "nodes": ["/dev/tenstorrent/1"]}])
        with self.assertRaises(RuntimeError):
            N.validate(s, 1, resample=lambda: s, pause=0.0)

    def test_opened_on_the_wrong_node_is_refused(self):
        with self.assertRaises(RuntimeError):
            N.validate(snap(own_nodes=["/dev/tenstorrent/0"]), 1, opened=True, pause=0.0)

    def test_opened_on_more_than_the_assigned_node_is_refused(self):
        with self.assertRaises(RuntimeError):
            N.validate(snap(own_nodes=["/dev/tenstorrent/0", "/dev/tenstorrent/1"]), 1,
                       opened=True, pause=0.0)

    def test_stock_driver_is_refused_on_any_node(self):
        for n in (0, 1):
            with self.assertRaises(RuntimeError):
                N.validate(snap(module_srcversion="28CFF5A6678E4F2D87F6383"), n, pause=0.0)

    def test_sampler_reads_the_assigned_nodes_attribute(self):
        self.assertTrue(str(N.aiclk_path(1)).endswith("tenstorrent!1/tt_aiclk"))
        self.assertNotEqual(N.aiclk_path(0), N.aiclk_path(1))

    def test_foreign_holders_reports_any_node_not_just_the_assigned_one(self):
        rows = [{"monotonic_ns": 1, "holders": [{"pid": 7, "nodes": ["/dev/tenstorrent/3"]}]},
                {"monotonic_ns": 2, "holders": [{"pid": 99, "nodes": ["/dev/tenstorrent/1"]}]}]
        got = N.foreign_holders(rows, 1, 99)
        self.assertEqual([g["monotonic_ns"] for g in got], [1])

    def test_foreign_holders_excludes_my_own_fork(self):
        rows = [{"monotonic_ns": 1,
                 "holders": [{"pid": os.getpid(), "nodes": ["/dev/tenstorrent/1"]}]}]
        self.assertEqual(N.foreign_holders(rows, 1, os.getpid()), [])

    def test_a_quarantined_node_reports_its_read_error_rather_than_a_clock(self):
        ident = N.identity(0)
        self.assertIn("tt_aiclk", ident)


class Coverage(unittest.TestCase):
    """The pinned gate is reused unchanged; these pin its behaviour for this row's evidence."""

    def _iv(self, a, b):
        return {"start_monotonic_ns": a, "end_monotonic_ns": b}

    def test_pinned_gate_accepts_a_clean_pinned_interval(self):
        s = [{"read_start_ns": 10 + i * 1_000_000, "read_end_ns": 11 + i * 1_000_000,
              "MHz": 1350} for i in range(50)]
        self.assertTrue(control.coverage(s, self._iv(0, 50_000_000))["pass"])

    def test_pinned_gate_rejects_a_clamped_chip(self):
        s = [{"read_start_ns": 10 + i * 1_000_000, "read_end_ns": 11 + i * 1_000_000,
              "MHz": 800} for i in range(50)]
        self.assertFalse(control.coverage(s, self._iv(0, 50_000_000))["pass"])

    def test_pinned_gate_rejects_a_read_error(self):
        s = [{"read_start_ns": 10 + i * 1_000_000, "read_end_ns": 11 + i * 1_000_000,
              "MHz": 1350} for i in range(50)]
        s.append({"read_start_ns": 20_000_000, "read_end_ns": 20_000_100,
                  "error": "OSError(19, 'No such device')"})
        self.assertFalse(control.coverage(s, self._iv(0, 50_000_000))["pass"])

    def test_pinned_gate_rejects_a_gap_above_ten_ms(self):
        s = [{"read_start_ns": 10, "read_end_ns": 11, "MHz": 1350},
             {"read_start_ns": 40_000_000, "read_end_ns": 40_000_100, "MHz": 1350},
             {"read_start_ns": 49_000_000, "read_end_ns": 49_000_100, "MHz": 1350}]
        self.assertFalse(control.coverage(s, self._iv(0, 50_000_000))["pass"])

    def test_pinned_gate_rejects_a_pre_interval_only_window(self):
        s = [{"read_start_ns": i * 1000, "read_end_ns": 1 + i * 1000, "MHz": 1350}
             for i in range(50)]
        self.assertFalse(control.coverage(s, self._iv(10_000_000, 60_000_000))["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
