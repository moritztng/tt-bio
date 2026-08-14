#!/usr/bin/env python3
"""Decompose the shipped fp32-host affinity trunk wall: reference MSAModule vs
reference PairformerModule totals across the 6 recycling iterations, measured on the
real predict path (monkeypatched timers, no repo edits). Also prints total wall.
"""
import runpy
import sys
import time

import tt_bio.reference as ref

_t = {"msa": 0.0, "pairformer": 0.0, "t0": time.perf_counter()}
_orig_pf = ref.PairformerModule.forward
_orig_msa = ref.MSAModule.forward


def pf_fwd(self, *a, **k):
    t0 = time.perf_counter()
    out = _orig_pf(self, *a, **k)
    _t["pairformer"] += time.perf_counter() - t0
    return out


def msa_fwd(self, *a, **k):
    t0 = time.perf_counter()
    out = _orig_msa(self, *a, **k)
    _t["msa"] += time.perf_counter() - t0
    return out


ref.PairformerModule.forward = pf_fwd
ref.MSAModule.forward = msa_fwd

import atexit


@atexit.register
def report():
    wall = time.perf_counter() - _t["t0"]
    print(f"HOSTSPLIT msa={_t['msa']:.2f}s pairformer={_t['pairformer']:.2f}s "
          f"wall={wall:.1f}s", flush=True)


sys.argv = [
    "tt-bio", "predict", "examples/affinity_fkg.yaml", "--model", "boltz2",
    "--single_sequence", "--override", "--affinity_mw_correction", "--debug",
    "--recycling_steps", "1", "--sampling_steps", "10", "--diffusion_samples", "1",
    "--sampling_steps_affinity", "10", "--diffusion_samples_affinity", "1",
    "--out_dir", "/tmp/hostsplit_out",
]
runpy.run_module("tt_bio.main", run_name="__main__")
