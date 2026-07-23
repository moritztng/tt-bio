"""p24: wall-clock breakdown of ONE real RFD3DiffusionModule.__call__ (n_recycle=2) by
component, via monkey-patched timers around encoder/decoder/diffusion_transformer/
diffusion_token_encoder calls. Companion to the spike_*_trace.py isolated-component
benchmarks: tells us how much of the real 154.5ms/step budget each traced-or-not
component actually accounts for, so p25 can prioritize wiring effort by real payoff
rather than guessing from the isolated-shape spikes alone.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/profile_step_breakdown.py
"""
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification
from tt_bio.rfd3_sampler import RFD3Sampler
import ttnn

PDB = os.path.join(os.path.dirname(__file__), "parity_artifacts", "iai_protein", "IAI_protein.pdb")
CONTIG = "A1-10,20,A31-40"
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")

totals = defaultdict(float)
counts = defaultdict(int)


def timed(name, fn):
    # These __call__s all end in ttnn.to_torch(), which blocks until the host
    # copy lands, so no extra ttnn.synchronize_device() is needed to get a
    # real wall-clock measurement.
    def wrapped(*a, **kw):
        t0 = time.time()
        out = fn(*a, **kw)
        dt = time.time() - t0
        totals[name] += dt
        counts[name] += 1
        return out
    return wrapped


def main():
    spec = InputSpecification.from_dict({"input": PDB, "contig": CONTIG})
    spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}

    ti_weights = torch.load(os.path.join(GOLDEN_DIR, "token_initializer.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dm_weights = torch.load(os.path.join(GOLDEN_DIR, "diffusion_module.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_weights)
    dev_dm = build_diffusion_module(dm_weights)

    # Monkey-patch the sub-component __call__s we care about. `__call__` is a
    # dunder: CPython looks it up on the TYPE for implicit `obj(...)` invocation,
    # so an instance-attribute override is silently ignored -- must patch the
    # class instead (safe here: single short-lived process, single instance).
    type(dev_dm.encoder).__call__ = timed("encoder", type(dev_dm.encoder).__call__)
    type(dev_dm.decoder).__call__ = timed("decoder", type(dev_dm.decoder).__call__)
    type(dev_dm.diffusion_transformer).__call__ = timed("dit", type(dev_dm.diffusion_transformer).__call__)
    type(dev_dm.diffusion_token_encoder).__call__ = timed(
        "diffusion_token_encoder", type(dev_dm.diffusion_token_encoder).__call__)
    dev_dm._downcast_c = timed("downcast_c", dev_dm._downcast_c)
    dev_dm._downcast_q = timed("downcast_q", dev_dm._downcast_q)

    coord0 = f["motif_pos"].float().unsqueeze(0)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
        g = torch.Generator().manual_seed(42)
        sampler = RFD3Sampler(num_timesteps=12)  # a handful of steps for a stable average
        t0 = time.time()
        X, _ = sampler.sample(dev_dm, 1, f["ref_pos"].shape[0], coord0, f, init,
                               f["is_motif_atom_with_fixed_coord"], generator=g)
        total = time.time() - t0

    n_steps = counts["dit"] // 2  # dit runs once per recycle (2/step); use as the step count denominator
    n_steps = max(n_steps, 1)
    print(f"[total] {total*1000:.1f}ms over {n_steps} steps -> {total*1000/n_steps:.2f}ms/step (this run)")
    print(f"{'component':<26}{'total_ms':>10}{'calls':>8}{'ms/call':>10}{'%_of_total':>12}")
    accounted = 0.0
    for name in ("encoder", "decoder", "dit", "diffusion_token_encoder", "downcast_c", "downcast_q"):
        tot = totals[name] * 1000
        n = counts[name]
        accounted += tot
        print(f"{name:<26}{tot:>10.1f}{n:>8}{(tot/n if n else 0):>10.3f}{100*tot/(total*1000):>11.1f}%")
    print(f"{'[unaccounted glue]':<26}{total*1000 - accounted:>10.1f}{'':>8}{'':>10}{100*(total*1000-accounted)/(total*1000):>11.1f}%")


if __name__ == "__main__":
    main()
