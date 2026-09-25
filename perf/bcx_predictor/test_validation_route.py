#!/usr/bin/env python3
"""A checkpoint the card does not hold folds on BindCraft 2's own trunk. Card-free.

The shipped `examples/pdl1.json` makes all five `model_*_multimer_v3` checkpoints design
models, so BindCraft 2 has nothing left for a held-out validation set and
`campaign.py:77-83` moves validation to the monomer pool `('model_1_ptm', 'model_2_ptm')`
(`bindcraft/af2.py:420`). The device pool holds the five multimer trunks, and
`multimer_pool.use` refuses an unknown name -- correctly, since folding a design on the
wrong checkpoint is worse than stopping. The consequence was that NO device arm could
accept a binder on the shipped example whatever its quality: `bcx-accept`'s arm passed all
five design stages at i_pTM 0.85 / pLDDT 0.95, produced its 10 MPNN redesigns and died in
`predict_validation_ensemble` (`state/bcx/VALIDATION-POOL.md`).

No card is needed to prove the route, and that is the point: the fold under test must NOT
touch the device, so `Tripwire` below raises on any use of it. The fold is a real
`model_1_ptm` fold of the real PD-L1 design state through BindCraft 2's own JAX trunk, with
the splice installed exactly as an arm installs it.

Run: python3 perf/bcx_predictor/test_validation_route.py     (~2 min, no device)
"""
import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"),
           str(ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import bc2_state as B                                      # noqa: E402  BC2 onto the path
from multimer_pool import POOL, MultimerPool               # noqa: E402
from splice import EvoformerOnDevice, evoformer_on_device   # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
ap.add_argument("--binder", type=int, default=73, help="the length bcx-accept's arm died on")
ap.add_argument("--bucket", type=int, default=32)
args = ap.parse_args()

VALIDATION_MODEL = "model_1_ptm"       # the name the arm died asking for
DESIGN_MODEL = "model_2_multimer_v3"


class Tripwire:
    """A device that is not there. Any use of it fails the test rather than hanging.

    `afgrad.Dev` carries no `models`, so that one name answers as a plain `Dev` does --
    which is what lets `EvoformerOnDevice` ask a pool to name its own checkpoints and fall
    back to the caller for a single-checkpoint trunk.
    """

    def __getattr__(self, name):
        if name == "models":
            raise AttributeError(name)
        raise AssertionError(f"the card was touched: Dev.{name}")


class FakePool:
    """Records which trunk the splice would have been pointed at. Loads no weights."""

    def __init__(self):
        self.models = POOL
        self.used = []

    def holds(self, name):
        return name in self.models

    def use(self, name):
        if not self.holds(name):
            raise KeyError(f"{name!r} is not in the pool {self.models}")
        self.used.append(name)


def predictor(pool, presets, recycles=0):
    m = TTBioAlphaFoldDesignModel(presets=presets, data_dir=args.params, models=presets,
                                  num_recycle=recycles, length_bucket_size=args.bucket,
                                  max_cache_size=2, trunk="device", pool=pool)
    m.dropout = False                  # campaign.py:265 builds the validation model this way
    return m


settings = B.campaign_settings(overrides=[f"length_bucket_size={args.bucket}",
                                          "campaign_seed=0",
                                          f"binder_lengths=[{args.binder}]"])
_ds, states, _losses = B.design_state(settings)
n_res = sum(sum(c.values()) for c in B.state_shape(states).values())
print(f"PD-L1 design state: binder {args.binder}, {n_res} residues, "
      f"padded {(n_res + args.bucket - 1) // args.bucket * args.bucket}", flush=True)

failures = []

# ------------------------------------------------------------------ 1. the defect, on purpose
# `checkpoints=()` is the pre-fix splice: it does not know what the card holds, so it claims
# every fold and the pool's refusal is reached. Without this leg a green leg 2 could just mean
# the pool is never consulted.
real_pool = MultimerPool(args.params, resident=1, verbose=False)   # files checked, none loaded
with evoformer_on_device(EvoformerOnDevice(Tripwire(), k_evo=48, checkpoints=())):
    try:
        predictor(real_pool, (VALIDATION_MODEL,)).predict(states, model=VALIDATION_MODEL)
        failures.append("the unrouted splice did not raise, so leg 2 proves nothing")
        witness = None
    except KeyError as exc:
        witness = str(exc)
        print(f"unrouted   : KeyError {witness}", flush=True)
    except AssertionError as exc:
        failures.append(f"the unrouted splice reached the card before the pool refused: {exc}")
        witness = None

# ------------------------------------------------------------------ 2. the route
evo = EvoformerOnDevice(Tripwire(), k_evo=48, checkpoints=POOL)
t0 = time.time()
with evoformer_on_device(evo) as swapped:
    pred = predictor(real_pool, (VALIDATION_MODEL,)).predict(states, model=VALIDATION_MODEL)
secs = time.time() - t0

target = next(iter(pred))
metrics = pred[target].metrics
plddt = np.asarray(metrics["plddt"])
ptm = float(metrics["ptm"])
print(f"routed     : {VALIDATION_MODEL} folded on BindCraft 2's JAX trunk in {secs:.1f}s -- "
      f"pTM {ptm:.4f}, pLDDT mean {plddt.mean():.4f}, {len(metrics)} metrics", flush=True)
print(f"host_folds : {evo.host_folds}", flush=True)

if not np.isfinite(plddt).all() or not np.isfinite(ptm):
    failures.append(f"the routed fold returned non-finite metrics: pTM {ptm}")
if not 0.0 < plddt.mean() <= 1.0:
    failures.append(f"pLDDT mean {plddt.mean()} is not a confidence")
if plddt.shape[0] != n_res:
    failures.append(f"pLDDT covers {plddt.shape[0]} residues, not the complex's {n_res}")
if swapped:
    failures.append(f"the card's stack was spliced into a fold it cannot run: {swapped}")
if evo.host_folds != {VALIDATION_MODEL: 1}:
    failures.append(f"host_folds did not record the route: {evo.host_folds}")
if evo.calls != {"primal": 0, "taped": 0, "backward": 0}:
    failures.append(f"the card ran something: {evo.calls}")

# ------------------------------------------------------------------ 3. the card still wins its own
# A route that sends everything to the host is not a fix, it is a removal.
fake = FakePool()
card_evo = EvoformerOnDevice(fake, k_evo=48)          # checkpoints inferred from the pool
if card_evo.checkpoints != POOL:
    failures.append(f"a pool did not answer for itself: {card_evo.checkpoints}")
with evoformer_on_device(card_evo):
    m = predictor(fake, (DESIGN_MODEL,))
    with m._route(DESIGN_MODEL):
        pass
    if fake.used != [DESIGN_MODEL]:
        failures.append(f"a resident checkpoint was not sent to the card: {fake.used}")
    if card_evo.host_folds:
        failures.append(f"a resident checkpoint was routed off card: {card_evo.host_folds}")
    with m._route(VALIDATION_MODEL):
        pass
    if fake.used != [DESIGN_MODEL]:
        failures.append(f"an off-card checkpoint still reached the pool: {fake.used}")
print(f"card side  : {fake.used} on card, {card_evo.host_folds} on host", flush=True)

if witness is None and not failures:
    failures.append("no witness for the defect")
if failures:
    print("FAIL: " + "; ".join(failures))
    sys.exit(1)
print("PASS")
