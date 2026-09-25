#!/usr/bin/env python3
"""What trace capture takes out of a REAL BindCraft 2 gradient round.

BindCraft 2's own campaign drives: nothing here shortens a round, a recycle or a stage, and the
predictor is `perf/bcx_predictor`'s device arm unchanged. A round is the interval between two
consecutive entries into `design_model.sequence_gradients` (`bindcraft/trajectory.py:124-155`),
which is `bcx-round`'s definition, so the two rows' denominators are the same quantity.

Two modes:

  --interleave  the trace arm is switched on and off per round, alternating order, inside one
                trajectory in one process. Legal because the two arms are bit-identical: the
                trajectory sees one program.
  --arm A       one arm for the whole run, logging a digest of every callback output. Two
                processes, one per arm, compared offline: that is the bit-identity claim on the
                loop's own path.

`--rounds N` bounds collection. Every round that runs, runs in full.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack"),
          str(ROOT / "perf" / "bcx_predictor")):
    if p not in sys.path:
        sys.path.insert(0, p)

import bc2_state as B                                                  # noqa: E402
import bindcraft.campaign as campaign                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402

OUT = ROOT / "perf" / "bcx_tracewire"
MONOMER = ("model_1_ptm", "model_2_ptm")


class _Enough(BaseException):
    """Collection is done. A BaseException so BindCraft 2's own `except Exception` cannot eat it."""


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float32))
                          .tobytes()).hexdigest()[:16]


def captured(rounds, i) -> bool:
    return i > 0 and rounds[i]["captures"] > rounds[i - 1]["captures"]


def trunk_s(rounds, i) -> float:
    """The device trunk's seconds inside round i: both callbacks, host side included."""
    return sum(rounds[i]["trunk"][k] - rounds[i - 1]["trunk"][k] for k in ("taped", "backward"))


def trunk_cpu(rounds, i) -> float:
    return rounds[i]["trunk"]["cpu"] - rounds[i - 1]["trunk"]["cpu"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params",
                    help="BindCraft 2's AF2 weights DIRECTORY")
    ap.add_argument("--af2-npz", default=None,
                    help="the npz tt-bio's own trunk loads (default afgrad.DEFAULT_PARAMS)")
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--region-mb", type=int, default=768)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--interleave", action="store_true")
    ap.add_argument("--arm", choices=["trace", "eager"], default=None,
                    help="fixed arm; with --digest this is the bit-identity leg")
    ap.add_argument("--digest", action="store_true", help="log a digest of every callback output")
    ap.add_argument("--out", default=None)
    ap.add_argument("--project", default=None)
    args = ap.parse_args()
    if not args.interleave and args.arm is None:
        ap.error("pick --interleave or --arm")

    project = args.project or str(OUT / "runs" / f"round_{args.arm or 'ab'}_seed{args.seed}")
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    settings = cleaned_campaign_settings(read_settings(
        os.path.join(B.BC2, "examples", "pdl1.json"),
        parse_setting_overrides([f"campaign_seed={args.seed}", "max_trajectories=1",
                                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                                 'validation_models=["model_2_ptm"]',
                                 f"length_bucket_size={args.bucket}",
                                 f"project_folder={project}"])))
    campaign.MULTIMER_POOL = MONOMER

    import trace_wire
    trace_wire.open_traced_device(args.region_mb)
    import afgrad as A
    import stack as S
    import ttbio_predictor as T
    from splice import EvoformerOnDevice, evoformer_on_device

    lv = S.Levers()
    dm, _ = A.load_models(args.af2_npz or A.DEFAULT_PARAMS, refs=("bf16",))
    dev = A.Dev(dm.to_device())
    lv.arm("stack")
    clock = S.Clock(dt=0.25)
    evo = EvoformerOnDevice(dev, k_evo=48, trace=True)
    evo.trace_on = (args.arm != "eager")
    if args.arm == "eager":
        # The eager leg of the bit-identity pair is the shipped program: the wire exists only
        # so both legs open the device the same way, and its zeros flag goes back off.
        dev.ag.DEVICE_ZEROS = False

    # Both callbacks return host arrays, so each wall time below is the trunk step's full cost to
    # the loop: enqueue, device and readout.
    digests: list = []
    # `cpu` is the calling thread's own CPU inside both callbacks: the host enqueue work, which is
    # the floor a dtype change cannot move.
    trunk = {"taped": 0.0, "backward": 0.0, "cpu": 0.0}
    _taped, _backward = evo._taped, evo._backward

    def taped(*a):
        t, c = time.perf_counter(), time.thread_time()
        out = _taped(*a)
        trunk["taped"] += time.perf_counter() - t
        trunk["cpu"] += time.thread_time() - c
        if args.digest:
            digests.append({"op": "taped", "d": [digest(out[0]), digest(out[1])]})
        return out

    def backward(*a):
        t, c = time.perf_counter(), time.thread_time()
        out = _backward(*a)
        trunk["backward"] += time.perf_counter() - t
        trunk["cpu"] += time.thread_time() - c
        if args.digest:
            digests.append({"op": "backward", "d": [digest(out[0]), digest(out[1])]})
        return out

    evo._taped, evo._backward = taped, backward

    rounds: list = []
    marks: list = []

    def close(t_now):
        if marks:
            t0, arm, load0 = marks[-1]
            rounds.append({"i": len(rounds), "arm": arm, "s": t_now - t0,
                           "aiclk": clock.window([(t0, t_now)]), "load1": load0,
                           "span": (t0, t_now), "calls": dict(evo.calls),
                           "trunk": dict(trunk), "seg": dict(evo.wire.seg),
                           "captures": len(evo.wire.captures)})
            print(json.dumps(rounds[-1]), flush=True)
            if len(rounds) >= 3 and rounds[-1]["calls"]["backward"] == rounds[-2]["calls"]["backward"]:
                raise RuntimeError("a timed round ran no device backward: the trunk is not the card's")

    real_sg = T.TTBioAlphaFoldDesignModel.sequence_gradients

    def sequence_gradients(self, *a, **kw):
        now = time.time()
        close(now)
        if len(rounds) >= args.rounds:
            raise _Enough()
        if args.interleave:
            # ABBA: the arm alternates and the pairs swap order, so a drift in host load
            # cannot land on one arm.
            evo.trace_on = [True, False, False, True][len(marks) % 4]
        marks.append((now, "trace" if evo.trace_on else "eager", round(os.getloadavg()[0], 1)))
        return real_sg(self, *a, **kw)

    T.TTBioAlphaFoldDesignModel.sequence_gradients = sequence_gradients
    import functools
    campaign.AlphaFoldDesignModel = functools.partial(T.TTBioAlphaFoldDesignModel,
                                                      trunk="device")
    mpnn = os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn", "weights_neutral")

    blob = {"stamp": A.stamp(args.card) | {"pci": S.sysfs_node()[1], "argv": sys.argv,
                                           "aiclk_node": clock.path,
                                           "region_mb": args.region_mb,
                                           "length_bucket_size": campaign_length_bucket(settings),
                                           "seed": args.seed},
            "mode": "interleave" if args.interleave else args.arm}
    t0 = time.time()
    try:
        # The same install run_arm.py does: without it BindCraft 2 runs its own JAX Evoformer
        # on the host and neither arm touches the card.
        with evoformer_on_device(evo):
            campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=mpnn, max_trajectories=1)
    except _Enough:
        print(f"collected {len(rounds)} rounds", flush=True)
    blob["wall_s"] = round(time.time() - t0, 1)
    blob["rounds"] = rounds
    blob["calls"] = dict(evo.calls)
    blob["wire"] = evo.wire.stats()
    if args.digest:
        blob["digests"] = digests
    # Round 1 carries the jit compile and round 0 is BindCraft 2 re-entering from its own
    # compile path (bcx-round), so the distribution is taken over what is left.
    # A round that captured carries a one-time cost the other 124 rounds of a trajectory do not.
    body = [r for r in rounds if r["i"] >= 2 and not captured(rounds, r["i"])]
    per = {}
    for arm in ("trace", "eager"):
        rs = [r for r in body if r["arm"] == arm]
        if rs:
            per[arm] = S.dist([r["s"] for r in rs]) | {
                "trunk_s": S.dist([trunk_s(rounds, r["i"]) for r in rs]),
                "trunk_cpu_s": S.dist([trunk_cpu(rounds, r["i"]) for r in rs]),
                "aiclk": clock.window([r["span"] for r in rs]),
                "load1": [r["load1"] for r in rs]}
    blob["per_arm"] = per
    if "trace" in per and "eager" in per:
        blob["round_x"] = per["eager"]["median"] / per["trace"]["median"]
        blob["round_removed_s"] = per["eager"]["median"] - per["trace"]["median"]
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (args.out or f"round_{blob['mode']}_seed{args.seed}.json")
    path.write_text(json.dumps(blob, indent=1, default=str))
    print(json.dumps({k: blob.get(k) for k in ("round_x", "round_removed_s", "wall_s",
                                               "calls")}, default=str), flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
