"""Device seconds for the structure module, forward and backward, with the clock it ran at.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-structmod \
      python3 scripts/bcx_structmod/device_time.py --ref .../ref_real_n288_fold.npz --reps 10

Three numbers, not one: the untaped forward (what an inference path would pay), the taped
forward (what a design step pays, because the tape has to keep what the backward reads) and the
backward. Every repeat is synchronised, and the AICLK is sampled DURING the timed window off
the card's own class node -- `tt_aiclk` lives on `/sys/class/tenstorrent/tenstorrent!N`, not
under `device/` -- because on this hardware the clock sets the time and a number without one is
not a measurement. The node is found from the process's own open file descriptor rather than
from TT_VISIBLE_DEVICES, which is not the /dev node number.

loadavg1 is recorded per repeat. This module is dispatch-bound, so a loaded host inflates it,
and a run taken at loadavg 20 is an upper bound on the device cost rather than the cost.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import ttnn

from tt_bio import af2_structure as sm
from tt_bio import autograd as ag
from tt_bio import taped_ttnn


def clock_node() -> str | None:
    """Which class node this process's open /dev/tenstorrent fd belongs to."""
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("/dev/tenstorrent/"):
            n = target.rsplit("/", 1)[1]
            path = f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk"
            if os.path.exists(path):
                return path
    return None


class Sampler(threading.Thread):
    def __init__(self, path, period=0.25):
        super().__init__(daemon=True)
        self.path, self.period, self.samples, self.stop = path, period, [], False

    def run(self):
        while not self.stop:
            try:
                self.samples.append(int(open(self.path).read().split()[0]))
            except Exception:
                pass
            time.sleep(self.period)


def summarize(name, times, samples, loads):
    out = {"phase": name, "reps": len(times),
           "median_s": statistics.median(times), "min_s": min(times), "max_s": max(times),
           "times_s": [round(t, 4) for t in times],
           "load1": [round(x, 2) for x in loads]}
    if samples:
        out.update({"aiclk_mhz_median": statistics.median(samples),
                    "aiclk_mhz_min": min(samples), "aiclk_mhz_max": max(samples),
                    "aiclk_samples": len(samples)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz")
    ap.add_argument("--dtype", default="float32", choices=("bfloat16", "float32"))
    ap.add_argument("--point-dtype", default="float32", choices=("bfloat16", "float32"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = np.load(args.ref)
    n = int(ref["single"].shape[0])
    raw = np.load(args.params)
    params = {k: raw[k] for k in raw.files if "structure_module" in k}
    dtypes = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}

    device = ttnn.open_device(device_id=args.device_id)
    node = clock_node()
    if node is None:
        raise SystemExit("no tt_aiclk class node for this process's device; refusing to "
                         "publish a time without the clock it ran at")
    try:
        weights = sm.StructureWeights(params, device, dtype=dtypes[args.dtype],
                                      point_dtype=dtypes[args.point_dtype])
        module = sm.AF2MultimerStructureModule(weights)

        def up(array, dtype):
            return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(array, np.float32)),
                                   layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)

        single_v = up(np.asarray(ref["single"])[None, None], dtypes[args.dtype])
        pair_v = up(np.asarray(ref["pair"])[None], dtypes[args.dtype])
        seq_mask = up(np.asarray(ref["seq_mask"]).reshape(1, 1, n, 1),
                      dtypes[args.point_dtype])
        ct_act = np.asarray(ref["ct_act"])
        ct_traj = np.asarray(ref["ct_traj"])
        ct_unnorm = np.asarray(ref["ct_unnormalized"])

        def run_untaped():
            module(single_v, pair_v, seq_mask)

        def run_taped():
            single = ag.Tensor(single_v, requires_grad=True)
            pair = ag.Tensor(pair_v, requires_grad=True)
            with taped_ttnn.tape():
                act, traj, angles = module(single, pair, seq_mask)
            return single, pair, act, traj, angles

        def seed_and_back(act, traj, angles):
            roots, seeds = [act], [up(ct_act[None, None], act.value.dtype)]
            for layer in range(sm.NUM_LAYER):
                flat = ct_traj[layer].reshape(n, 12)
                for slot, comp in enumerate(traj[layer]):
                    roots.append(comp)
                    seeds.append(up(flat[:, slot].reshape(1, 1, n, 1), comp.value.dtype))
                roots.append(angles[layer])
                seeds.append(up(ct_unnorm[layer].reshape(1, 1, n, 14),
                                angles[layer].value.dtype))
            ag.backward(roots, seeds)

        for _ in range(args.warm):
            run_untaped()
            s, p, a, t, g = run_taped()
            seed_and_back(a, t, g)
        ttnn.synchronize_device(device)

        report = {"n": n, "dtype": args.dtype, "point_dtype": args.point_dtype,
                  "reps": args.reps, "warm": args.warm, "clock_node": node,
                  "card_type": open(os.path.dirname(node) + "/tt_card_type").read().strip(),
                  "host": os.uname().nodename, "phases": {}}

        for name, body in (("forward_untaped", None), ("forward_taped", None),
                           ("backward", None)):
            sampler = Sampler(node)
            sampler.start()
            times, loads = [], []
            for _ in range(args.reps):
                if name == "forward_untaped":
                    t0 = time.time()
                    run_untaped()
                    ttnn.synchronize_device(device)
                    times.append(time.time() - t0)
                elif name == "forward_taped":
                    t0 = time.time()
                    s, p, a, t, g = run_taped()
                    ttnn.synchronize_device(device)
                    times.append(time.time() - t0)
                    seed_and_back(a, t, g)          # keep memory from piling up
                    ttnn.synchronize_device(device)
                else:
                    s, p, a, t, g = run_taped()
                    ttnn.synchronize_device(device)
                    t0 = time.time()
                    seed_and_back(a, t, g)
                    ttnn.synchronize_device(device)
                    times.append(time.time() - t0)
                loads.append(os.getloadavg()[0])
            sampler.stop = True
            sampler.join(timeout=2)
            report["phases"][name] = summarize(name, times, sampler.samples, loads)
    finally:
        ttnn.close_device(device)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
