#!/usr/bin/env python3
"""Multicast the matmul operand instead of daisy-chaining it: parity, then a paired block A/B.

`b2z2-tile-arrival-latency` instrumented the matmul reader on Wormhole and split its wait:
dependency serialisation 67.6 %, the unicast forward 15.2 %, circular-buffer room 13.9 %, the
injector's DRAM read 3.2 %, bytes genuinely in flight 0.63 %. The first two terms are the daisy
chain -- one injector core per grid axis reads DRAM and every other core takes a semaphore-gated
hop from its predecessor -- and they are 82.8 % of the reader's own time.

`MM_MCAST_OPERAND` (`tt_bio/kernels/mm_mcast.py`) replaces the chain with one
`noc_async_write_multicast` per block. The prediction, its two pricings and its falsifiers are in
`perf/b2z2_mcast/PREDICTION.md`, pre-registered before this script ever ran.

What this does, in one process so no arm can be compared across a tree change:

1. **Parity first, and it is scored, not assumed.** One settled `PairformerLayer` call is replayed
   with the chain and with the multicast off the same cloned inputs, and both outputs must be
   `torch.equal`. A negative control perturbs one element of the input and must make that same
   comparison fail -- otherwise the check reads nothing.
2. **A paired, mirrored A/B.** Each arm is captured into its own trace and replayed back to back
   with no host work in between, in the order A B B A so a linear drift over the rep cancels. The
   A/A floor is the two A slots against each other, measured every rep, not asserted.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {"doc": __doc__}
OUT_PATH: Path | None = None

# whglx hangs forever at device open with a trace region of 1 GiB or more on a 32-chip mesh.
TRACE_REGION = 512 << 20


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def set_arm(on):
    os.environ["TT_BIO_MM_MCAST"] = "1" if on else "0"


def clone_args(ttnn, args, kwargs):
    def c(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x
    return tuple(c(x) for x in args), {k: c(v) for k, v in kwargs.items()}


def to_torch(ttnn, out):
    return [ttnn.to_torch(t) for t in (out if isinstance(out, (tuple, list)) else [out])
            if isinstance(t, ttnn.Tensor)]


def equal(a, b):
    import torch
    return len(a) == len(b) and all(x.shape == y.shape and torch.equal(x, y)
                                    for x, y in zip(a, b))


def capture(ttnn, dev, fn, args, kwargs):
    for _ in range(2):
        fn(*args, **kwargs)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    fn(*args, **kwargs)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    return tid


def replay_ms(ttnn, dev, tid, reps):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    return 1e3 * (time.perf_counter() - t0) / reps


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "block_ab.json")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=12, help="trace replays per timed slot")
    ap.add_argument("--n", type=int, default=7, help="mirrored A B B A reps")
    a = ap.parse_args()
    OUT_PATH = a.out

    set_arm(False)
    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps
    import tt_bio.mm_generic as MG

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=TRACE_REGION)
    fix = ROOT / "perf" / "size512" / "fixtures"
    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg": open("/proc/loadavg").read().split()[:3],
                  "grid": list(T.COMPUTE_GRID_MAIN), "size": a.size,
                  "reps": a.reps, "n": a.n}
    print(json.dumps(OUT["env"]), flush=True)
    dump()

    one_fold, meta, _state = B.build_fold(
        "boltz2", HERE / f".msa_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
        fix / f"cdk2x2_{a.size}.a3m")
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dev = T.get_device()

    # ---- grab one settled PairformerLayer call --------------------------------------------
    grab, counts = {}, [0]
    orig = T.PairformerLayer.__dict__["__call__"]

    def wrapped(self_obj, *args, **kw):
        counts[0] += 1
        if not grab and counts[0] >= 3 and getattr(self_obj, "transform_s", False):
            ca, ck = clone_args(ttnn, args, kw)
            grab.update(obj=self_obj, args=ca, kwargs=ck)
            print(f"  grabbed PairformerLayer on call {counts[0]}", flush=True)
        return orig(self_obj, *args, **kw)

    print("=== cold fold ===", flush=True)
    t0 = time.perf_counter()
    one_fold()
    OUT["cold_s"] = round(time.perf_counter() - t0, 3)
    dump()

    T.PairformerLayer.__call__ = wrapped
    print("=== fold 2 (grabbing) ===", flush=True)
    t0 = time.perf_counter()
    one_fold()
    T.PairformerLayer.__call__ = orig
    OUT["fold2_s"] = round(time.perf_counter() - t0, 3)
    OUT["pairformer_calls_per_fold"] = counts[0]
    if not grab:
        OUT["error"] = "no PairformerLayer grabbed"
        dump()
        return 1
    layer, gargs, gkwargs = grab["obj"], grab["args"], grab["kwargs"]
    dump()

    def run(on):
        set_arm(on)
        ca, ck = clone_args(ttnn, gargs, gkwargs)
        return to_torch(ttnn, layer(*ca, **ck))

    # ---- 1. parity, with a negative control that must break the same comparison ------------
    print("=== parity ===", flush=True)
    chain, mcast = run(False), run(True)
    base_equal = equal(chain, mcast)
    shapes = [list(t.shape) for t in chain]

    # The control perturbs the pair track by one element and must break the same comparison. It
    # goes through a torch round trip, so a second run with the round trip and NO perturbation
    # pins that the round trip is not what the control is detecting.
    def vol(x):
        v = 1
        for d in x.shape:
            v *= int(d)
        return v

    def rebuilt(bump):
        set_arm(False)
        ca, ck = clone_args(ttnn, gargs, gkwargs)
        ca = list(ca)
        zi = max((i for i, x in enumerate(ca) if isinstance(x, ttnn.Tensor)),
                 key=lambda i: vol(ca[i]))
        zt = ttnn.to_torch(ca[zi])
        if bump:
            flat = zt.reshape(-1)
            eps = torch.finfo(torch.bfloat16).eps
            flat[0] = flat[0] + (eps * 8 if flat[0] == 0 else flat[0].abs() * eps * 8)
        ca[zi] = ttnn.from_torch(zt, layout=ca[zi].layout, dtype=ca[zi].dtype, device=dev,
                                 memory_config=ca[zi].memory_config())
        return to_torch(ttnn, layer(*ca, **ck)), list(ca[zi].shape)

    roundtrip, zshape = rebuilt(False)
    perturbed, _ = rebuilt(True)
    roundtrip_equal = equal(chain, roundtrip)
    control_equal = equal(chain, perturbed)

    OUT["parity"] = {"bit_exact": base_equal, "negative_control_equal": control_equal,
                     "control_is_valid": roundtrip_equal and not control_equal,
                     "roundtrip_only_equal": roundtrip_equal,
                     "perturbed_tensor_shape": zshape, "out_shapes": shapes,
                     "max_abs_diff": [float((x - y).abs().max()) for x, y in zip(chain, mcast)]}
    print("  " + json.dumps(OUT["parity"]), flush=True)
    dump()
    if not base_equal or not OUT["parity"]["control_is_valid"]:
        OUT["parity"]["verdict"] = "STOP: bit-exactness failed or the control reads nothing"
        dump()
        return 2

    # ---- 2. the paired, mirrored A/B -------------------------------------------------------
    print("=== capture ===", flush=True)
    set_arm(False)
    tid_a = capture(ttnn, dev, layer, gargs, gkwargs)
    set_arm(True)
    tid_b = capture(ttnn, dev, layer, gargs, gkwargs)
    OUT["mcast_programs_built"] = sum(
        1 for e in MG._CACHE.values() if e["dims"].get("mcast"))
    OUT["total_programs_built"] = len(MG._CACHE)
    print(f"  generic matmul programs: {OUT['total_programs_built']} total, "
          f"{OUT['mcast_programs_built']} multicast", flush=True)
    dump()

    for tid in (tid_a, tid_b):
        for _ in range(3):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)

    rows = []
    for i in range(a.n):
        a1 = replay_ms(ttnn, dev, tid_a, a.reps)
        b1 = replay_ms(ttnn, dev, tid_b, a.reps)
        b2 = replay_ms(ttnn, dev, tid_b, a.reps)
        a2 = replay_ms(ttnn, dev, tid_a, a.reps)
        rows.append({"chain": [round(a1, 4), round(a2, 4)],
                     "mcast": [round(b1, 4), round(b2, 4)]})
        print(f"  rep {i}: chain {a1:.4f} {a2:.4f}  mcast {b1:.4f} {b2:.4f}  "
              f"ratio {(a1 + a2) / (b1 + b2):.5f}", flush=True)
        OUT["rows"] = rows
        dump()

    chain_ms = st.median([x for r in rows for x in r["chain"]])
    mcast_ms = st.median([x for r in rows for x in r["mcast"]])
    OUT["result"] = {
        "chain_block_ms": round(chain_ms, 4),
        "mcast_block_ms": round(mcast_ms, 4),
        "ratio_chain_over_mcast": round(chain_ms / mcast_ms, 5),
        "aa_floor_chain": round(st.median([r["chain"][0] for r in rows])
                                / st.median([r["chain"][1] for r in rows]), 5),
        "aa_floor_mcast": round(st.median([r["mcast"][0] for r in rows])
                                / st.median([r["mcast"][1] for r in rows]), 5),
    }
    print(json.dumps(OUT["result"], indent=1), flush=True)
    ttnn.release_trace(dev, tid_a)
    ttnn.release_trace(dev, tid_b)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
