#!/usr/bin/env python3
"""How much of the Pairformer block's wall is host dispatch, not device work.

Ten passes of this program measured device compute. Nobody measured the other side. The trunk
runs a 48-block Pairformer stack ten times (protenix.py:1982 N_CYCLES=10, :2006 48 blocks) --
480 invocations of one shape with one set of weights and, at 512 aa, no host round trip inside a
layer (the three `host_acc` sites gate on a 1.5 GiB pair tensor and 512 aa is 134.2 MB). That is
the shape ttnn trace exists for, and its wall against eager wall is the dispatch-exposed
fraction directly, with no model and no profiler.

Both arms run the SAME op sequence on the SAME input buffers. The eager arm re-runs the layer;
the trace arm replays a capture of one layer. The difference is host dispatch and nothing else.

  eager - replay
  -------------- = the fraction of the block a trace would delete
      eager

This is a timing probe, not a landing. It asserts no numerics: a partial trace over a loop with
eager work between replays is a documented correctness trap (memory
ttnn-trace-interleaved-eager-corruption), and the trajectory-PCC gate that catches it is only
worth running if this number clears the bar first.

  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:pairformer-resident-chunking \
  ~/tt-bio/env/bin/python3 perf/bigswing/trunk_trace_probe.py --n 512 --fast \
      --out perf/bigswing/trace_probe_512_fast_qb2c0.json
"""
import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "ledger_298"))


def _ttnn_version():
    """The wheel version. ttnn exposes no __version__, and qb2 (0.68.0) against qb1 (0.67.4,
    the production pin) is the one thing that must never be mixed inside a comparison."""
    try:
        from importlib.metadata import version
        return version("ttnn")
    except Exception:                                            # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["protenix-v2", "opendde"], default="protenix-v2")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--fast", action="store_true",
                    help="probe the shipping configuration. --fast is banked at 1.2207x, so a "
                         "default-mode dispatch figure is drawn from an account already spent.")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10,
                    help="replays per arm. 10 = one trunk's worth of cycles for one block.")
    ap.add_argument("--region-gib", type=float, default=4.0,
                    help="trace region reserved at device open. The capture holds every "
                         "intermediate the block allocates, so this is the knob that decides "
                         "whether capture is possible at all -- raise it if capture throws.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Read by get_device() at open time (tenstorrent.py:1238), so it must be set before the
    # first get_device() call anywhere in the process -- i.e. before build().
    os.environ["TT_BIO_TRACE_REGION_SIZE"] = str(int(args.region_gib * 2 ** 30))

    # qb2's cards are P300 boards. With TT_VISIBLE_DEVICES pinning a single chip, ttnn 0.68.0
    # classifies the cluster as CUSTOM and open_device() is a TT_FATAL without a mesh graph
    # descriptor (tt_cluster.cpp:273). tt_bio's own entry points (main.py, full_parity_gate.py,
    # perf_regression.py) set this for you; a bare perf tool that calls get_device() directly
    # does not, which is why this probe could never have run as written.
    if not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
        if _detect_p300_devices():
            mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
            if mgd:
                os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    import torch
    import ttnn
    from tt_bio import tenstorrent as T
    from tt_bio.tenstorrent import get_device, set_fast_mode
    from pf_block_ops import build

    rec = {"host": platform.node(), "mgd": os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "model": args.model, "n": args.n, "fast": args.fast,
           "iters": args.iters, "region_bytes": int(args.region_gib * 2 ** 30),
           "ttnn": _ttnn_version(),
           "loadavg_start": os.getloadavg()}

    dev = get_device()
    assert T.trace_region_size() > 0, "no trace region reserved; capture cannot run"
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    if args.fast:
        set_fast_mode(True)
    assert T._FAST_MODE is args.fast, T._FAST_MODE
    layer, c_z = build(args.model, ckc)

    N = args.n
    torch.manual_seed(0)
    s0 = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    z0 = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    rec["c_z"] = c_z
    print(f"model={args.model} c_z={c_z} N={N} fast={args.fast} "
          f"region={args.region_gib} GiB ttnn={rec['ttnn']}", flush=True)

    # The layer returns its own inputs: every residual is `z = ttnn.add_(z, z_update)`
    # (tenstorrent.py:2683 and the four below it), which mutates in place and hands back the
    # same buffer. Deallocating the return value therefore frees s0/z0 and the next iteration
    # dies on "Buffer is not allocated". Drop the reference instead; nothing leaks, because the
    # only thing returned is what we allocated.
    for _ in range(args.warm):
        layer(s0, z0)
    ttnn.synchronize_device(dev)

    # --- eager arm. Same buffers every iteration, updated in place, so the two arms differ only
    # in who issues the commands. Chaining fresh s,z the way pf_block_ops does would churn the
    # allocator differently from a replay and make the ratio measure two things at once.
    t0 = time.perf_counter()
    for _ in range(args.iters):
        layer(s0, z0)
    ttnn.synchronize_device(dev)
    eager = (time.perf_counter() - t0) / args.iters
    rec["eager_ms"] = eager * 1e3
    print(f"eager  = {eager * 1e3:.3f} ms/block", flush=True)

    # --- trace arm.
    try:
        ttnn.synchronize_device(dev)
        tc = time.perf_counter()
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = layer(s0, z0)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        rec["capture_ms"] = (time.perf_counter() - tc) * 1e3
        print(f"capture = {rec['capture_ms']:.1f} ms", flush=True)

        t0 = time.perf_counter()
        for _ in range(args.iters):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        replay = (time.perf_counter() - t0) / args.iters
        rec["replay_ms"] = replay * 1e3
        rec["ratio"] = eager / replay
        rec["dispatch_exposed_pct"] = 100.0 * (eager - replay) / eager
        print(f"replay = {replay * 1e3:.3f} ms/block   eager/replay = {eager / replay:.4f}   "
              f"dispatch-exposed = {rec['dispatch_exposed_pct']:.2f} %", flush=True)
        # release_trace frees the capture's tensors itself, so `out` must not be deallocated
        # and must not lose its last Python reference first (esmc.py:471, protenix.py:2220).
        ttnn.release_trace(dev, tid)
        del out
    except Exception as e:                                       # noqa: BLE001
        # A capture that cannot be taken is a verdict, not a crash: it says the block's
        # allocation pattern is outside what ttnn trace records, and it says so for a
        # nameable reason. Record it and let the reason be read.
        rec["capture_error"] = f"{type(e).__name__}: {e}"[:600]
        traceback.print_exc()
        print("CAPTURE FAILED -- this is a result. See capture_error in the JSON.", flush=True)

    rec["loadavg_end"] = os.getloadavg()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(f"wrote {args.out}", flush=True)
    print(f"loadavg {rec['loadavg_start']} -> {rec['loadavg_end']}", flush=True)


if __name__ == "__main__":
    main()
