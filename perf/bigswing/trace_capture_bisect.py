#!/usr/bin/env python3
"""Which op stops `end_trace_capture` from returning -- and whether it is an op at all.

Pass 14 (state doc §99) found that `ttnn.end_trace_capture` never returns for a Pairformer block
on qb2 / ttnn 0.68.0, at N=128 and N=512 alike, spinning at 100 % CPU. It read that as a defect in
the block's op set, on the grounds that "production traces the diffusion denoiser on the same
device, same wheel, same host".

That premise is false. `Protenix.fold(trace=False)` is the default (`protenix.py:1893`) and
`scripts/gpu_vs_tt/tt_baseline.py` passes `trace=False` at both call sites (:163, :248), so no fold
this program has ever quoted exercised trace capture on qb2. The control was never run.

So this tool runs the control first: capture a bare `ttnn.add`, then a bare `ttnn.matmul`, with no
model at all. If those do not close either, the defect is the host/wheel/cluster path and not the
Pairformer, and the whole trace lever has to be measured on qb1 at the 0.67.4 production pin.

Each part runs in a CHILD PROCESS, because the hang is inside C++ and a Python SIGALRM handler
cannot interrupt it -- the interpreter never regains control. A hang is therefore a data point
instead of a turn-ender. On a hang the parent takes TWO native backtraces ~12 s apart: one stack
names the frame, two stacks distinguish a spin in one place from slow forward progress. The child
sets PR_SET_PTRACER_ANY first because qb2 has yama ptrace_scope=1, under which a sibling gdb cannot
attach.

  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:pairformer-resident-chunking \
  ~/tt-bio/env/bin/python3 perf/bigswing/trace_capture_bisect.py --n 128 \
      --parts control_add,control_matmul --timeout 90 \
      --out perf/bigswing/capture_control_128_qb2c0.json
"""
import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "ledger_298"))

PARTS = ("control_add", "control_matmul", "trimul_start", "trimul_end", "triatt_start",
         "triatt_end", "transition_z", "pwa", "transition_s", "full_block")
NEEDS_LAYER = {p for p in PARTS if not p.startswith("control_")}
TT_SMI = os.path.expanduser("~/tt-bio/env/bin/tt-smi")


def _ttnn_version():
    try:
        from importlib.metadata import version
        return version("ttnn")
    except Exception:                                            # noqa: BLE001
        return "unknown"


def _p300_preamble():
    """qb2's cards are P300 boards. With TT_VISIBLE_DEVICES pinning one chip, ttnn 0.68.0 calls
    the cluster CUSTOM and open_device() is a TT_FATAL without a mesh graph descriptor
    (tt_cluster.cpp:273). tt_bio's own entry points set this; a bare perf tool does not. This is
    the gap §96 found in 9 of 9 tools under perf/."""
    if os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        return
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices():
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd


# --------------------------------------------------------------------------------------- child

def run_part(args):
    """Capture exactly one part and exit. Prints a marker line, and writes the marker FILE, right
    before end_trace_capture so the parent knows a hang is that call and not device open."""
    # qb2 has yama ptrace_scope=1, so a sibling gdb cannot attach unless the target opts in.
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl(0x59616d61, ctypes.c_ulong(-1), 0, 0, 0)          # PR_SET_PTRACER, ..._ANY

    os.environ["TT_BIO_TRACE_REGION_SIZE"] = str(int(args.region_gib * 2 ** 30))
    _p300_preamble()

    import torch
    import ttnn
    from tt_bio import tenstorrent as T
    from tt_bio.tenstorrent import get_device, set_fast_mode

    part, N = args.run_part, args.n
    dev = get_device()
    assert T.trace_region_size() > 0, "no trace region reserved; capture cannot run"
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    if args.fast:
        set_fast_mode(True)
    torch.manual_seed(0)

    layer = None
    c_z = 128
    if part in NEEDS_LAYER:
        from pf_block_ops import build
        layer, c_z = build(args.model, ckc)

    if part.startswith("control_"):
        # No model, no weights, no program config. The point is to find out whether this stack
        # closes ANY capture, so keep the op set as small as it can be.
        a = ttnn.from_torch(torch.randn(1, 1, 512, 512), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.randn(1, 1, 512, 512), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        if part == "control_add":
            def call():
                return ttnn.add(a, b)
        else:
            def call():
                return ttnn.matmul(a, b, compute_kernel_config=ckc)
    else:
        s0 = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)
        z0 = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)
        fns = {
            "trimul_start":  lambda: layer.triangle_multiplication_start(z0, None),
            "trimul_end":    lambda: layer.triangle_multiplication_end(z0, None),
            "triatt_start":  lambda: layer.triangle_attention_start(z0, None),
            "triatt_end":    lambda: layer.triangle_attention_end(z0, None),
            "transition_z":  lambda: layer.transition_z(z0),
            "pwa":           lambda: layer.attention_pair_bias(s0, z0, seq_mask=None),
            "transition_s":  lambda: layer.transition_s(s0),
            "full_block":    lambda: layer(s0, z0),
        }
        call = fns[part]

    print(f"PART {part} N={N} c_z={c_z} fast={args.fast} ttnn={_ttnn_version()}", flush=True)

    # Warm OUTSIDE capture, so JIT compilation and any lazily-created constant is already done and
    # a hang cannot be blamed on either. --warm 0 tests the opposite case on purpose.
    for _ in range(args.warm):
        call()
    ttnn.synchronize_device(dev)
    print("WARM_OK", flush=True)

    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    print("CAPTURE_BEGUN", flush=True)
    t0 = time.perf_counter()
    call()
    rec_ms = (time.perf_counter() - t0) * 1e3
    print(f"RECORDED {rec_ms:.1f} ms", flush=True)
    Path(args.marker).write_text(f"{os.getpid()} end_trace_capture {part}\n")
    print("ENTERING_END_TRACE_CAPTURE", flush=True)
    t0 = time.perf_counter()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    end_ms = (time.perf_counter() - t0) * 1e3
    print(f"END_TRACE_CAPTURE_OK {end_ms:.1f} ms", flush=True)

    t0 = time.perf_counter()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    print(f"REPLAY_OK {(time.perf_counter() - t0) * 1e3:.3f} ms", flush=True)
    ttnn.release_trace(dev, tid)
    print(f"RESULT {json.dumps({'part': part, 'record_ms': rec_ms, 'end_ms': end_ms})}", flush=True)


# -------------------------------------------------------------------------------------- parent

def _gdb(pid, depth=60):
    try:
        p = subprocess.run(["gdb", "-p", str(pid), "-batch", "-nx",
                            "-ex", "set pagination off", "-ex", "set confirm off",
                            "-ex", f"thread apply all bt {depth}"],
                           capture_output=True, text=True, timeout=180)
        return (p.stdout + p.stderr)[-24000:]
    except Exception as e:                                       # noqa: BLE001
        return f"gdb failed: {type(e).__name__}: {e}"


def _reset_card():
    """SIGKILL to a process wedged inside end_trace_capture leaves the card dirty and the next
    get_device() hangs in _assert_local_dispatch (§99). tt-smi -r clears it."""
    try:
        p = subprocess.run([TT_SMI, "-r", "0"], capture_output=True, text=True, timeout=300)
        return p.returncode
    except Exception as e:                                       # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["protenix-v2", "opendde"], default="protenix-v2")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--parts", default="control_add,control_matmul")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--region-gib", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="seconds allowed AFTER the child reports ENTERING_END_TRACE_CAPTURE")
    ap.add_argument("--open-timeout", type=float, default=600.0,
                    help="seconds allowed to reach that marker (device open + weight load)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-part", help="internal: child mode")
    ap.add_argument("--marker", default="")
    args = ap.parse_args()

    if args.run_part:
        run_part(args)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"host": platform.node(), "ttnn": _ttnn_version(), "model": args.model, "n": args.n,
           "fast": args.fast, "warm": args.warm, "region_gib": args.region_gib,
           "loadavg_start": os.getloadavg(), "parts": {}}

    for part in args.parts.split(","):
        part = part.strip()
        if part not in PARTS:
            raise SystemExit(f"unknown part {part!r}; known: {PARTS}")
        marker = out.parent / f".marker_{part}"
        marker.unlink(missing_ok=True)
        log = out.parent / f"{out.stem}_{part}.log"
        cmd = [sys.executable, __file__, "--run-part", part, "--model", args.model,
               "--n", str(args.n), "--warm", str(args.warm),
               "--region-gib", str(args.region_gib), "--out", str(out), "--marker", str(marker)]
        if args.fast:
            cmd.append("--fast")
        print(f"\n=== {part} ===", flush=True)
        t0 = time.perf_counter()
        with open(log, "w") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
            entry = {"log": str(log)}
            deadline_open = t0 + args.open_timeout
            marker_at = None
            while True:
                if proc.poll() is not None:
                    break
                now = time.perf_counter()
                if marker_at is None and marker.exists():
                    marker_at = now
                    print(f"  entered end_trace_capture at {now - t0:.1f} s", flush=True)
                if marker_at is None and now > deadline_open:
                    entry["verdict"] = "TIMEOUT_BEFORE_CAPTURE"
                    break
                if marker_at is not None and now - marker_at > args.timeout:
                    entry["verdict"] = "HANG_IN_END_TRACE_CAPTURE"
                    entry["hang_s"] = round(now - marker_at, 1)
                    break
                time.sleep(0.5)

        if "verdict" not in entry:
            entry["verdict"] = "PASS" if proc.returncode == 0 else f"EXIT_{proc.returncode}"
            entry["wall_s"] = round(time.perf_counter() - t0, 1)
            body = log.read_text()
            for line in body.splitlines():
                if line.startswith("RESULT "):
                    entry.update(json.loads(line[len("RESULT "):]))
                if line.startswith("REPLAY_OK"):
                    entry["replay_ms"] = float(line.split()[1])
            if entry["verdict"] != "PASS":
                entry["tail"] = body[-3000:]
        else:
            # Two samples, ~12 s apart: one names the frame, two say spin vs progress.
            entry["stack_1"] = _gdb(proc.pid)
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < 12.0:
                time.sleep(0.5)
            entry["stack_2"] = _gdb(proc.pid)
            entry["cpu"] = subprocess.run(["ps", "-o", "pcpu=,stat=,wchan=", "-p", str(proc.pid)],
                                          capture_output=True, text=True).stdout.strip()
            proc.kill()
            proc.wait(timeout=60)
            entry["tt_smi_reset"] = _reset_card()
            entry["tail"] = log.read_text()[-3000:]

        marker.unlink(missing_ok=True)
        print(f"  {part}: {entry['verdict']}", flush=True)
        rec["parts"][part] = entry
        out.write_text(json.dumps(rec, indent=1))

    rec["loadavg_end"] = os.getloadavg()
    out.write_text(json.dumps(rec, indent=1))
    print("\n=== SUMMARY ===", flush=True)
    for k, v in rec["parts"].items():
        print(f"  {k:16s} {v['verdict']}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
