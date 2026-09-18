#!/usr/bin/env python3
"""Price the 52 refused launch keys standalone, on one chip, in one session, at a pinned and
during-sampled 1350 MHz, with both roofs measured in the same session.

WHY A STANDALONE ARM WHEN AN IN-SITU NUMBER EXISTS. It exists for the block as a whole:
`c12-profiled-fold` measured 13.2090 s of in-fold device time and its `unsplit_s` field holds
1.7508 s of device ops it could not attribute to a ttnn class -- which is this block. So the
block's total is already measured. What is NOT measured is the block PER KEY, and a second
instrument built on different data is the only way to test the total. The test is directional
and pre-registered: `c12-genericop-rate` measured in situ 1.16-1.18x FASTER than every
standalone arm on the same kernels, so a standalone sum must come out AT OR ABOVE the in-situ
2.0277 s. A standalone sum below it would mean one of the two instruments is wrong.

CONTROLS, in this order, before any row is written:
  * the 8192^3 known-answer case -- exactly 1,099,511,627,776 FLOPs and exactly 402,653,184
    bytes -- checked in `arms_ctl.cube_control()`. 95.9 % of one recent 0.4544 s claim in this
    campaign was counter artifacts, so this is not a formality and the run aborts on it.
  * the dense cube measured at BOTH ends of the session, so session drift is a number.
  * the DRAM roof measured in-session, because a rate is only a roof on the chip it was taken on.
  * the clock forced through ARC 0x33 and sampled at 1 kHz from a separate process THROUGH every
    timed interval. An interval that is not min == max == 1350 with no gap over 10 ms is an
    ARTIFACT and is dropped, not reported.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from math import prod
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "perf/c10_bare_baseline"), str(ROOT)]

import arms as A                                                              # noqa: E402
import control                                                                # noqa: E402
from force_aiclk import FORCE_AICLK, smc                                       # noqa: E402

TARGET_MHZ = 1350
CUBE_N = 8192
CUBE_FLOPS = 1_099_511_627_776
CUBE_BYTES = 402_653_184
GROUP_TARGET_S = 0.020
BRACKET_TARGET_S = 0.030
LIVE_BYTES = 2_000_000_000
MAX_REPS = 2048
BF16 = 2


def cube_control():
    """The known-answer case both counters must return before a table row is written."""
    f = 2 * CUBE_N * CUBE_N * CUBE_N
    b = CUBE_N * CUBE_N * BF16 * 3
    return {"n": CUBE_N, "flops": f, "bytes": b,
            "flops_expected": CUBE_FLOPS, "bytes_expected": CUBE_BYTES,
            "flops_pass": f == CUBE_FLOPS, "bytes_pass": b == CUBE_BYTES}


def snapshot():
    """`control.snapshot` without its containment assertion.

    The shared helper runs `systemctl is-active qb2-endpoint-containment.service` under
    `check_output` and dies on a non-zero exit, and this brief says containment stays OFF, so
    the state is RECORDED here rather than required. Everything else -- boot id, driver
    srcversion, device holders, own fds -- is taken verbatim from the same helper.
    """
    try:
        cont = subprocess.run(["systemctl", "is-active",
                               "qb2-endpoint-containment.service"],
                              capture_output=True, text=True).stdout.strip()
    except OSError as e:                                                      # noqa: BLE001
        cont = repr(e)
    return {"monotonic_ns": time.monotonic_ns(), "utc_ns": time.time_ns(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "module_srcversion":
                Path("/sys/module/tenstorrent/srcversion").read_text().strip(),
            "containment": cont,
            "holders": control.holders(), "own_nodes": control.own_nodes()}


def board_power_W(node):
    root = Path("/sys/class/tenstorrent/tenstorrent!%d" % node)
    try:
        hw = next((root / "device").glob("hwmon/hwmon*"))
        return int((hw / "power1_input").read_text()) / 1e6
    except (OSError, StopIteration, ValueError) as e:                         # noqa: BLE001
        return repr(e)


def build_make(ttnn, torch, dev, spec):
    """(keep, fn, mode) for one arm, or a string naming why it could not be constructed.

    `mode` is "out" when the call allocates a result that must be freed inside the bracket.
    """
    DRAMC = ttnn.DRAM_MEMORY_CONFIG
    p, arm = spec["params"], spec["arm"]

    def t(shape, layout="TILE", mc=None):
        lay = ttnn.TILE_LAYOUT if layout == "TILE" else ttnn.ROW_MAJOR_LAYOUT
        x = torch.randn(*shape, dtype=torch.bfloat16)
        return ttnn.from_torch(x, layout=lay, device=dev, memory_config=mc or DRAMC)

    try:
        if arm in ("permute",):
            a = t(p["src"])
            return [a], (lambda: ttnn.permute(a, tuple(p["dims"]))), "out"
        if arm == "transpose":
            d = [i for i, (x, y) in enumerate(zip(p["src"], spec["out"])) if x != y]
            a = t(p["src"])
            d0, d1 = (d + [0, 1])[:2]
            return [a], (lambda: ttnn.transpose(a, d0, d1)), "out"
        if arm == "reshape":
            a = t(p["src"])
            return [a], (lambda: ttnn.reshape(a, tuple(p["dst"]))), "out"
        if arm == "slice":
            a = t(p["src"])
            st, e = list(p["starts"]), list(p["ends"])
            degenerate = st == [0] * len(st) and e == list(p["src"])
            return [a], (lambda: ttnn.slice(a, st, e)), ("view" if degenerate else "out")
        if arm == "pad":
            a = t(p["src"])
            pads = [(int(lo), int(hi)) for lo, hi in p["padding"]]
            return [a], (lambda: ttnn.pad(a, padding=pads, value=0.0)), "out"
        if arm == "concat":
            parts = [t(s) for s in p["parts"]]
            d = int(p["dim"])
            return parts, (lambda: ttnn.concat(parts, dim=d)), "out"
        if arm == "chunk":
            a = t(p["src"])
            n, d = int(p["chunks"]), int(p["dim"])
            return [a], (lambda: ttnn.chunk(a, n, dim=d)), "list"
        if arm == "to_layout":
            a = t(p["src"], p["src_layout"])
            tgt = (ttnn.ROW_MAJOR_LAYOUT if p["target_layout"] == "ROW_MAJOR"
                   else ttnn.TILE_LAYOUT)
            return [a], (lambda: ttnn.to_layout(a, tgt)), "out"
        if arm == "to_memory_config_l1":
            a = t(p["src"])
            return [a], (lambda: ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG)), "out"
        if arm == "softmax":
            a = t(p["src"])
            return [a], (lambda: ttnn.softmax(a, -1)), "out"
        if arm == "cos":
            a = t(p["src"])
            return [a], (lambda: ttnn.cos(a)), "out"
        if arm == "matmul":
            a, b = t(p["a"]), t(p["b"])
            return [a, b], (lambda: ttnn.matmul(a, b, memory_config=DRAMC)), "out"
        if arm == "sdpa":
            q = t(p["q"])
            k = t(p["kv"])
            v = t(p["kv"])
            keep = [q, k, v]
            if p.get("mask"):
                m = t(p["mask"])
                keep.append(m)
                return keep, (lambda: ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, attn_mask=m, is_causal=False)), "out"
            return keep, (lambda: ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False)), "out"
        if arm == "nlp_create_qkv_heads":
            a = t(p["src"])
            nh, hd = int(p["num_heads"]), int(p["head_dim"])
            # the kernel requires (n_q + 2*n_kv) to divide the fused width. The capture records
            # the fused width and the q head count, so n_kv follows: total heads = width/head_dim.
            total = p["src"][-1] // hd if hd else 0
            nkv = max(1, (total - nh) // 2) if total > nh else nh
            return [a], (lambda: ttnn.experimental.nlp_create_qkv_heads(
                a, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False,
                memory_config=DRAMC)), "list"
        if arm == "nlp_concat_heads":
            a = t(p["src"])
            return [a], (lambda: ttnn.experimental.nlp_concat_heads(
                a, memory_config=DRAMC)), "out"
    except Exception as e:                                                     # noqa: BLE001
        return "operand construction failed: %r" % (e,)
    return "no runner for arm %r" % arm


def time_arm(ttnn, dev, keep, fn, mode, out_bytes, blocks, live_budget=LIVE_BYTES):
    """Allocate, warm, time `blocks` brackets. Per-call cost is the best bracket.

    A bracket accumulates BRACKET_TARGET_S of enqueue-plus-synchronise time so the 1 kHz clock
    sampler has enough during-interval samples to qualify the interval at all; a 1 ms interval
    cannot be shown to have run at 1350 MHz. Output frees happen inside the bracket but outside
    the accumulated time, so no row is inflated by a host-side free the op did not have to do.
    """
    def free(r):
        if mode == "view":
            return          # the call returned a view over its own input; freeing it frees that
        if mode == "out":
            ttnn.deallocate(r)
        elif mode == "list":
            for x in (r if isinstance(r, (list, tuple)) else [r]):
                ttnn.deallocate(x)

    best, marks, reps = None, [], None
    t0 = time.perf_counter()
    for _ in range(2):
        free(fn())
    ttnn.synchronize_device(dev)
    warm = (time.perf_counter() - t0) / 2
    cal = max(1, min(8, int(GROUP_TARGET_S / warm) if warm > 0 else 8))
    t0 = time.perf_counter()
    outs = [fn() for _ in range(cal)]
    ttnn.synchronize_device(dev)
    est = (time.perf_counter() - t0) / cal
    for o in outs:
        free(o)
    live_cap = max(1, int(live_budget / out_bytes)) if out_bytes else MAX_REPS
    reps = max(1, min(MAX_REPS, live_cap, int(GROUP_TARGET_S / est) if est > 0 else 1))
    for _ in range(blocks):
        s_ns = time.monotonic_ns()
        acc, n = 0.0, 0
        while acc < BRACKET_TARGET_S:
            outs = []
            tA = time.perf_counter()
            for _ in range(reps):
                outs.append(fn())
            ttnn.synchronize_device(dev)
            acc += time.perf_counter() - tA
            n += reps
            for o in outs:
                free(o)
        e_ns = time.monotonic_ns()
        marks.append({"start_monotonic_ns": s_ns, "end_monotonic_ns": e_ns,
                      "reps": reps, "calls": n, "accumulated_s": acc, "s_per_call": acc / n})
        best = acc / n if best is None else min(best, acc / n)
    return {"s_per_call": best, "reps": reps, "warm_s": warm, "marks": marks}


def roof_arms(ttnn, torch, dev):
    """The two roofs, measured here. A borrowed rate makes a prediction a band, not a number."""
    kc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=False,
                                          packer_l1_acc=False) \
        if hasattr(ttnn, "WormholeComputeKernelConfig") else None

    def cube():
        n = CUBE_N
        a = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kw = {"compute_kernel_config": kc} if kc is not None else {}
        return [a, b], (lambda: ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)), \
            "out", n * n * BF16

    def dram():
        n = CUBE_N
        a = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return [a, b], (lambda: ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)), \
            "out", n * n * BF16
    return {"cube": cube, "dram": dram}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=3)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=False)

    pred = HERE / "prediction.json"
    if not pred.is_file():
        raise RuntimeError("prediction.json absent: register predictions before measuring")

    ctl = cube_control()
    if not (ctl["flops_pass"] and ctl["bytes_pass"]):
        raise RuntimeError("known-answer counter control FAILED: %s" % ctl)

    keys = json.loads((HERE / "keys_512.json").read_text())["keys"]
    built, unbuildable = A.build_all(keys)

    R = {"pid": os.getpid(), "host": socket.gethostname(), "node": a.node,
         "started_utc_ns": time.time_ns(),
         "prediction": json.loads(pred.read_text()),
         "cube_control": ctl,
         "git_rev": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            text=True).strip(),
         "dirty": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
         "n_arms": len(built), "unbuildable": [
             {"key": s["key"], "calls": s["calls"], "B": s["B"], "why": s["refused"]}
             for s in unbuildable],
         "rows": [], "roofs": {}, "errors": [], "completed": False}
    save = (lambda: control.write_json(out / "replay.json", R))
    save()

    if R["host"] != "tt-quietbox2":
        raise RuntimeError("wrong host: %s" % R["host"])
    for k, v in [("TT_VISIBLE_DEVICES", str(a.node)), ("TT_BIO_LEASE_CARDS", str(a.node))]:
        if os.environ.get(k) != v:
            raise RuntimeError("expected %s=%s, got %r" % (k, v, os.environ.get(k)))

    root = Path("/sys/class/tenstorrent/tenstorrent!%d" % a.node)
    R["node_identity"] = {n: (root / n).read_text().strip()
                          for n in ("tt_card_type", "tt_asic_id", "tt_serial", "tt_aiclk")
                          if (root / n).exists()}
    R["power_before_W"] = board_power_W(a.node)
    R["before"] = snapshot()
    save()

    import torch                                                              # noqa: PLC0415
    import ttnn                                                               # noqa: PLC0415
    R["ttnn_path"] = ttnn.__file__
    # NOT ttnn.open_device. A single-card TT_VISIBLE_DEVICES makes the cluster type CUSTOM on
    # this tt-metal and hard-fatals in tt_cluster.cpp:273 ("Custom fabric mesh graph descriptor
    # path must be specified"). tt-bio's own opener is what the fold uses, so it is both the
    # working route and the faithful one.
    import tt_bio.tenstorrent as T                                            # noqa: PLC0415
    R["tt_bio_path"] = T.__file__
    dev = T.get_device()
    fd = os.open("/dev/tenstorrent/%d" % a.node, os.O_RDWR | os.O_APPEND)
    sampler = None
    try:
        R["force_response"] = list(smc(fd, FORCE_AICLK, TARGET_MHZ))
        if R["force_response"][0] != 0:
            raise RuntimeError("FORCE_AICLK failed: %s" % R["force_response"])
        time.sleep(0.15)
        sampler = subprocess.Popen(
            [sys.executable, str(ROOT / "perf/c10_bare_baseline/control.py"),
             str(out / "clock.jsonl"), str(os.getpid())],
            stdin=subprocess.PIPE, stdout=(out / "sampler.log").open("w"),
            stderr=subprocess.STDOUT)
        time.sleep(0.4)
        save()

        roofs = roof_arms(ttnn, torch, dev)

        def run_roof(label, factory):
            keep, fn, mode, ob = factory()
            try:
                r = time_arm(ttnn, dev, keep, fn, mode, ob, a.blocks)
            finally:
                for x in keep:
                    try:
                        ttnn.deallocate(x)
                    except Exception:                                          # noqa: BLE001
                        pass
            n = CUBE_N
            r["label"] = label
            r["TFLOPs"] = (2 * n * n * n / r["s_per_call"] / 1e12) if label.startswith("cube") \
                else None
            r["GBps"] = (3 * n * n * BF16 / r["s_per_call"] / 1e9) if label.startswith("dram") \
                else None
            R["roofs"][label] = r
            save()
            return r

        run_roof("cube_open", roofs["cube"])
        run_roof("dram", roofs["dram"])

        for spec in built:
            row = {k: spec[k] for k in ("key", "op", "arm", "calls", "B", "pinned",
                                        "derivation", "params", "out", "in_shapes")}
            made = build_make(ttnn, torch, dev, spec)
            if isinstance(made, str):
                row["error"] = made
                R["rows"].append(row)
                save()
                continue
            keep, fn, mode = made
            ob = prod(spec["out"]) * BF16 if spec["out"] else 0
            # L1 is ~1.4 MB per core against 24 GB of DRAM, so an L1-destination arm needs its
            # own live budget or the group allocates past the end of L1 and the arm dies.
            budget = 8_000_000 if spec["arm"] == "to_memory_config_l1" else LIVE_BYTES
            try:
                row.update(time_arm(ttnn, dev, keep, fn, mode, ob, a.blocks,
                                    live_budget=budget))
            except Exception as e:                                             # noqa: BLE001
                row["error"] = "run failed: %r" % (e,)
            finally:
                for x in keep:
                    try:
                        ttnn.deallocate(x)
                    except Exception:                                          # noqa: BLE001
                        pass
            R["rows"].append(row)
            save()
            print(json.dumps({"key": row["key"], "s_per_call": row.get("s_per_call"),
                              "error": row.get("error")}), flush=True)

        run_roof("cube_close", roofs["cube"])
        R["power_after_W"] = board_power_W(a.node)
        R["completed"] = True
    finally:
        try:
            R["release_response"] = list(smc(fd, FORCE_AICLK, 0))
        except Exception as e:                                                 # noqa: BLE001
            R["errors"].append("clock release failed: %r" % (e,))
        os.close(fd)
        if sampler is not None:
            try:
                sampler.stdin.write(b"stop\n")
                sampler.stdin.flush()
                sampler.stdin.close()
            except Exception:                                                  # noqa: BLE001
                pass
            try:
                sampler.wait(timeout=10)
            except Exception:                                                  # noqa: BLE001
                sampler.kill()
        try:
            T.close_device() if hasattr(T, "close_device") else ttnn.close_device(dev)
        except Exception as e:                                                 # noqa: BLE001
            R["errors"].append("close_device failed: %r" % (e,))
        R["after"] = snapshot()
        R["ended_utc_ns"] = time.time_ns()
        save()

    # qualify every interval against the during-run clock samples
    samples = [json.loads(ln) for ln in (out / "clock.jsonl").read_text().splitlines() if ln]
    R["clock_samples"] = len(samples)
    mhz = [s["MHz"] for s in samples if "MHz" in s]
    R["clock_min_MHz"], R["clock_max_MHz"] = (min(mhz), max(mhz)) if mhz else (None, None)
    for coll in (R["rows"], list(R["roofs"].values())):
        for row in coll:
            if "marks" not in row:
                continue
            qual = [control.coverage(samples, m) for m in row["marks"]]
            row["clock"] = qual
            ok = [m for m, q in zip(row["marks"], qual) if q["pass"]]
            row["qualified_blocks"] = len(ok)
            row["s_per_call_qualified"] = min((m["s_per_call"] for m in ok), default=None)
    save()
    print("clock %s-%s MHz over %d samples" % (R["clock_min_MHz"], R["clock_max_MHz"],
                                               len(samples)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
