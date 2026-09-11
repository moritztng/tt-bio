#!/usr/bin/env python3
"""Where the pairformer Transition's device time goes, per op, per track, per silu arm.

`perf/b2x_op_cost/FINDINGS.md` prices one shipped pairformer block at 41.4152 ms against a cost
model of 17.67 ms (its bytes at the measured 445.0 GB/s roof plus its op count at the measured
6.36 us per-op device floor), and names the pair-track `Transition` the outlier at 3.21x: 8.705
ms/call where 474.2 MB is 1.07 ms and 258 ops is 1.64 ms.

The first thing this file found is that the 8.705 ms is not the pair track. Volume cannot tell
the two 4D Transition inputs apart at 512 aa -- the pair track is 1x512x512x128 and the MSA
track is 1x1024x512x64, both 33.55 M elements -- and the grab that produced the figure selected
on volume. Reproduced here: the call grabbed by that rule is the MSA-track one and it replays at
8.7114 ms, 0.07 % from the published 8.705. So the 3.21x ratio put the MSA track's device time
over the pair track's bytes and op count. This file measures both, each against its own bytes
and its own op count.

Two candidates for the deficit, both falsifiable:

  BYTES     the counter undercounts real DRAM traffic. Falsified if a buffer-address recount of
            the isolated call still cannot explain the measured time at 445.0 GB/s.
  PER-CORE  the bytes are right and the cores are idle or badly mapped. Falsified if the ops are
            near-fully parallel and the per-core work matches the byte count.

Legs, one device open, every number a ttnn trace replay (the host pays ~1.5 us per replay, so
nothing here is a dispatch measurement), and no per-op cost anywhere is a phase total divided by
an op count:

  bytes   graph-capture the call, count DRAM traffic per op with buffer-address dedupe
          (`perf/b2x_difflayer/real_traffic.py`, both ownership rules), plus the device-program
          count. The BYTES falsifier.
  floor   trace-replay the shipped call.
  ops     rebuild the swiglu body op by op at a ladder of row-block heights, with the module's
          own weights and memory configs, and replay each op ALONE. Summing the isolated ops
          against the shipped call separates "the deficit is in the ops" from "the deficit is in
          the sequence".
  grid    sweep `core_grid` on the body's fc1 matmul, squares and rectangles. The smallest grid
          whose time matches the full grid's IS the measured count of cores doing work. The
          PER-CORE falsifier.
  hsweep  replay the shipped call with TT_BIO_TRANSITION_H_CHUNK forced. Bytes and FLOPs are
          invariant in the row-block height; only op count and per-core work move.
  kmm     equal-FLOP matmuls at the body's K against fatter K, same fidelity and grid.

Every leg runs twice where the fused activation is involved: `activation="silu"` as shipped, and
the `TT_BIO_UNFUSED_SILU=1` arm. That defect is already root-caused elsewhere (ttnn computes
silu's sigmoid with the accurate Cody-Waite exp whatever the caller asks for, commit 1c70fc39 /
`state/protenix-trunk--z-silu-lowering-fix.md`); it is measured here only to price it on Boltz-2
at 512 aa, which no artifact had done.

No model code is changed. Every captured region is shipped code called with its own arguments.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

BW_EFF = 445.0e9          # measured asymptote, perf/b2x_op_cost/op_cost_curve.py
T_FIXED_US = 6.36         # measured per-op device floor, same file
PRIOR_MS = 8.7050         # subunit_floor_512_qb2c0.json, row labelled "Transition, pair track"

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


def trace_us(ttnn, dev, fn, R=1, n_replay=1, n_med=5, warm=3):
    """Device us per fn() call: a trace of R reps of fn, replayed n_replay times per timing."""
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    ok = True
    try:
        for _ in range(R):
            fn()
    except BaseException:
        ok = False
        raise
    finally:
        try:
            ttnn.end_trace_capture(dev, tid, cq_id=0)
        except Exception:
            pass
        if not ok:
            for f in (lambda: ttnn.release_trace(dev, tid),
                      lambda: ttnn.synchronize_device(dev)):
                try:
                    f()
                except Exception:
                    pass
    try:
        ttnn.synchronize_device(dev)
        for _ in range(2):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        per, cpu = [], []
        for _ in range(n_med):
            t0, c0 = time.perf_counter(), time.thread_time()
            for _ in range(n_replay):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            cpu.append(time.thread_time() - c0)
            ttnn.synchronize_device(dev)
            per.append((time.perf_counter() - t0) / (R * n_replay))
    finally:
        try:
            ttnn.release_trace(dev, tid)
        except Exception:
            pass
    return {"us": round(1e6 * st.median(per), 4), "R": R,
            "all_us": [round(1e6 * p, 4) for p in per],
            "spread_pct": round(100 * (max(per) - min(per)) / max(st.median(per), 1e-12), 2),
            "host_us_per_call": round(1e6 * st.median(cpu) / (R * n_replay), 3)}


def err(e):
    return {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}


def run_track(ttnn, T, torch, dev, grabbed, track, legs, hlist):
    """Every leg for one Transition input shape."""
    r: dict = {}
    mod, z = grabbed["obj"], grabbed["z"]
    H, W, C = int(z.shape[1]), int(z.shape[2]), int(z.shape[3])
    HID = int(mod.fc1_weight.shape[-1])
    ck = mod.compute_kernel_config
    dt = mod.dtype if mod.dtype is not None else T._dtype()
    h_ship = T.TRANSITION_H_CHUNK_SIZE_FAST if T._FAST_MODE else T.TRANSITION_H_CHUNK_SIZE
    if not T._FAST_MODE and W <= T.TRANSITION_H_CHUNK_BIG_MAX_W and C <= 256:
        h_ship = T.TRANSITION_H_CHUNK_SIZE_BIG
    z_b = H * W * C * 2
    r["shape"] = {"z": [int(d) for d in z.shape], "H": H, "W": W, "C": C, "hidden": HID,
                  "h_chunk_shipped": h_ship, "n_chunks_shipped": -(-H // h_ship),
                  "z_MB": round(z_b / 1e6, 3), "dtype": str(dt),
                  "l1_per_core_B_at_h": {
                      str(h): int(2 * h * W * (C + 2 * HID) / 110) for h in (16, 32, 64, 128)}}
    print(f"\n##### track={track}  " + json.dumps(r["shape"]), flush=True)

    def call_shipped():
        ttnn.deallocate(mod(z))

    # ---- bytes -------------------------------------------------------------------------------
    if "bytes" in legs:
        print("=== bytes: buffer-address dedupe on the shipped call ===", flush=True)
        try:
            import itemize as IT
            from real_traffic import counts
            ttnn.deallocate(mod(z))
            ttnn.synchronize_device(dev)
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            o = mod(z)
            ttnn.synchronize_device(dev)
            g = ttnn.graph.end_graph_capture()
            ttnn.deallocate(o)
            call = {"sig": f"{track}_transition", "nodes": g}
            row = {}
            for rule in ("stack", "range"):
                o_tls = IT.top_level_spans
                IT.top_level_spans = lambda nodes, _r=rule: o_tls(nodes, _r)
                try:
                    c = counts(call)
                finally:
                    IT.top_level_spans = o_tls
                row[rule] = {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in c.items() if k != "per_op"}
                agg: dict = {}
                for t_, n_, w_, rd_ in c["per_op"]:
                    q = agg.setdefault(n_, [0, 0, 0, 0])
                    q[0] += t_ / 1e6
                    q[1] += w_ / 1e6
                    q[2] += rd_ / 1e6
                    q[3] += 1
                row[rule]["by_op_MB"] = {k: [round(v[0], 3), round(v[1], 3), round(v[2], 3),
                                             v[3]] for k, v in
                                         sorted(agg.items(), key=lambda kv: -kv[1][0])}
            devops: dict = {}
            for n in g:
                if n.get("node_type") == "function_start":
                    nm = str((n.get("params") or {}).get("name", ""))
                    if nm.endswith("DeviceOperation"):
                        devops[nm] = devops.get(nm, 0) + 1
            row["devops"] = dict(sorted(devops.items(), key=lambda kv: -kv[1]))
            row["n_devops"] = sum(devops.values())
            real = row["range"]["real_MB"]
            row["bytes_explain_ms"] = round(1e3 * real * 1e6 / BW_EFF, 4)
            row["ops_explain_ms"] = round(1e-3 * row["n_devops"] * T_FIXED_US, 4)
            r["bytes"] = row
            print(f"  devops {row['n_devops']}  real {real:.1f} MB (range) / "
                  f"{row['stack']['real_MB']:.1f} MB (stack)  -> bytes {row['bytes_explain_ms']}"
                  f" ms + ops {row['ops_explain_ms']} ms", flush=True)
            for k, v in list(row["range"]["by_op_MB"].items())[:8]:
                print(f"     {v[0]:9.2f} MB  {k:30s} w {v[1]:8.2f} r {v[2]:8.2f} x{v[3]}",
                      flush=True)
        except Exception as e:                                              # noqa: BLE001
            r["bytes"] = err(e)
            print("  FAILED " + str(r["bytes"]), flush=True)

    # ---- floor, both silu arms ----------------------------------------------------------------
    if "floor" in legs:
        print("=== floor: the shipped call by trace replay, both silu arms ===", flush=True)
        r["floor"] = {}
        for arm, unf in (("fused", False), ("unfused", True)):
            T._UNFUSED_SILU = unf
            try:
                q = trace_us(ttnn, dev, call_shipped, R=1, n_replay=4)
                q["ms"] = round(q["us"] / 1e3, 4)
                r["floor"][arm] = q
                print(f"  {arm:8s} {q['ms']:9.4f} ms/call  spread {q['spread_pct']} %  "
                      f"host {q['host_us_per_call']} us", flush=True)
            except Exception as e:                                          # noqa: BLE001
                r["floor"][arm] = err(e)
                print(f"  {arm:8s} FAILED {r['floor'][arm]['error']}", flush=True)
        T._UNFUSED_SILU = False
        if all("ms" in r["floor"].get(k, {}) for k in ("fused", "unfused")):
            r["floor"]["unfuse_gain_x"] = round(
                r["floor"]["fused"]["ms"] / r["floor"]["unfused"]["ms"], 4)
            print(f"  unfuse gain on this call: {r['floor']['unfuse_gain_x']}x", flush=True)

    # ---- ops ----------------------------------------------------------------------------------
    if "ops" in legs:
        print("=== ops: the body op by op, isolated ===", flush=True)
        r["ops"] = {}
        for h in hlist:
            if h > H:
                continue
            n_chunk = -(-H // h)
            rows: dict = {}
            keep = []
            try:
                c_dram = z[:, 0:h]
                keep.append(c_dram)
                xn = ttnn.layer_norm(c_dram, weight=mod.norm_weight, bias=mod.norm_bias,
                                     epsilon=1e-5, compute_kernel_config=ck,
                                     memory_config=ttnn.L1_MEMORY_CONFIG)
                keep.append(xn)
                x1 = ttnn.linear(xn, mod.fc1_weight, compute_kernel_config=ck,
                                 memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dt,
                                 core_grid=T.CORE_GRID_MAIN)
                keep.append(x1)
                x2 = ttnn.linear(xn, mod.fc2_weight, compute_kernel_config=ck,
                                 memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dt,
                                 core_grid=T.CORE_GRID_MAIN)
                keep.append(x2)
                xm = ttnn.multiply(x1, x2, memory_config=ttnn.L1_MEMORY_CONFIG)
                keep.append(xm)
                ttnn.synchronize_device(dev)
                chunk_b, hid_b = h * W * C * 2, h * W * HID * 2
                w_b = C * HID * 2
                mm_flops = 2.0 * h * W * C * HID

                def d(x):
                    ttnn.deallocate(x)

                cands = [
                    ("layer_norm",
                     lambda: d(ttnn.layer_norm(c_dram, weight=mod.norm_weight,
                                               bias=mod.norm_bias, epsilon=1e-5,
                                               compute_kernel_config=ck,
                                               memory_config=ttnn.L1_MEMORY_CONFIG)),
                     chunk_b, 0.0),
                    ("fc1_fused_silu",
                     lambda: d(ttnn.linear(xn, mod.fc1_weight, activation="silu",
                                           compute_kernel_config=ck,
                                           memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dt,
                                           core_grid=T.CORE_GRID_MAIN)), w_b, mm_flops),
                    ("fc1_bare",
                     lambda: d(ttnn.linear(xn, mod.fc1_weight, compute_kernel_config=ck,
                                           memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dt,
                                           core_grid=T.CORE_GRID_MAIN)), w_b, mm_flops),
                    ("silu_standalone",
                     lambda: ttnn.silu(x1, memory_config=ttnn.L1_MEMORY_CONFIG,
                                       output_tensor=x1), 0, 0.0),
                    ("fc2",
                     lambda: d(ttnn.linear(xn, mod.fc2_weight, compute_kernel_config=ck,
                                           memory_config=ttnn.L1_MEMORY_CONFIG, dtype=dt,
                                           core_grid=T.CORE_GRID_MAIN)), w_b, mm_flops),
                    ("multiply_", lambda: ttnn.multiply_(xm, x2), 0, 0.0),
                    ("fc3_dram",
                     lambda: d(ttnn.linear(xm, mod.fc3_weight, compute_kernel_config=ck,
                                           dtype=dt, core_grid=T.CORE_GRID_MAIN,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)),
                     w_b + chunk_b, 2.0 * h * W * HID * C),
                ]
                R = max(2, min(48, int(3000 / max(1.0, (chunk_b + hid_b) / 1e6 * 8))))
                for nm, fn, dram_b, flops in cands:
                    try:
                        q = trace_us(ttnn, dev, fn, R=R)
                        q["dram_MB"] = round(dram_b / 1e6, 4)
                        q["GBs"] = round(dram_b / (q["us"] * 1e-6) / 1e9, 1) if dram_b else None
                        q["GFLOP"] = round(flops / 1e9, 4) if flops else None
                        q["TFLOPs"] = round(flops / (q["us"] * 1e-6) / 1e12, 2) if flops else None
                        rows[nm] = q
                        print(f"  h={h:4d} {nm:17s} {q['us']:9.3f} us "
                              f"{q['dram_MB']:8.3f} MB "
                              f"{('%7.1f GB/s' % q['GBs']) if q['GBs'] else '        -   '} "
                              f"{('%6.2f TF/s' % q['TFLOPs']) if q['TFLOPs'] else ''} "
                              f"(R={R} sp {q['spread_pct']}%)", flush=True)
                    except Exception as e:                                  # noqa: BLE001
                        rows[nm] = err(e)
                        print(f"  h={h:4d} {nm:17s} FAILED {rows[nm]['error']}", flush=True)
                        try:
                            ttnn.synchronize_device(dev)
                        except Exception:
                            pass
                g = lambda k: rows.get(k, {}).get("us", float("nan"))              # noqa: E731
                rows["_n_chunks"] = n_chunk
                rows["_body_us_fused"] = round(g("layer_norm") + g("fc1_fused_silu")
                                               + g("fc2") + g("multiply_") + g("fc3_dram"), 3)
                rows["_body_us_unfused"] = round(g("layer_norm") + g("fc1_bare")
                                                 + g("silu_standalone") + g("fc2")
                                                 + g("multiply_") + g("fc3_dram"), 3)
                rows["_sum_body_ms_fused"] = round(n_chunk * rows["_body_us_fused"] / 1e3, 4)
                rows["_sum_body_ms_unfused"] = round(n_chunk * rows["_body_us_unfused"] / 1e3, 4)
                r["ops"][str(h)] = rows
                print(f"  h={h:4d} body fused {rows['_body_us_fused']:.2f} us x {n_chunk} = "
                      f"{rows['_sum_body_ms_fused']:.4f} ms | unfused "
                      f"{rows['_body_us_unfused']:.2f} us x {n_chunk} = "
                      f"{rows['_sum_body_ms_unfused']:.4f} ms", flush=True)
            except Exception as e:                                          # noqa: BLE001
                r["ops"][str(h)] = {"setup_error": err(e)["error"]}
                print(f"  h={h} setup FAILED {r['ops'][str(h)]['setup_error']}", flush=True)
            for k in keep:
                try:
                    ttnn.deallocate(k)
                except Exception:
                    pass
            try:
                ttnn.synchronize_device(dev)
            except Exception:
                pass

        # the two whole-tensor ops, once per call and not per chunk
        try:
            n_chunk = -(-H // h_ship)
            chs = ttnn.chunk(z, n_chunk, dim=1)
            ttnn.synchronize_device(dev)
            outs = [ttnn.clone(c) for c in chs]
            ttnn.synchronize_device(dev)
            whole = {}
            whole["chunk"] = trace_us(
                ttnn, dev, lambda: [ttnn.deallocate(c) for c in ttnn.chunk(z, n_chunk, dim=1)],
                R=1, n_replay=8)
            whole["concat"] = trace_us(
                ttnn, dev, lambda: ttnn.deallocate(ttnn.concat(outs, dim=1)), R=1, n_replay=8)
            for k in ("chunk", "concat"):
                whole[k]["dram_MB"] = round(2 * z_b / 1e6, 3)
                whole[k]["GBs"] = round(2 * z_b / (whole[k]["us"] * 1e-6) / 1e9, 1)
                whole[k]["pct_roof"] = round(100 * 2 * z_b / (whole[k]["us"] * 1e-6) / BW_EFF, 1)
                print(f"  {k:7s} {whole[k]['us']:9.3f} us  {whole[k]['dram_MB']} MB  "
                      f"{whole[k]['GBs']} GB/s  {whole[k]['pct_roof']} % of roof", flush=True)
            for c in chs:
                ttnn.deallocate(c)
            for c in outs:
                ttnn.deallocate(c)
            r["whole_tensor_ops"] = whole
        except Exception as e:                                              # noqa: BLE001
            r["whole_tensor_ops"] = err(e)
            print("  whole-tensor FAILED " + str(r["whole_tensor_ops"]), flush=True)
        try:
            ttnn.synchronize_device(dev)
        except Exception:
            pass

    # ---- grid ---------------------------------------------------------------------------------
    if "grid" in legs:
        print("=== grid: core_grid sweep on the body's fc1, shipped chunk shape ===", flush=True)
        r["grid"] = {}
        try:
            c_dram = z[:, 0:h_ship]
            xn = ttnn.layer_norm(c_dram, weight=mod.norm_weight, bias=mod.norm_bias,
                                 epsilon=1e-5, compute_kernel_config=ck,
                                 memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.synchronize_device(dev)
            fl = 2.0 * h_ship * W * C * HID
            r["grid"]["_n_tiles"] = {"M": h_ship * W // 32, "N": HID // 32, "K": C // 32}
            for gx, gy in [(1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (11, 10),
                           (4, 10), (8, 10), (11, 4), (11, 8), (2, 10), (11, 2)]:
                nm = f"{gx}x{gy}"
                for arm, act in (("bare", None), ("silu", "silu")):
                    try:
                        cg = ttnn.CoreGrid(y=gy, x=gx)
                        q = trace_us(ttnn, dev,
                                     lambda cg=cg, act=act: ttnn.deallocate(
                                         ttnn.linear(xn, mod.fc1_weight, activation=act,
                                                     compute_kernel_config=ck,
                                                     memory_config=ttnn.L1_MEMORY_CONFIG,
                                                     dtype=dt, core_grid=cg)), R=8)
                        q["cores"] = gx * gy
                        q["TFLOPs"] = round(fl / (q["us"] * 1e-6) / 1e12, 3)
                        r["grid"][f"{nm}_{arm}"] = q
                        print(f"  {nm:6s} {arm:5s} {q['cores']:4d} cores {q['us']:9.3f} us  "
                              f"{q['TFLOPs']:6.3f} TFLOP/s (sp {q['spread_pct']}%)", flush=True)
                    except Exception as e:                                  # noqa: BLE001
                        r["grid"][f"{nm}_{arm}"] = err(e)
                        print(f"  {nm:6s} {arm:5s} FAILED "
                              f"{r['grid'][f'{nm}_{arm}']['error'][:80]}", flush=True)
                        try:
                            ttnn.synchronize_device(dev)
                        except Exception:
                            pass
            ttnn.deallocate(xn)
            ttnn.deallocate(c_dram)
        except Exception as e:                                              # noqa: BLE001
            r["grid"]["setup_error"] = err(e)["error"]
        try:
            ttnn.synchronize_device(dev)
        except Exception:
            pass

    # ---- hsweep -------------------------------------------------------------------------------
    if "hsweep" in legs:
        print("=== hsweep: the shipped call at forced row-block heights, both arms ===",
              flush=True)
        r["hsweep"] = {}
        for arm, unf in (("fused", False), ("unfused", True)):
            T._UNFUSED_SILU = unf
            for h in hlist:
                if h > H:
                    continue
                os.environ["TT_BIO_TRANSITION_H_CHUNK"] = str(h)
                try:
                    q = trace_us(ttnn, dev, call_shipped, R=1, n_replay=4)
                    q["ms"] = round(q["us"] / 1e3, 4)
                    q["n_chunks"] = -(-H // h)
                    r["hsweep"][f"{arm}_{h}"] = q
                    print(f"  {arm:8s} h={h:4d} ({q['n_chunks']:4d} chunks) {q['ms']:9.4f} ms "
                          f"sp {q['spread_pct']}%", flush=True)
                except Exception as e:                                      # noqa: BLE001
                    r["hsweep"][f"{arm}_{h}"] = err(e)
                    print(f"  {arm:8s} h={h:4d} FAILED "
                          f"{r['hsweep'][f'{arm}_{h}']['error'][:90]}", flush=True)
                    try:
                        ttnn.synchronize_device(dev)
                    except Exception:
                        pass
            os.environ.pop("TT_BIO_TRANSITION_H_CHUNK", None)
        T._UNFUSED_SILU = False

    # ---- kmm ----------------------------------------------------------------------------------
    if "kmm" in legs:
        print("=== kmm: equal-FLOP matmuls, the body's K against fatter K ===", flush=True)
        r["kmm"] = {}
        f0 = 2.0 * (h_ship * W) * C * HID
        for k in (C, 2 * C, 4 * C, 8 * C, 16 * C):
            m = max(32, (int(f0 / (2.0 * k * HID)) // 32) * 32)
            nm = f"K{k}"
            try:
                aa = ttnn.from_torch(torch.randn(m, k), layout=ttnn.TILE_LAYOUT,
                                     dtype=ttnn.bfloat16, device=dev,
                                     memory_config=ttnn.L1_MEMORY_CONFIG)
                bb = ttnn.from_torch(torch.randn(k, HID), layout=ttnn.TILE_LAYOUT,
                                     dtype=ttnn.bfloat16, device=dev,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
                fl = 2.0 * m * k * HID
                q = trace_us(ttnn, dev,
                             lambda: ttnn.deallocate(
                                 ttnn.matmul(aa, bb, compute_kernel_config=ck,
                                             memory_config=ttnn.L1_MEMORY_CONFIG,
                                             dtype=dt, core_grid=T.CORE_GRID_MAIN)), R=8)
                q.update({"M": m, "K": k, "N": HID, "GFLOP": round(fl / 1e9, 4),
                          "TFLOPs": round(fl / (q["us"] * 1e-6) / 1e12, 3)})
                r["kmm"][nm] = q
                print(f"  {nm:7s} M={m:7d} N={HID:5d} {q['us']:9.3f} us {q['GFLOP']:7.3f} GFLOP"
                      f"  {q['TFLOPs']:6.3f} TFLOP/s", flush=True)
                ttnn.deallocate(aa)
                ttnn.deallocate(bb)
            except Exception as e:                                          # noqa: BLE001
                r["kmm"][nm] = err(e)
                print(f"  {nm:7s} FAILED {r['kmm'][nm]['error'][:90]}", flush=True)
                try:
                    ttnn.synchronize_device(dev)
                except Exception:
                    pass
    return r


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--legs", default="bytes,floor,ops,grid,hsweep,kmm")
    ap.add_argument("--hlist", default="8,16,32,64,128")
    a = ap.parse_args()
    OUT_PATH = a.out
    legs = [x for x in a.legs.split(",") if x]
    hlist = [int(x) for x in a.hlist.split(",")]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=1 << 30)
    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "loadavg_at_start": loadavg(), "size": a.size,
                  "BW_eff_GBs": BW_EFF / 1e9, "t_fixed_us": T_FIXED_US,
                  "prior_subunit_ms": PRIOR_MS}
    dump()
    one_fold, meta, state = B.build_fold("boltz2", HERE / f".msa_{a.size}", tgt, a3m)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dev = T.get_device()
    OUT["env"]["grid_str"] = str(dev.compute_with_storage_grid_size())
    dump()

    grabs: dict = {}
    seen: dict = {}
    n = {"calls": 0}
    cls = T.Transition
    orig = cls.__dict__["__call__"]

    def wrapped(self_obj, *args, **kw):
        n["calls"] += 1
        if args and hasattr(args[0], "shape") and len(args[0].shape) == 4:
            sh = tuple(int(d) for d in args[0].shape)
            seen[str(sh)] = seen.get(str(sh), 0) + 1
            # Volume cannot tell the two tracks apart at 512 aa: 1x512x512x128 and 1x1024x512x64
            # are both 33.55 M elements. The pair track is square in its two token axes.
            key = "pair" if sh[1] == sh[2] else "msa"
            if key not in grabs and seen[str(sh)] >= 3:
                grabs[key] = {"obj": self_obj, "z": ttnn.clone(args[0])}
                print(f"  grabbed {key}-track Transition, z={sh}", flush=True)
        return orig(self_obj, *args, **kw)
    cls.__call__ = wrapped
    print("=== fold (warms kernels, grabs one call of each track) ===", flush=True)
    t, m = one_fold()
    cls.__call__ = orig
    OUT["fold_s"] = round(t, 3)
    OUT["transition_calls_per_fold"] = n["calls"]
    OUT["transition_4d_shapes_per_fold"] = seen
    print("  4D Transition shapes/fold: " + json.dumps(seen), flush=True)
    dump()
    if not grabs:
        print("FAILED to grab any 4D Transition", flush=True)
        return 1

    OUT["tracks"] = {}
    for track in [k for k in ("pair", "msa") if k in grabs]:
        try:
            OUT["tracks"][track] = run_track(ttnn, T, torch, dev, grabs[track], track, legs,
                                             hlist)
        except Exception as e:                                              # noqa: BLE001
            OUT["tracks"][track] = err(e)
            print(f"  track {track} ABORTED {OUT['tracks'][track]['error']}", flush=True)
        dump()

    OUT["env"]["loadavg_at_end"] = loadavg()
    OUT["env"]["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
