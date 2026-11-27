"""Exact 2D hierarchical device all-reduce of the RFD4 gradient (178,448,304 fp32/rank) on BH Galaxy.

Delegates all collective logic to ``tt_bio.train.ccl`` so the benchmark exercises the exact
shipped implementation. Runs on an 8x4 Blackhole Galaxy mesh (FABRIC_1D_RING, max_payload=8192 B)
and its submeshes (8x4, 4x4, 2x4, 1x4). Samples AICLK from sysfs during timed warm reps.

IMPORTANT (tt-metal v0.68.0 mesh_device.cpp:861): only ONE MeshDevice (either full 8x4 OR one
child submesh) may ever touch CQ 0 per process. When multiple ``--widths`` are given, this script
spawns a clean subprocess per width so parent/child and sequential submesh CQs never collide.
"""
import argparse, glob, hashlib, json, math, os, socket, subprocess, sys, threading, time
import numpy as np, torch, ttnn

from tt_bio.train.ccl import (
    DEFAULT_FULL_MESH,
    FABRIC_FP32_MAX_PAYLOAD_BYTES,
    DeviceCCLReducer,
    configure_fp32_ring_fabric,
    exact_2d_fp32_sum,
    exact_all_gather_2d,
    exact_all_reduce_2d,
    exact_reduce_scatter_2d,
    flat_2d_shape,
    fp32_reduction_order_floor,
    plan_reduction_groups,
)

P = 178_448_304
FULL_ROWS, FULL_COLS = DEFAULT_FULL_MESH
# [1, 1, 16384, 10944] = 512 x 342 tiles = 179,306,496 fp32 (0.48% zero tail pad).
# Uploaded/read back in 4 horizontal strips of [1, 1, 4096, 10944] so N=32 host tensors
# stay at 1.43e9 elems (< int32_max).
_, _, TOTAL_R, COLS = flat_2d_shape(P, DEFAULT_FULL_MESH, cols=10944)
N_STRIPS = 4
assert TOTAL_R % N_STRIPS == 0
STRIP_R = TOTAL_R // N_STRIPS

# Exact RFD4 558-parameter shape counts (sum = 178,448,304 fp32 elems, matching checkpoint nn.*).
RFD4_PARAM_SHAPE_COUNTS = [
    ((6144, 768), 14),
    ((768, 3072), 14),
    ((2304, 768), 14),
    ((768, 768), 38),
    ((768, 256), 84),
    ((256, 384), 56),
    ((384, 384), 15),
    ((768, 384), 7),
    ((384, 768), 7),
    ((384, 256), 16),
    ((768, 897), 1),
    ((1024, 256), 2),
    ((256, 512), 3),
    ((256, 609), 1),
    ((768,), 151),
    ((147, 768), 1),
    ((12, 384), 14),
    ((1473, 32), 1),
    ((384, 107), 1),
    ((2304,), 14),
    ((32, 768), 1),
    ((384,), 52),
    ((256,), 44),
    ((3, 768), 1),
    ((128, 16), 1),
    ((44, 32), 1),
    ((7, 32), 1),
    ((6, 32), 1),
    ((11, 16), 1),
    ((4, 16), 1),
]


def sample_aiclk():
    vals = []
    for p in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*/tt_aiclk")):
        try:
            vals.append(int(open(p).read().strip()))
        except OSError:
            pass
    return vals


class ClkSampler:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            v = sample_aiclk()
            if v:
                self.samples.extend(v)
            self._stop.wait(self.interval)

    def __enter__(self):
        v = sample_aiclk()
        if v:
            self.samples.extend(v)
        self._thr.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thr.join(timeout=1.0)
        v = sample_aiclk()
        if v:
            self.samples.extend(v)

    def summary(self):
        if not self.samples:
            return {"min": 0, "med": 0, "max": 0, "n_reads": 0}
        arr = np.array(self.samples, dtype=np.int64)
        return {"min": int(arr.min()), "med": int(np.median(arr)), "max": int(arr.max()), "n_reads": int(arr.size)}


def strip_data(s_idx, N, rows=STRIP_R, cols=COLS, pad_p=P):
    g = torch.Generator().manual_seed(1000 + s_idx)
    x = torch.randn(N, 1, rows, cols, generator=g)
    x *= torch.pow(10.0, -6 * torch.rand(N, 1, rows, 1, generator=g))
    lo = s_idx * rows * cols
    if pad_p is not None and lo + rows * cols > pad_p:
        x.view(N, -1)[:, max(pad_p - lo, 0):] = 0
    return x


def quick_probe(mesh, rm, cm, nl=2, ag_ring=True, force_linear=False):
    N = rm * cm
    R, C = 1024, 1024
    x = strip_data(0, N, rows=R, cols=C, pad_p=None)
    ser, floor = fp32_reduction_order_floor(x)
    tx = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    t0 = time.time()
    rs = exact_reduce_scatter_2d(tx, (rm, cm), num_links=nl, force_linear=force_linear)
    ttnn.synchronize_device(mesh)
    t_rs = time.time() - t0
    t1 = time.time()
    out = exact_all_gather_2d(rs, (rm, cm), num_links=nl, ag_ring=ag_ring)
    ttnn.synchronize_device(mesh)
    t_ag = time.time() - t1
    b = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    agree = all(torch.equal(b[0], b[i]) for i in range(N))
    err = float((b[0] - ser).abs().max())
    out.deallocate()
    tx.deallocate()
    ok = agree and err <= floor
    print(f"  probe {rm}x{cm} nl={nl} ag_ring={ag_ring} lin={force_linear}: rs={t_rs:.3f}s ag={t_ag:.3f}s "
          f"agree={agree} err={err:.3e} floor={floor:.3e} PASS={ok}", flush=True)
    return ok


def upload_full_grad(sub, N):
    """Upload N_STRIPS strips (< int32_max elems each) and concat on device into [1, 1, TOTAL_R, COLS]."""
    up = lambda x: ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=sub,
        mesh_mapper=ttnn.ShardTensorToMesh(sub, dim=0),
    )
    strips, sers, floor = [], [], 0.0
    for s_idx in range(N_STRIPS):
        x = strip_data(s_idx, N)
        ser, f = fp32_reduction_order_floor(x)
        sers.append(ser)
        floor = max(floor, f)
        strips.append(up(x))
    full_t = ttnn.concat(strips, dim=2)
    ttnn.synchronize_device(sub)
    for st in strips:
        st.deallocate()
    return full_t, sers, floor


def verify_full_output(sub, out, sers, N):
    """Read back [1, 1, TOTAL_R, COLS] in N_STRIPS slices (< int32_max elems each) and grade."""
    err, agree, h = 0.0, True, hashlib.sha256()
    for s_idx in range(N_STRIPS):
        sl = ttnn.slice(out, [0, 0, s_idx * STRIP_R, 0], [1, 1, (s_idx + 1) * STRIP_R, COLS])
        b = ttnn.to_torch(sl, mesh_composer=ttnn.ConcatMeshToTensor(sub, dim=0))
        sl.deallocate()
        agree &= all(torch.equal(b[0], b[i]) for i in range(N))
        err = max(err, float((b[0] - sers[s_idx]).abs().max()))
        h.update(b[0].numpy().tobytes())
    return err, agree, h.hexdigest()[:16]


def verify_rs_output(sub, out, sers, N):
    """Verify each rank's RS shard matches the corresponding row range of concatenated sers."""
    b = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(sub, dim=0))
    r_step = TOTAL_R // N
    err, h = 0.0, hashlib.sha256()
    for r_idx in range(N):
        lo = r_idx * r_step
        hi = lo + r_step
        parts = []
        pos = lo
        while pos < hi:
            s_idx = pos // STRIP_R
            s_lo = pos - s_idx * STRIP_R
            s_hi = min(STRIP_R, s_lo + (hi - pos))
            parts.append(sers[s_idx][:, s_lo:s_hi, :])
            pos += s_hi - s_lo
        ref_slice = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        err = max(err, float((b[r_idx] - ref_slice).abs().max()))
    h.update(b.numpy().tobytes())
    return err, h.hexdigest()[:16]


def bench_on_mesh(sub, rm, cm, out_path, nl=2, ag_ring=True, force_linear=False, reps=5, max_payload=8192, sha=""):
    N = rm * cm
    t0 = time.time()
    grad, sers, floor = upload_full_grad(sub, N)
    print(f"width={N} ({rm}x{cm}) nl={nl}: uploaded 1x[{TOTAL_R},{COLS}] in {time.time() - t0:.1f}s, "
          f"floor={floor:.3e}", flush=True)

    res = {
        "n": N, "shape": [rm, cm], "host": socket.gethostname(),
        "board": "BH-Galaxy-8x4", "fabric": "FABRIC_1D_RING", "sha": sha,
        "max_packet_payload_bytes": max_payload, "num_links": nl,
        "ag_ring": ag_ring, "force_linear": force_linear,
        "params": P, "padded": TOTAL_R * COLS, "tensor_shape": [1, 1, TOTAL_R, COLS],
        "floor": floor, "reps_total": reps, "reps_warm": reps - 1,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {},
    }

    def run_arm(name, src, op, check_full=True):
        times, out_last = [], None
        ttnn.synchronize_device(sub)
        t = time.time()
        o0 = op(src)
        ttnn.synchronize_device(sub)
        t_cold = time.time() - t
        print(f"    [{name}] cold={t_cold:.4f}s", flush=True)
        o0.deallocate()
        with ClkSampler(0.01) as clk:
            for r in range(reps - 1):
                ttnn.synchronize_device(sub)
                t = time.time()
                o = op(src)
                ttnn.synchronize_device(sub)
                times.append(time.time() - t)
                if r < reps - 2:
                    o.deallocate()
                else:
                    out_last = o
        aiclk = clk.summary()
        med = float(np.median(times))
        info = {
            "s_cold": t_cold, "s": times, "s_med": med,
            "s_min": float(min(times)), "s_max": float(max(times)),
            "GBps_per_chip_payload": P * 4 / med / 1e9,
            "aiclk": aiclk,
        }
        if check_full:
            err, agree, dig = verify_full_output(sub, out_last, sers, N)
            out_last.deallocate()
            ok = agree and err <= floor
            info.update({
                "max_err_vs_serial": err, "ranks_agree": agree,
                "verdict": "PASS" if ok else "FAIL", "digest": dig,
            })
        else:
            err, dig = verify_rs_output(sub, out_last, sers, N)
            out_last.deallocate()
            info.update({
                "max_err_vs_serial": err,
                "verdict": "PASS" if err <= floor else "FAIL", "digest": dig,
            })
        res["arms"][name] = info
        print(f"  {name}: med={med:.4f}s (cold={t_cold:.4f}s warm={[round(x,4) for x in times]}) "
              f"err={info['max_err_vs_serial']:.3e} verdict={info['verdict']} "
              f"aiclk={aiclk['min']}/{aiclk['med']}/{aiclk['max']}({aiclk['n_reads']}) dig={info['digest']}",
              flush=True)

    # First time RS alone so we see immediately whether RS or AG behaves differently on submeshes.
    run_arm(
        "rs2d", grad,
        lambda g: exact_reduce_scatter_2d(g, (rm, cm), num_links=nl, force_linear=force_linear),
        check_full=False,
    )
    run_arm(
        "a2a2d", grad,
        lambda g: exact_all_reduce_2d(g, (rm, cm), num_links=nl, ag_ring=ag_ring, force_linear=force_linear),
        check_full=True,
    )

    # Broken arm: zero rank N-1 on device (must FAIL)
    mv = torch.ones(N, 1, 1, 1)
    mv[N - 1] = 0.0
    tmask = ttnn.from_torch(
        mv, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=sub,
        mesh_mapper=ttnn.ShardTensorToMesh(sub, dim=0),
    )
    try:
        broken_in = ttnn.multiply(grad, tmask)
        ttnn.synchronize_device(sub)
        tmask.deallocate()
        run_arm(
            "a2a2d_broken", broken_in,
            lambda g: exact_all_reduce_2d(g, (rm, cm), num_links=nl, ag_ring=ag_ring, force_linear=force_linear),
            check_full=True,
        )
        broken_in.deallocate()
    except Exception as e:
        tmask.deallocate()
        print(f"  mask broadcast fallback ({str(e)[:80]}): zeroing rank N-1 on strip 0", flush=True)
        up = lambda x: ttnn.from_torch(
            x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=sub,
            mesh_mapper=ttnn.ShardTensorToMesh(sub, dim=0),
        )
        x0 = strip_data(0, N)
        x0[N - 1] = 0.0
        b0 = up(x0)
        rest = ttnn.slice(grad, [0, 0, STRIP_R, 0], [1, 1, TOTAL_R, COLS])
        broken_in = ttnn.concat([b0, rest], dim=2)
        ttnn.synchronize_device(sub)
        b0.deallocate()
        rest.deallocate()
        run_arm(
            "a2a2d_broken", broken_in,
            lambda g: exact_all_reduce_2d(g, (rm, cm), num_links=nl, ag_ring=ag_ring, force_linear=force_linear),
            check_full=True,
        )
        broken_in.deallocate()

    grad.deallocate()
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {out_path}", flush=True)
    return res


def bench_dict_on_mesh(sub, rm, cm, out_path, nl=2, ag_ring=True, force_linear=False, reps=5, max_payload=8192, sha=""):
    """Benchmark DeviceCCLReducer directly on all 558 real RFD4 parameter shapes (178,448,304 fp32)."""
    N = rm * cm
    shapes = {}
    idx = 0
    for shp, cnt in RFD4_PARAM_SHAPE_COUNTS:
        for j in range(cnt):
            shapes[f"p{idx:03d}_{'x'.join(map(str, shp))}_{j}"] = shp
            idx += 1
    assert len(shapes) == 558 and sum(math.prod(s) for s in shapes.values()) == P
    plan = plan_reduction_groups(shapes, (rm, cm))
    print(f"dict-bench {rm}x{cm}: 558 params ({P:,} elems) -> {len(plan.width_groups)} width groups "
          f"({[wg.cols for wg in plan.width_groups]}) + {len(plan.small_groups)} small groups", flush=True)

    # Pick first + last tensor from EVERY width group, EVERY folded small slot, and EVERY small group.
    check_names = []
    for wg in plan.width_groups:
        check_names.append(wg.entries[0][0])
        if len(wg.entries) > 1:
            check_names.append(wg.entries[-1][0])
        for slot in wg.folded_smalls:
            check_names.append(slot.entries[0][0])
            if len(slot.entries) > 1:
                check_names.append(slot.entries[-1][0])
    for sg in plan.small_groups:
        check_names.append(sg.entries[0][0])
        if len(sg.entries) > 1:
            check_names.append(sg.entries[-1][0])
    check_set = set(check_names)

    # Upload genuinely independent per-rank fp32 data for every graded tensor (zero device pre-op
    # like ttnn.multiply, which truncates fp32 -> TF32 on Blackhole), and replicated fp32 data for
    # the remaining ungraded tensors so upload stays fast (~5-10s) while all 558 tensors (178.4M
    # params) are reduced on device every iteration.
    t0 = time.time()
    dev_grads = {}
    ref_ranks = {}
    g_rng = torch.Generator().manual_seed(777)
    for name, shp in shapes.items():
        mag = 10.0 ** (-4.0 * float(torch.rand(1, generator=g_rng).item()))
        if name in check_set:
            ranks = torch.randn((N, *shp), generator=g_rng, dtype=torch.float32) * mag
            ref_ranks[name] = ranks
            t_sh = ttnn.from_torch(
                ranks.reshape(N, 1, *shp),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=sub,
                mesh_mapper=ttnn.ShardTensorToMesh(sub, dim=0),
            )
            t_dev = ttnn.reshape(t_sh, list(shp))
            if t_dev is not t_sh and t_dev.buffer_address() != t_sh.buffer_address():
                t_sh.deallocate()
            dev_grads[name] = t_dev
        else:
            base = torch.randn(shp, generator=g_rng, dtype=torch.float32) * mag
            dev_grads[name] = ttnn.from_torch(base, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=sub)
    ttnn.synchronize_device(sub)
    print(f"  uploaded 558 device grads ({len(check_names)} per-rank graded) in {time.time() - t0:.1f}s", flush=True)

    reducer = DeviceCCLReducer((rm, cm), num_links=nl, ag_ring=ag_ring, force_linear=force_linear)

    def run_dict_arm(arm_name, src_grads, check_names):
        ttnn.synchronize_device(sub)
        t = time.time()
        o0 = reducer(src_grads)
        ttnn.synchronize_device(sub)
        t_cold = time.time() - t
        for v in o0.values():
            if v is not None and v.is_allocated():
                ttnn.deallocate(v)
        times, out_last = [], None
        with ClkSampler(0.01) as clk:
            for r in range(reps - 1):
                ttnn.synchronize_device(sub)
                t = time.time()
                o = reducer(src_grads)
                ttnn.synchronize_device(sub)
                times.append(time.time() - t)
                if r < reps - 2:
                    for v in o.values():
                        if v is not None and v.is_allocated():
                            ttnn.deallocate(v)
                else:
                    out_last = o
        aiclk = clk.summary()
        med = float(np.median(times))
        # Verify EVERY graded tensor individually against BOTH exact CPU 2D IEEE-754 tree sum (0.0 ULP)
        # AND its own 4-order fp32 reduction-order floor.
        max_err, max_err_2d, max_floor, agree, all_under_floor, h = 0.0, 0.0, 0.0, True, True, hashlib.sha256()
        worst = {"name": "", "err": 0.0, "floor": 0.0, "ratio": 0.0}
        for nm in check_names:
            t_out = out_last[nm]
            shp = shapes[nm]
            v_out = ttnn.reshape(t_out, [1, 1, *shp])
            b = ttnn.to_torch(
                v_out,
                mesh_composer=ttnn.ConcatMeshToTensor(sub, dim=0),
            ).reshape(N, *shp)
            if v_out is not t_out and v_out.buffer_address() != t_out.buffer_address():
                v_out.deallocate()
            agree &= all(torch.equal(b[0], b[i]) for i in range(N))
            ref_2d = exact_2d_fp32_sum(ref_ranks[nm], (rm, cm), tree=True)
            err_2d = float((b[0] - ref_2d).abs().max())
            max_err_2d = max(max_err_2d, err_2d)
            ser, fl = fp32_reduction_order_floor(ref_ranks[nm])
            err = float((b[0] - ser).abs().max())
            lim = max(fl, 1e-7)
            all_under_floor &= (err <= lim)
            ratio = err / lim
            if ratio > worst["ratio"]:
                worst = {"name": nm, "err": err, "floor": fl, "ratio": ratio}
            max_err = max(max_err, err)
            max_floor = max(max_floor, fl)
            h.update(b[0].numpy().tobytes())
        for v in out_last.values():
            if v is not None and v.is_allocated():
                ttnn.deallocate(v)
        ok = agree and all_under_floor and (max_err_2d == 0.0)
        info = {
            "s_cold": t_cold, "s": times, "s_med": med,
            "s_min": float(min(times)), "s_max": float(max(times)),
            "GBps_per_chip_payload": P * 4 / med / 1e9,
            "max_err_vs_serial": max_err, "max_err_vs_2d_exact": max_err_2d, "floor": max_floor,
            "worst_tensor": worst,
            "ranks_agree": agree, "verdict": "PASS" if ok else "FAIL",
            "digest": h.hexdigest()[:16], "aiclk": aiclk,
            "checked_tensors": len(check_names),
        }
        print(f"  {arm_name}: med={med:.4f}s (cold={t_cold:.4f}s warm={[round(x,4) for x in times]}) "
              f"err_2d={max_err_2d:.3e} err_ser={max_err:.3e} floor={max_floor:.3e} "
              f"worst={worst['name']}({worst['ratio']:.2f}x) agree={agree} verdict={info['verdict']} "
              f"aiclk={aiclk['min']}/{aiclk['med']}/{aiclk['max']}({aiclk['n_reads']}) dig={info['digest']}",
              flush=True)
        return info

    res = {
        "mode": "dict_558", "n": N, "shape": [rm, cm], "host": socket.gethostname(),
        "board": "BH-Galaxy-8x4", "fabric": "FABRIC_1D_RING", "sha": sha,
        "max_packet_payload_bytes": max_payload, "num_links": nl,
        "ag_ring": ag_ring, "force_linear": force_linear,
        "params": P, "n_tensors": 558,
        "width_groups": [{"cols": wg.cols, "rows": wg.total_rows, "padded": wg.padded_rows,
                          "n": len(wg.entries), "folded_smalls": len(wg.folded_smalls)}
                         for wg in plan.width_groups],
        "small_groups": [{"m": sg.m, "n": sg.n, "batch": sg.total_batch, "cnt": len(sg.entries)}
                         for sg in plan.small_groups],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {},
    }
    res["arms"]["dict_558"] = run_dict_arm("dict_558", dev_grads, check_names)

    # Broken arm: zero rank N-1 on the first width-group tensor on host (must FAIL)
    first_nm = plan.width_groups[0].entries[0][0]
    first_shp = shapes[first_nm]
    broken_ranks = ref_ranks[first_nm].clone()
    broken_ranks[N - 1] = 0.0
    t_brk = ttnn.from_torch(
        broken_ranks.reshape(N, 1, *first_shp),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=sub,
        mesh_mapper=ttnn.ShardTensorToMesh(sub, dim=0),
    )
    t_brk_dev = ttnn.reshape(t_brk, list(first_shp))
    if t_brk_dev is not t_brk and t_brk_dev.buffer_address() != t_brk.buffer_address():
        t_brk.deallocate()
    broken_grads = dict(dev_grads)
    broken_grads[first_nm] = t_brk_dev
    ttnn.synchronize_device(sub)
    res["arms"]["dict_558_broken"] = run_dict_arm("dict_558_broken", broken_grads, check_names)
    broken_grads[first_nm].deallocate()

    reducer.deallocate_caches()
    for v in dev_grads.values():
        if v.is_allocated():
            ttnn.deallocate(v)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {out_path}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--widths", default="8x4,4x4,2x4,1x4")
    ap.add_argument("--num-links", type=int, default=2)
    ap.add_argument("--max-payload", type=int, default=FABRIC_FP32_MAX_PAYLOAD_BYTES)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ag-linear", action="store_true")
    ap.add_argument("--force-linear", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--dict-bench", action="store_true")
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--sha", default="")
    args = ap.parse_args()

    sha = args.sha
    if not sha:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short=9", "HEAD"],
                cwd=os.path.dirname(__file__), text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            sha = "unknown"

    widths = [tuple(int(v) for v in w.split("x")) for w in args.widths.split(",")]
    # One clean process per width so parent/child and sequential submesh CQs never collide (mesh_device.cpp:861).
    if len(widths) > 1 and not args.probe_only:
        for rm, cm in widths:
            if args.deadline and time.time() > args.deadline:
                print(f"DEADLINE REACHED before {rm}x{cm}, stopping cleanly", flush=True)
                break
            cmd = [
                sys.executable, os.path.abspath(__file__), args.out_dir,
                "--widths", f"{rm}x{cm}",
                "--num-links", str(args.num_links),
                "--max-payload", str(args.max_payload),
                "--reps", str(args.reps),
                "--sha", sha,
            ]
            if args.ag_linear:
                cmd.append("--ag-linear")
            if args.force_linear:
                cmd.append("--force-linear")
            if args.no_probe:
                cmd.append("--no-probe")
            if args.dict_bench:
                cmd.append("--dict-bench")
            if args.deadline:
                cmd.extend(["--deadline", str(args.deadline)])
            print(f"=== SPAWN {rm}x{cm} ===", flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"ERROR: width {rm}x{cm} exited rc={rc}", flush=True)
                sys.exit(rc)
        return

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "32")))
    os.makedirs(args.out_dir, exist_ok=True)
    rm, cm = widths[0]

    configure_fp32_ring_fabric(args.max_payload)
    full = ttnn.open_mesh_device(ttnn.MeshShape(FULL_ROWS, FULL_COLS))
    sub = full if (rm, cm) == (FULL_ROWS, FULL_COLS) else full.create_submesh(ttnn.MeshShape(rm, cm))
    assert sub.get_num_devices() == rm * cm
    print(f"OPENED {rm}x{cm} ({sub.get_num_devices()} on 8x4), max_payload={args.max_payload}, "
          f"nl={args.num_links}, ag_ring={not args.ag_linear}, lin={args.force_linear}, sha={sha}, "
          f"initial aiclk={sample_aiclk()[:4]}", flush=True)
    try:
        if not args.no_probe:
            ok = quick_probe(sub, rm, cm, nl=args.num_links, ag_ring=not args.ag_linear, force_linear=args.force_linear)
            assert ok, f"quick_probe failed on {rm}x{cm}"
        if args.probe_only:
            return
        if args.dict_bench:
            out_path = os.path.join(args.out_dir, f"dict558_{rm * cm}.json")
            bench_dict_on_mesh(
                sub, rm, cm, out_path,
                nl=args.num_links, ag_ring=not args.ag_linear, force_linear=args.force_linear,
                reps=args.reps, max_payload=args.max_payload, sha=sha,
            )
        else:
            out_path = os.path.join(args.out_dir, f"ring2d_{rm * cm}.json")
            bench_on_mesh(
                sub, rm, cm, out_path,
                nl=args.num_links, ag_ring=not args.ag_linear, force_linear=args.force_linear,
                reps=args.reps, max_payload=args.max_payload, sha=sha,
            )
    finally:
        if sub is not full:
            ttnn.synchronize_device(sub)
            ttnn.close_mesh_device(sub)
        ttnn.close_mesh_device(full)
        print(f"CLOSED {rm}x{cm} MESH CLEANLY", flush=True)


if __name__ == "__main__":
    main()
