"""Exact 2D hierarchical device all-reduce of the RFD4 gradient (178,448,304 fp32/rank) on BH Galaxy.

Runs on an 8x4 Blackhole Galaxy mesh (FABRIC_1D_RING) and its submeshes (1x4, 2x4, 4x4, 8x4).
Each chip holds its own rank's 178,448,304-param fp32 gradient in 11 x 64 MB chunks on device.
Exact reduce-scatter along axis 0 then axis 1 via all_to_all_async_generic (in_dim=3, out_dim=2,
num_links=1) + rank-order fp32 ttnn.add, followed by all_gather (dim=2, Linear) along axis 1 then 0.
Also measures RS-only (rs2d, ZeRO-1 gradient phase) and a broken arm (rank N-1 zeroed on device;
must FAIL). Samples AICLK from sysfs during timed reps.
"""
import glob, hashlib, json, os, socket, sys, threading, time
import numpy as np, torch, ttnn

P, COLS, CR = 178_448_304, 1024, 16384  # 11 chunks x 64 MB fp32 (~3.4% zero tail pad)
NCH = -(-(-(-P // COLS)) // CR)
FULL_ROWS, FULL_COLS = 8, 4


def sample_aiclk():
    vals = []
    for p in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*/tt_aiclk")):
        try:
            vals.append(int(open(p).read().strip()))
        except OSError:
            pass
    return vals


class ClkSampler:
    def __init__(self, interval=0.05):
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


def exact_rs_2d(t, rm, cm, nl=1):
    """Exact fp32 2D reduce-scatter along axis 0 (size rm) then axis 1 (size cm)."""
    _, _, R, C = t.shape
    cur = t
    if rm > 1:
        topo0 = ttnn.Topology.Ring if rm == FULL_ROWS else ttnn.Topology.Linear
        y0 = ttnn.experimental.all_to_all_async_generic(
            cur, in_dim=3, out_dim=2, cluster_axis=0, topology=topo0, num_links=nl
        )
        r0 = R // rm
        s0 = ttnn.slice(y0, [0, 0, 0, 0], [1, 1, r0, C])
        for i in range(1, rm):
            s0 = ttnn.add(s0, ttnn.slice(y0, [0, 0, 0, i * C], [1, 1, r0, (i + 1) * C]))
        y0.deallocate()
        cur = s0
    if cm > 1:
        topo1 = ttnn.Topology.Ring if cm == FULL_COLS else ttnn.Topology.Linear
        _, _, r_in, _ = cur.shape
        y1 = ttnn.experimental.all_to_all_async_generic(
            cur, in_dim=3, out_dim=2, cluster_axis=1, topology=topo1, num_links=nl
        )
        if cur is not t:
            cur.deallocate()
        r1 = r_in // cm
        s1 = ttnn.slice(y1, [0, 0, 0, 0], [1, 1, r1, C])
        for i in range(1, cm):
            s1 = ttnn.add(s1, ttnn.slice(y1, [0, 0, 0, i * C], [1, 1, r1, (i + 1) * C]))
        y1.deallocate()
        cur = s1
    return cur


def exact_ag_2d(s, rm, cm, ag_topo=ttnn.Topology.Linear):
    """All-gather along axis 1 then axis 0 restoring [1, 1, R, C]."""
    cur = s
    if cm > 1:
        g1 = ttnn.all_gather(cur, dim=2, cluster_axis=1, topology=ag_topo)
        cur.deallocate()
        cur = g1
    if rm > 1:
        g0 = ttnn.all_gather(cur, dim=2, cluster_axis=0, topology=ag_topo)
        cur.deallocate()
        cur = g0
    return cur


def exact_ar_2d(t, rm, cm, nl=1):
    return exact_ag_2d(exact_rs_2d(t, rm, cm, nl=nl), rm, cm)


def chunk_data(c, N, rows=CR, cols=COLS, pad_p=P):
    g = torch.Generator().manual_seed(1000 + c)
    x = torch.randn(N, 1, rows, cols, generator=g)
    x *= torch.pow(10.0, -6 * torch.rand(N, 1, rows, 1, generator=g))
    lo = c * rows * cols
    if pad_p is not None and lo + rows * cols > pad_p:
        x.view(N, -1)[:, max(pad_p - lo, 0):] = 0
    return x


def ref_sums(x):
    N = x.shape[0]
    ser = x[0].clone()
    for i in range(1, N):
        ser.add_(x[i])
    rev = x[N - 1].clone()
    for i in range(N - 2, -1, -1):
        rev.add_(x[i])
    t = [x[i].clone() for i in range(N)]
    while len(t) > 1:
        t = [t[i] + t[i + 1] if i + 1 < len(t) else t[i] for i in range(0, len(t), 2)]
    floor = max(float((rev - ser).abs().max()), float((t[0] - ser).abs().max()))
    return ser, floor


def quick_probe(mesh, rm, cm):
    N = rm * cm
    R, C = 256, 1024
    x = chunk_data(0, N, rows=R, cols=C, pad_p=None)
    ser, floor = ref_sums(x)
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh,
                         mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    print(f"  probe {rm}x{cm}: running exact_rs_2d...", flush=True)
    rs = exact_rs_2d(tx, rm, cm)
    ttnn.synchronize_device(mesh)
    print(f"  probe {rm}x{cm}: rs shape={tuple(rs.shape)}, running exact_ag_2d...", flush=True)
    out = exact_ag_2d(rs, rm, cm)
    ttnn.synchronize_device(mesh)
    b = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    agree = all(torch.equal(b[0], b[i]) for i in range(N))
    err = float((b[0] - ser).abs().max())
    out.deallocate()
    tx.deallocate()
    ok = agree and err <= floor
    print(f"  probe {rm}x{cm}: agree={agree} err={err:.3e} floor={floor:.3e} PASS={ok}", flush=True)
    return ok


def bench_width(full, rm, cm, out_path, reps=4, run_rs_only=True):
    N = rm * cm
    sub = full if (rm, cm) == (FULL_ROWS, FULL_COLS) else full.create_submesh(ttnn.MeshShape(rm, cm))
    assert sub.get_num_devices() == N
    t0 = time.time()
    up = lambda x: ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=sub,
                                   mesh_mapper=ttnn.ShardTensorToMesh(sub, dim=0))
    grads, sers, floor = [], [], 0.0
    for c in range(NCH):
        x = chunk_data(c, N)
        s, f = ref_sums(x)
        sers.append(s)
        floor = max(floor, f)
        grads.append(up(x))
    print(f"width={N} ({rm}x{cm}): uploaded {NCH} chunks in {time.time() - t0:.1f}s, floor={floor:.3e}", flush=True)

    res = {
        "n": N, "shape": [rm, cm], "host": socket.gethostname(),
        "board": "BH-Galaxy-8x4", "fabric": "FABRIC_1D_RING",
        "params": P, "padded": NCH * CR * COLS, "chunks": NCH,
        "floor": floor, "reps_total": reps, "reps_warm": reps - 1,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {},
    }

    def run_arm(name, src, op, check_full=True):
        times, outs = [], None
        # Warmup rep 0
        ttnn.synchronize_device(sub)
        t = time.time()
        o0 = [op(g) for g in src]
        ttnn.synchronize_device(sub)
        t_cold = time.time() - t
        for z in o0:
            z.deallocate()
        # Timed warm reps 1..reps-1 with AICLK sampled during
        with ClkSampler(0.05) as clk:
            for r in range(reps - 1):
                ttnn.synchronize_device(sub)
                t = time.time()
                o = [op(g) for g in src]
                ttnn.synchronize_device(sub)
                times.append(time.time() - t)
                if r < reps - 2:
                    for z in o:
                        z.deallocate()
                else:
                    outs = o
        aiclk = clk.summary()
        med = float(np.median(times))
        info = {
            "s_cold": t_cold, "s": times, "s_med": med,
            "s_min": float(min(times)), "s_max": float(max(times)),
            "GBps_per_chip_payload": P * 4 / med / 1e9,
            "aiclk": aiclk,
        }
        if check_full:
            err, agree, h = 0.0, True, hashlib.sha256()
            for c, o in enumerate(outs):
                b = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(sub, dim=0))
                agree &= all(torch.equal(b[0], b[i]) for i in range(N))
                err = max(err, float((b[0] - sers[c]).abs().max()))
                h.update(b[0].numpy().tobytes())
                o.deallocate()
            ok = agree and err <= floor
            info.update({
                "max_err_vs_serial": err, "ranks_agree": agree,
                "verdict": "PASS" if ok else "FAIL", "digest": h.hexdigest()[:16],
            })
        else:
            # RS-only: verify each shard matches the corresponding slice of ser
            err, h = 0.0, hashlib.sha256()
            r_step = CR // N
            for c, o in enumerate(outs):
                b = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(sub, dim=0))
                for r_idx in range(N):
                    ref_slice = sers[c][:, r_idx * r_step:(r_idx + 1) * r_step, :]
                    err = max(err, float((b[r_idx] - ref_slice).abs().max()))
                h.update(b.numpy().tobytes())
                o.deallocate()
            info.update({
                "max_err_vs_serial": err,
                "verdict": "PASS" if err <= floor else "FAIL",
                "digest": h.hexdigest()[:16],
            })
        res["arms"][name] = info
        print(f"  {name}: med={med:.4f}s (warm={times}) err={info['max_err_vs_serial']:.3e} "
              f"verdict={info['verdict']} aiclk={aiclk['min']}/{aiclk['med']}/{aiclk['max']} dig={info['digest']}", flush=True)

    run_arm("a2a2d", grads, lambda g: exact_ar_2d(g, rm, cm), check_full=True)
    if run_rs_only:
        run_arm("rs2d", grads, lambda g: exact_rs_2d(g, rm, cm), check_full=False)

    # Broken arm: zero rank N-1 across all chunks on device (or fallback: zero rank N-1 on chunk 0)
    mv1 = torch.ones(N, 1, 1, 1)
    mv1[N - 1] = 0.0
    tmask1 = up(mv1)
    try:
        broken_in = [ttnn.multiply(g, tmask1) for g in grads]
        ttnn.synchronize_device(sub)
        tmask1.deallocate()
        run_arm("a2a2d_broken", broken_in, lambda g: exact_ar_2d(g, rm, cm), check_full=True)
        for z in broken_in:
            z.deallocate()
    except Exception as e:
        tmask1.deallocate()
        print(f"  mask broadcast fallback ({str(e)[:80]}): zeroing rank N-1 on chunk 0", flush=True)
        x0 = chunk_data(0, N)
        x0[N - 1] = 0
        b0 = up(x0)
        run_arm("a2a2d_broken", [b0] + grads[1:], lambda g: exact_ar_2d(g, rm, cm), check_full=True)
        b0.deallocate()
    for z in grads:
        z.deallocate()

    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {out_path}", flush=True)
    return res


def main():
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "32")))
    out_dir = sys.argv[1]
    widths_arg = sys.argv[2] if len(sys.argv) > 2 else "8x4,1x4,2x4,4x4"
    deadline_epoch = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    os.makedirs(out_dir, exist_ok=True)
    widths = [tuple(int(v) for v in w.split("x")) for w in widths_arg.split(",")]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    full = ttnn.open_mesh_device(ttnn.MeshShape(FULL_ROWS, FULL_COLS))
    print(f"OPENED 8x4 ({full.get_num_devices()}), initial aiclk={sample_aiclk()[:4]}", flush=True)
    try:
        # Quick 5s probe on first width to confirm op contract before full upload
        rm0, cm0 = widths[0]
        sub0 = full if (rm0, cm0) == (FULL_ROWS, FULL_COLS) else full.create_submesh(ttnn.MeshShape(rm0, cm0))
        assert quick_probe(sub0, rm0, cm0), "quick_probe failed"

        for rm, cm in widths:
            if deadline_epoch and time.time() > deadline_epoch:
                print(f"DEADLINE REACHED before {rm}x{cm}, stopping cleanly", flush=True)
                break
            out_path = os.path.join(out_dir, f"ring2d_{rm*cm}.json")
            bench_width(full, rm, cm, out_path, reps=4, run_rs_only=True)
    finally:
        ttnn.close_mesh_device(full)
        print("CLOSED MESH CLEANLY", flush=True)


if __name__ == "__main__":
    main()
