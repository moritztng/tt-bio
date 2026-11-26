"""Lever L1 (r4t DECISIONS D18), first piece: Muon's Newton-Schulz on device, graded per matrix against float64.

Same op order and coefficients as upstream (RFD4-Proteina utils/grouped_muon.py::_orthogonalize, copied in
tt_bio/train/optim.py Muon.orthogonalize on wk/r4t-objective): flip to wide, divide by the Frobenius norm clamped at
eps, then 5 x [gram = X X^T; X = a X + (b gram + c gram gram) X]. Matrices of one shape run as one batched matmul.

Bar B1 (pre-registered, state/r4t/rows/perf.md): e_dev = ||O_dev - O_f64|| / ||O_f64|| <= 1.5 x e_host (upstream's own
bf16 host error against the same float64 reference), max over all 297 RFD4 denoiser matrices.
`--broken` scales one matrix's device result by 1.05, which must fail B1 (the gate shown able to fail).

Arms: `gram` (default, one bf16 rounding of the gram used in both terms; see ns_dev) and `naive` (first attempt,
2026-11-26 04:06Z: B1 FAIL at 2.1-5x host error, root cause in ns_dev's docstring). Timing: whole optimizer-step NS (all 297 matrices) synced, warm,
--reps times, AICLK sampled during from sysfs.
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time

A, B, C, EPS = 3.4445, -4.775, 2.0315, 1e-7


def ns_ref(x, dtype):
    """Upstream's sequence at `dtype` (torch.bfloat16 = upstream host, torch.float64 = reference). x is (k, m, n), m <= n."""
    import torch
    x = x.to(dtype)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=EPS)
    for _ in range(5):
        g = torch.bmm(x, x.transpose(-1, -2))
        x = torch.baddbmm(x, torch.baddbmm(g, g, g, beta=B, alpha=C), x, beta=A)
    return x.double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--arm", choices=["gram", "naive"], default="gram")
    ap.add_argument("--fidelity", default="HiFi4")
    ap.add_argument("--broken", action="store_true")
    ap.add_argument("--no-l1acc", action="store_true", help="packer_l1_acc off (partials stay in DEST)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    from tt_bio import tenstorrent as T

    shapes = json.loads(open(a.shapes).read().splitlines()[2])
    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    clk, stop = [], threading.Event()

    def sampler():
        p = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
        while not stop.is_set():
            try:
                clk.append((time.time(), int(open(p).read().split()[0])))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.02)
    threading.Thread(target=sampler, daemon=True).start()

    ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, a.fidelity),
                                                  math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=not a.no_l1acc)
    f32, b16 = ttnn.float32, ttnn.bfloat16
    mm = lambda p, q, dt, **k: ttnn.matmul(p, q, dtype=dt, compute_kernel_config=ckc, **k)

    def ns_dev(x):
        """x: device (k, m, n) bf16, m <= n. Returns bf16 device tensor.

        `gram` arm (default): the gram is rounded to bf16 ONCE and that same rounded gram feeds both polynomial
        terms (b g + c g g), as upstream's bf16 baddbmm does; p is formed in fp32 and rounded once for the matmul,
        a x + p x is summed in fp32 and rounded once. `naive` arm: the first device order (b g from the fp32 gram,
        c g g from the rounded one, a x rounded before the add). CPU emulation over 8 random 256x768 matrices
        (ns_emu.py): naive 2.1-2.3x host error, gram 1.01x. Mixing a rounded and an unrounded gram breaks the
        polynomial-in-one-matrix structure, and five Newton-Schulz iterations amplify it."""
        x32 = ttnn.typecast(x, f32)
        ss = ttnn.sum(ttnn.sum(ttnn.multiply(x32, x32), dim=-1, keepdim=True), dim=-2, keepdim=True)
        inv = ttnn.reciprocal(ttnn.clamp(ttnn.sqrt(ss), min=EPS))
        x = ttnn.multiply(x32, inv, dtype=b16)
        for _ in range(5):
            if a.arm == "naive":
                g = mm(x, x, f32, transpose_b=True)
                gb = ttnn.typecast(g, b16)
                p = ttnn.add(ttnn.multiply(g, B), ttnn.multiply(mm(gb, gb, f32), C), dtype=f32)
                x = ttnn.add(ttnn.multiply(x, A), mm(ttnn.typecast(p, b16), x, f32), dtype=b16)
            else:
                # The device sits a median 3.4 % above exact-bf16 emulation (b1p_a_dist.py). An extra error at the gram
                # reproduces it (b1p_trunc.py, median 0.994), but neither an fp32 gram + RNE typecast (1.0339, 9f32ebf49)
                # nor packer_l1_acc off (1.2175, worse) moves it: the gap is in the matmul's own accumulation.
                g = mm(x, x, b16, transpose_b=True)
                p = ttnn.add(ttnn.multiply(ttnn.typecast(g, f32), B), ttnn.multiply(mm(g, g, f32), C), dtype=b16)
                x = ttnn.add(ttnn.multiply(ttnn.typecast(x, f32), A), mm(p, x, f32), dtype=b16)
        return x

    gen = torch.Generator().manual_seed(0)
    groups = []
    for (r, c), cnt in shapes:
        w = torch.randn(cnt, r, c, generator=gen)
        if r > c:
            w = w.transpose(-1, -2).contiguous()
        groups.append(w)
    dins = [ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for w in groups]

    def step():
        return [ns_dev(x) for x in dins]
    outs = step(); ttnn.synchronize_device(dev)
    times = []
    for _ in range(a.reps):
        t0 = time.time(); o = step(); ttnn.synchronize_device(dev); t1 = time.time()
        times.append((t0, t1)); del o
    stop.set()
    ck = [c for t, c in clk if any(t0 <= t <= t1 for t0, t1 in times)]
    worst, fail, per = 0.0, [], []
    h = hashlib.sha256()
    for gi, (w, o) in enumerate(zip(groups, outs)):
        od = ttnn.to_torch(o).double()
        if a.broken and gi == 0:
            od[0] *= 1.05
        h.update(ttnn.to_torch(o).float().numpy().tobytes())
        ref = ns_ref(w, torch.float64); hb = ns_ref(w, torch.bfloat16)
        for k in range(w.shape[0]):
            n = ref[k].norm()
            ed, eh = float((od[k] - ref[k]).norm() / n), float((hb[k] - ref[k]).norm() / n)
            ok = ed <= 1.5 * eh
            worst = max(worst, ed / eh)
            per.append([list(w.shape[1:]), k, ed, eh])
            if not ok:
                fail.append([list(w.shape[1:]), k, ed, eh])
    dts = [t1 - t0 for t0, t1 in times]
    res = dict(arm=a.arm, fidelity=a.fidelity, l1acc=not a.no_l1acc, broken=a.broken, host="qb2", chip=card, board="p300c", sha=sha,
               n_matrices=sum(g.shape[0] for g in groups), reps=a.reps, step_s=[round(d, 4) for d in dts],
               step_s_min=round(min(dts), 4), aiclk=[min(ck), max(ck), len(ck)] if ck else None,
               worst_edev_over_ehost=round(worst, 3), n_fail=len(fail), fail=fail[:20],
               B1="PASS" if not fail else "FAIL", digest=h.hexdigest()[:16], per=per)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("fail", "per")}))


if __name__ == "__main__":
    main()
