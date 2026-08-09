#!/usr/bin/env python3
"""E3 -- the confidence-head pLDDT einsum (protenix.py:1465) as a mechanism ladder.

Every rung computes (or deliberately mis-computes) the same reduction

    out[n, b] = sum_c aln[n, c] * pw[t[n], c, b]        n < N_ATOM, c < 384, b < 50

at the production 298 aa shape, and each rung removes exactly ONE mechanism so the cost split is a
measurement rather than an argument:

  cur_auto    the shipping path verbatim: embedding gather -> reshape -> to_layout -> batched
              matmul with no config hint. B = N_atom, Mt = 1, Nt = 2.
  cur_grid    identical tensors, core_grid=CORE_GRID_MAIN on the matmul only.
              Difference vs cur_auto = the CORE-OCCUPANCY defect.
  cur_pcfg    identical tensors, explicit MatmulMultiCoreReuseProgramConfig.
              Difference vs cur_grid = whatever in0_block_w is worth on top of the grid.
  ctrl_b1     same M-padded batched in0, but in1 has batch 1 (one shared weight matrix).
              Difference vs cur_grid = the cost of MATERIALISING per-atom weights.
              Numerically wrong on purpose (every atom uses type 0). A control, not a candidate.
  ctrl_dense  (N_atom,384) @ (384,50), no batch dimension at all.
              Difference vs ctrl_b1 = the 31/32 M-PADDING waste.
              Also numerically wrong on purpose, same reason.
  reform      the real reformulation: one dense (N_atom,384) @ (384, 24*64) against ALL 24 atom
              types at once, a one-hot mask, and a block-sum matmul. No gather, no M padding.
              Numerically correct.
  reform_l1   reform, chunked over atoms with the (N_atom,1536) intermediate pinned in L1.
              Only worth running if reform lands far off its own roofline floor.

Timing: every region synchronises immediately before the clock starts and again before it stops,
and each rung is warmed before it is timed (PLAYBOOKS ACCELERATE rules 1 and 2).

Parity: the reference is torch's einsum in fp32 over the SAME bf16-rounded operands the device
sees, so a difference is device arithmetic and not input rounding. cur_* and reform are compared
to it and to each other with torch.equal.
"""
import argparse, json, statistics as st, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

# examples/prot300.yaml -- 298 aa CDK2, 298 tokens, 2398 atoms (FINDINGS.md, W10).
N_ATOM, C, NB, N_TA = 2398, 384, 50, 24
NBP = 64                      # 50 bins padded up to one tile face
TILE = 32
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

up = lambda x, m=TILE: ((x + m - 1) // m) * m


def timed(dev, fn, warm=3, pipe=4, reps=7):
    """Median seconds per call. Sync before the clock starts and before it stops."""
    for _ in range(warm):
        r = fn()
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            r = fn()
            if isinstance(r, ttnn.Tensor):
                ttnn.deallocate(r)
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roofs", help="roofs json from perf/ledger_298/roofs_card.py")
    ap.add_argument("--out", default="perf/conf_plddt/ladder.json")
    ap.add_argument("--a2ta", help=".pt with the real atom_to_tokatom_idx vector; synthetic if absent")
    ap.add_argument("--n-atom", type=int, default=N_ATOM)
    ap.add_argument("--reform-l1", action="store_true", help="also run the L1-chunked rung")
    ap.add_argument("--chunk", type=int, default=768)
    a = ap.parse_args()
    n = a.n_atom

    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    roofs = json.load(open(a.roofs)) if a.roofs else {}
    rd = roofs.get("dram_roofs", {}).get("read_peak_GBs")
    wr = roofs.get("dram_roofs", {}).get("write_peak_GBs")
    cp = roofs.get("compute_roof", {}).get("peak_TFLOPs")
    print(f"grid {dg.x}x{dg.y}  core_grid_main {CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}  "
          f"ttnn {ttnn.__version__}  roofs read={rd} write={wr} GB/s compute={cp} TFLOP/s", flush=True)

    # ---- host operands. bf16-round FIRST so the reference sees exactly the device's inputs ----
    torch.manual_seed(0)
    aln_h = torch.randn(n, C).bfloat16().float()
    pw_h = (torch.randn(N_TA, C, NB) * 0.05).bfloat16().float()
    if a.a2ta:
        t_h = torch.load(a.a2ta).long().reshape(-1)[:n]
        assert t_h.numel() == n, f"a2ta has {t_h.numel()} entries, need {n}"
    else:
        # Backbone atom indices dominate a real protein; a uniform draw would flatter the gather's
        # locality-insensitive rungs equally, but skew it to stay honest about DRAM page reuse.
        t_h = torch.clamp((torch.randn(n).abs() * 5).long(), 0, N_TA - 1)
    ref = torch.einsum("nc,ncb->nb", aln_h, pw_h[t_h])                    # fp32, exact for these inputs

    T = lambda x, layout=TL, dt=ttnn.bfloat16, mc=DRAM: ttnn.from_torch(
        x, layout=layout, device=dev, dtype=dt, memory_config=mc)

    # ---- resident operands, built once (they are sample-invariant in the model too) ----
    aln = T(aln_h)                                                        # (n, 384)
    pw_dev = T(pw_h.reshape(N_TA, C * NB).contiguous(), RM)               # (24, 19200) gather table
    a2ta_dev = T(t_h.reshape(-1, 1).to(torch.int32), RM, ttnn.uint32)     # (n, 1)
    pw_one = T(pw_h[0].reshape(1, C, NB))                                 # (1,384,50) control weight
    pw_2d = T(pw_h[0])                                                    # (384,50)  control weight

    pwf_h = torch.zeros(C, N_TA * NBP)
    for t in range(N_TA):
        pwf_h[:, t * NBP:t * NBP + NB] = pw_h[t]
    pw_flat = T(pwf_h)                                                    # (384, 1536)
    mask_h = torch.zeros(n, N_TA * NBP)
    mask_h[torch.arange(n).unsqueeze(1), (t_h.unsqueeze(1) * NBP + torch.arange(NBP).unsqueeze(0))] = 1.0
    mask = T(mask_h)                                                      # (n, 1536)
    s_h = torch.zeros(N_TA * NBP, NBP)
    for t in range(N_TA):
        s_h[t * NBP:(t + 1) * NBP, :] = torch.eye(NBP)
    s_sel = T(s_h)                                                        # (1536, 64) block-sum

    res, rows = {}, []
    B2 = 2  # bf16 bytes

    def rec(name, secs, rbytes, wbytes, flops, note=""):
        row = {"arm": name, "ms": round(secs * 1e3, 4),
               "read_MB": round(rbytes / 1e6, 2), "write_MB": round(wbytes / 1e6, 2),
               "GFLOP_padded": round(flops / 1e9, 3),
               "AI_FLOP_per_byte": round(flops / (rbytes + wbytes), 2) if rbytes + wbytes else None,
               "note": note}
        if rd:
            row["pct_read_roof"] = round(100 * (rbytes / secs / 1e9) / rd, 1)
        if wr:
            row["pct_write_roof"] = round(100 * (wbytes / secs / 1e9) / wr, 1)
        if cp:
            row["pct_compute_roof"] = round(100 * (flops / secs / 1e12) / cp, 1)
        rows.append(row)
        print("  " + json.dumps(row), flush=True)
        return row

    # ---------- the shipping path, op by op ----------
    print("=== cur_* : the shipping path (protenix.py:1450-1465) ===", flush=True)
    f_emb = lambda: ttnn.embedding(a2ta_dev, pw_dev, layout=RM, memory_config=DRAM)
    t_emb = timed(dev, f_emb)
    rec("cur.embedding", t_emb, N_TA * C * NB * B2, n * C * NB * B2, 0, "gather 24-row table -> per-atom")

    pw_g_rm = ttnn.reshape(f_emb(), (n, C, NB))
    t_lay = timed(dev, lambda: ttnn.to_layout(pw_g_rm, TL))
    rec("cur.to_layout(in1)", t_lay, n * C * NB * B2, n * C * up(NB) * B2, 0, "RM->TILE, 50 -> 64 pad")
    pw_g = ttnn.to_layout(pw_g_rm, TL)
    ttnn.deallocate(pw_g_rm)

    t_rs = timed(dev, lambda: ttnn.reshape(aln, (n, 1, C)))
    rec("cur.reshape(in0)", t_rs, n * C * B2, n * TILE * C * B2, 0, "(n,384)->(n,1,384): 1 row per tile")
    aln_b = ttnn.reshape(aln, (n, 1, C))

    mm_r = n * TILE * C * B2 + n * C * up(NB) * B2
    mm_w = n * TILE * up(NB) * B2
    mm_f = 2 * n * TILE * C * up(NB)
    for lbl, kw in (("cur.matmul auto", {}),
                    ("cur.matmul core_grid", {"core_grid": CORE_GRID_MAIN}),
                    ("cur.matmul pcfg", {"program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(CORE_GRID_MAIN.x, CORE_GRID_MAIN.y),
                        in0_block_w=1, out_subblock_h=1, out_subblock_w=2,
                        per_core_M=1, per_core_N=up(NB) // TILE)})):
        try:
            t = timed(dev, lambda kw=kw: ttnn.matmul(aln_b, pw_g, compute_kernel_config=ckc, **kw))
            rec(lbl, t, mm_r, mm_w, mm_f, "B=n, Mt=1, Nt=2, both operands DRAM-interleaved")
            res[lbl] = ttnn.to_torch(ttnn.matmul(aln_b, pw_g, compute_kernel_config=ckc, **kw)).float().reshape(n, -1)[:, :NB]
        except Exception as e:                       # a rejected program config is TT_FATAL, not this
            print(f"  {lbl}: EXC {str(e)[:160]}", flush=True)

    # ---------- controls: price the gather and the M padding separately ----------
    print("=== ctrl_* : controls, numerically wrong on purpose ===", flush=True)
    c1_r = n * TILE * C * B2 + C * up(NB) * B2
    t = timed(dev, lambda: ttnn.matmul(aln_b, pw_one, compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN))
    rec("ctrl_b1 (in1 batch 1)", t, c1_r, mm_w, mm_f, "isolates the per-atom weight materialisation")

    cd_r = n * C * B2 + C * up(NB) * B2
    cd_w = n * up(NB) * B2
    cd_f = 2 * up(n) * C * up(NB)
    t = timed(dev, lambda: ttnn.matmul(aln, pw_2d, compute_kernel_config=ckc))
    rec("ctrl_dense (no batch)", t, cd_r, cd_w, cd_f, "isolates the 31/32 M-padding waste")

    # ---------- the reformulation ----------
    print("=== reform : dense against all 24 types, one-hot select ===", flush=True)
    W = N_TA * NBP
    r1_r, r1_w, r1_f = n * C * B2 + C * W * B2, n * W * B2, 2 * up(n) * C * W
    t1 = timed(dev, lambda: ttnn.matmul(aln, pw_flat, compute_kernel_config=ckc))
    rec("reform.matmul_all", t1, r1_r, r1_w, r1_f, "(n,384)@(384,1536), dense, full grid")
    y = ttnn.matmul(aln, pw_flat, compute_kernel_config=ckc)
    t2 = timed(dev, lambda: ttnn.multiply(y, mask))
    rec("reform.mask", t2, 2 * n * W * B2, n * W * B2, 0, "one-hot select of the atom's 64-wide block")
    ym = ttnn.multiply(y, mask)
    r3_r, r3_w, r3_f = n * W * B2 + W * NBP * B2, n * NBP * B2, 2 * up(n) * W * NBP
    t3 = timed(dev, lambda: ttnn.matmul(ym, s_sel, compute_kernel_config=ckc))
    rec("reform.blocksum", t3, r3_r, r3_w, r3_f, "sum the 24 blocks; 23 addends are exactly 0")
    rec("reform TOTAL", t1 + t2 + t3, r1_r + 2 * n * W * B2 + r3_r, r1_w + n * W * B2 + r3_w,
        r1_f + r3_f, "sum of the three rungs")
    res["reform"] = ttnn.to_torch(ttnn.matmul(ym, s_sel, compute_kernel_config=ckc)).float().reshape(n, -1)[:, :NB]

    if a.reform_l1:
        print("=== reform_l1 : chunked over atoms, intermediate pinned in L1 ===", flush=True)
        ch = a.chunk

        def run_l1():
            outs = []
            for i in range(0, n, ch):
                j = min(i + ch, n)
                ai = ttnn.slice(aln, [i, 0], [j, C])
                mi = ttnn.slice(mask, [i, 0], [j, W])
                yi = ttnn.matmul(ai, pw_flat, compute_kernel_config=ckc, memory_config=L1)
                zi = ttnn.multiply(yi, mi, memory_config=L1)
                outs.append(ttnn.matmul(zi, s_sel, compute_kernel_config=ckc, memory_config=DRAM))
                for tsr in (ai, mi, yi, zi):
                    ttnn.deallocate(tsr)
            o = ttnn.concat(outs, dim=0)
            for tsr in outs:
                ttnn.deallocate(tsr)
            return o
        t = timed(dev, run_l1, warm=2, pipe=2, reps=5)
        nch = (n + ch - 1) // ch
        rec("reform_l1 TOTAL", t, n * C * B2 + nch * C * W * B2 + n * W * B2, n * NBP * B2,
            r1_f + r3_f, f"chunk={ch} ({nch} chunks), y never reaches DRAM")
        res["reform_l1"] = ttnn.to_torch(run_l1()).float().reshape(n, -1)[:, :NB]

    # ---------- parity ----------
    print("=== parity vs fp32 torch over the same bf16 operands ===", flush=True)
    par = {}
    keys = [k for k in res]
    base = res.get("cur.matmul auto")
    for k in keys:
        v = res[k]
        d = (v - ref).abs()
        par[k] = {"max_abs_err": float(d.max()), "rel_max": float((d / ref.abs().clamp(min=1e-6)).max()),
                  "pcc": float(torch.corrcoef(torch.stack([v.reshape(-1), ref.reshape(-1)]))[0, 1]),
                  "equal_to_cur_auto": bool(base is not None and torch.equal(v, base))}
        print(f"  {k:26s} {json.dumps(par[k])}", flush=True)

    json.dump({"shape": {"n_atom": n, "c": C, "nb": NB, "nb_padded": NBP, "n_ta": N_TA},
               "grid": f"{dg.x}x{dg.y}", "ttnn": ttnn.__version__,
               "roofs": {"read_GBs": rd, "write_GBs": wr, "compute_TFLOPs": cp},
               "rows": rows, "parity": par}, open(a.out, "w"), indent=2)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
