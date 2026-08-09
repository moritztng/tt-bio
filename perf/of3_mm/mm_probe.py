"""OF3 batched-matmul site probe.

For each OF3 shape class (padded shapes, bf16, DRAM-interleaved, HiFi4 + fp32 dest acc),
time ttnn.matmul with:
  auto        - no config hint (what the model ships today)
  grid11      - core_grid=CoreGrid(y=10, x=11)   (tt_bio CORE_GRID_MAIN)
  grid13      - core_grid=CoreGrid(y=10, x=13)   (full device grid)
  reuse_*     - explicit MatmulMultiCoreReuseProgramConfig variants

and check torch.equal of every arm against auto.

Every timed region is bracketed by ttnn.synchronize_device on both sides.
"""
import argparse, json, time
import torch
import ttnn

GY, GX_DEV = 10, 13
GX_11 = 11


def ck(dev):
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def mk(dev, shape, seed):
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(shape, generator=g, dtype=torch.float32) - 0.5) * 0.2
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def subblock(m, n):
    # fp32_dest_acc_en -> out_subblock_h*out_subblock_w <= 4
    best = (1, 1)
    for h in range(1, m + 1):
        if m % h:
            continue
        for w in range(1, n + 1):
            if n % w or h * w > 4:
                continue
            if h * w > best[0] * best[1]:
                best = (h, w)
    return best


def reuse_cfg(Mt, Nt, Kt, per_core_M, per_core_N, in0_block_w, gx=GX_DEV, gy=GY):
    h, w = subblock(per_core_M, per_core_N)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=h,
        out_subblock_w=w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def time_arm(fn, iters):
    dev = ttnn.GetDefaultDevice()
    o = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / iters
    for i in range(1, len(outs)):
        ttnn.deallocate(outs[i])
    return dt * 1e3, outs[0]


def pcc(x, y):
    a = x.double().flatten()
    b = y.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm())
    return float((a @ b / d)) if float(d) > 0 else float("nan")


CASES = {}


def case(name, ashape, bshape, note):
    CASES[name] = dict(a=ashape, b=bshape, note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--nb", type=int, default=75, help="atom-transformer block count")
    ap.add_argument("--only", default=None)
    ap.add_argument("--ref", action="store_true", help="also score every arm against an fp32 host matmul")
    a = ap.parse_args()

    N = 320          # 298 aa padded to a tile multiple
    nb = a.nb

    # OF3 trunk triangle attention (tenstorrent.py:256/269 via _fp32_softmax_attention)
    # head_dim 32, 4 heads, batch = S * H
    case("triatt_qk", (N, 4, N, 32), (N, 4, 32, N), "trunk tri-att q@kT")
    case("triatt_av", (N, 4, N, N), (N, 4, N, 32), "trunk tri-att attn@v")
    # OF3 trunk token AttentionPairBias, head_dim 24 -> padded 32, 16 heads
    case("tokatt_qk", (1, 16, N, 32), (1, 16, 32, N), "trunk token apb q@kT")
    case("tokatt_av", (1, 16, N, N), (1, 16, N, 32), "trunk token apb attn@v")
    # OF3 DiT (openfold3_diffusion_transformer.py:184/193), 16 heads, head_dim 48->64
    case("dit_qk", (1, 16, N, 64), (1, 16, 64, N), "DiT q@kT")
    case("dit_av", (1, 16, N, N), (1, 16, N, 64), "DiT attn@v")
    # OF3 atom transformer (openfold3_atom_transformer.py:153/159), rank-5
    case("atom_qk", (1, nb, 4, 32, 32), (1, nb, 4, 32, 128), "atom-tf q@kT rank5")
    case("atom_av", (1, nb, 4, 32, 128), (1, nb, 4, 128, 32), "atom-tf attn@v rank5")

    dev = ttnn.open_device(device_id=0)
    ttnn.SetDefaultDevice(dev)
    cfg = ck(dev)
    results = []
    names = [a.only] if a.only else list(CASES)
    for name in names:
        c = CASES[name]
        ash, bsh = c["a"], c["b"]
        Mt, Kt, Nt = ash[-2] // 32, ash[-1] // 32, bsh[-1] // 32
        B = 1
        for d in ash[:-2]:
            B *= d
        ta, tb = mk(dev, ash, 1), mk(dev, bsh, 2)
        row = dict(case=name, note=c["note"], a=list(ash), b=list(bsh),
                   B=B, Mt=Mt, Kt=Kt, Nt=Nt, arms={})

        def run(tag, **kw):
            try:
                ms, out = time_arm(lambda: ttnn.matmul(ta, tb, compute_kernel_config=cfg, **kw),
                                   a.iters)
                row["arms"][tag] = dict(ms=ms)
                return out
            except Exception as e:
                row["arms"][tag] = dict(error=repr(e)[:400])
                return None

        base = run("auto")
        base_t = ttnn.to_torch(base) if base is not None else None
        if base is not None:
            ttnn.deallocate(base)
        ref = None
        if a.ref:
            at = ttnn.to_torch(ta).float()
            bt = ttnn.to_torch(tb).float()
            ref = torch.matmul(at, bt)
            del at, bt
            row["auto_pcc_vs_fp32"] = pcc(base_t, ref)

        arms = [("grid11", dict(core_grid=ttnn.CoreGrid(y=GY, x=GX_11))),
                ("grid13", dict(core_grid=ttnn.CoreGrid(y=GY, x=GX_DEV)))]
        # explicit reuse: per_core_M == Mt (safe by G1's predicate)
        arms.append(("reuse_Mt_Nt_k1",
                     dict(program_config=reuse_cfg(Mt, Nt, Kt, Mt, Nt, 1))))
        if Kt > 1:
            arms.append(("reuse_Mt_Nt_kall",
                         dict(program_config=reuse_cfg(Mt, Nt, Kt, Mt, Nt, Kt))))
        # split M: only legal when total blocks <= grid cores
        for pm in sorted({d for d in range(1, Mt) if Mt % d == 0}, reverse=True):
            blocks = (B * Mt // pm) * (Nt // Nt)
            if blocks <= GX_DEV * GY:
                arms.append((f"reuse_M{pm}_Nt_k1",
                             dict(program_config=reuse_cfg(Mt, Nt, Kt, pm, Nt, 1))))
                break
        for tag, kw in arms:
            out = run(tag, **kw)
            if out is not None:
                if base_t is not None:
                    ot = ttnn.to_torch(out)
                    row["arms"][tag]["bit_exact"] = bool(torch.equal(ot, base_t))
                    row["arms"][tag]["pcc_vs_auto"] = pcc(ot, base_t)
                    d = (ot.double() - base_t.double()).abs()
                    row["arms"][tag]["maxabs_vs_auto"] = float(d.max())
                    row["arms"][tag]["rel_l2_vs_auto"] = float(d.norm() / base_t.double().norm())
                    row["out_absmax"] = float(base_t.double().abs().max())
                    if ref is not None:
                        row["arms"][tag]["pcc_vs_fp32"] = pcc(ot, ref)
                    del ot
                ttnn.deallocate(out)
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
        results.append(row)
        print(json.dumps(row), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
