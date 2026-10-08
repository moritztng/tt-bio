"""Shrink the WH HiFi4 + fp32-acc matmul defect to the smallest case that still shows it.

    TT_VISIBLE_DEVICES=N python perf/spd_overhead/hifi4_repro.py --out OUT.pt

transition_outliers.py found fc2 of the pair Transition (736x256x1024, seed 0) returning -4.0
where float64 says 0.00039, at pixel (190, 429) channel 806, under HiFi4 + fp32_dest_acc_en only.
This script takes the device's own layer-norm row for that pixel and the weight column, saves both
as bf16, and runs the dot product again as ever smaller matmuls: the full row block, one [32, 256]
x [256, 32] tile pair with the row and column at their original tile positions, the same tile pair
at position 0, and the same with a plain ttnn.matmul. Each case is run at HiFi4 and HiFi3, fp32
acc on, and prints the device value next to float64.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--px", default="190,429,806")
    ap.add_argument("--h", type=int, default=5, help="the Transition's row-block height on this chip")
    a = ap.parse_args()
    import torch, ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    dev = T.get_device()
    K = lambda fid: ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=True,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    S, C, HID = 736, 256, 1024
    r, c, ch = map(int, a.px.split(","))
    g = torch.Generator().manual_seed(0)
    bf = lambda t: t.to(torch.bfloat16).to(torch.float64)
    sd = {"norm.weight": bf(1 + 0.1 * torch.randn(C, generator=g)),
          "norm.bias": bf(0.1 * torch.randn(C, generator=g)),
          "fc1.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
          "fc2.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
          "fc3.weight": bf(torch.randn(C, HID, generator=g) / HID ** 0.5)}
    zt = bf(torch.randn(1, S, S, C, generator=g))
    tr = T.Transition({k: v.float() for k, v in sd.items()}, K("HiFi4"))
    h = a.h
    s = (r // h) * h
    z = ttnn.from_torch(zt[:, s:s + h].float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    xn = ttnn.layer_norm(z, weight=tr.norm_weight, bias=tr.norm_bias, epsilon=1e-5,
                         compute_kernel_config=K("HiFi4"), memory_config=ttnn.L1_MEMORY_CONFIG)
    xnd = ttnn.to_torch(xn).to(torch.float64)                 # [1, h, S, C], exact bf16 values
    x = xnd[0, r - s, c].clone()                              # [256]
    w = sd["fc2.weight"][ch].clone()                          # [256]
    res = {"px": [r, c, ch], "h": h, "f64": float(x @ w),
           "kpart_f64": [float(x[i:i + 32] @ w[i:i + 32]) for i in range(0, C, 32)]}
    dt = lambda t: ttnn.to_torch(t).to(torch.float64)

    def lin(xin, wt, fid, grid=True):
        kw = dict(compute_kernel_config=K(fid), memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.float32)
        if grid:
            kw["core_grid"] = T.CORE_GRID_MAIN
        return dt(ttnn.linear(xin, wt, **kw))

    cases = {}
    for fid in ("HiFi4", "HiFi3"):
        y = lin(xn, tr.fc2_weight, fid)
        cases[f"block_{fid}"] = float(y[0, r - s, c, ch])
    # One tile pair. Row position inside its tile in the flattened [h*S, C] view, column inside its tile.
    flat = (r - s) * S + c
    tr_row, tc = flat % 32, ch % 32
    rows = xnd[0].reshape(-1, C)[flat - tr_row: flat - tr_row + 32]     # the 32 rows of the original tile
    cols = sd["fc2.weight"][ch - tc: ch - tc + 32].T                    # [256, 32]
    for tag, A, i, j in (("tile_orig", rows, tr_row, tc),
                         ("tile_row0", torch.cat([x[None], torch.zeros(31, C, dtype=torch.float64)]), 0, None),
                         ("tile_alone", None, 0, 0)):
        if tag == "tile_alone":
            A = torch.zeros(32, C, dtype=torch.float64); A[0] = x
            B = torch.zeros(C, 32, dtype=torch.float64); B[:, 0] = w
        else:
            B = cols
        jj = tc if j is None else j
        At = ttnn.from_torch(A.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        Bt = ttnn.from_torch(B.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for fid in ("HiFi4", "HiFi3"):
            cases[f"{tag}_linear_{fid}"] = float(lin(At, Bt, fid, grid=False)[i, jj])
            mm = dt(ttnn.matmul(At, Bt, compute_kernel_config=K(fid), dtype=ttnn.float32))
            cases[f"{tag}_matmul_{fid}"] = float(mm[i, jj])
        ref = A @ B
        cases[f"{tag}_f64"] = float(ref[i, jj])
    res["cases"] = cases
    # Bisect: the same bf16 values re-uploaded as a 2-D [h*S, C] tensor, cut to an M window holding
    # the row and an N window holding the channel, with and without the module's core grid.
    A2 = xnd[0].reshape(-1, C)
    W2 = sd["fc2.weight"].T                                    # [C, HID]
    bis = []
    for m in (A2.shape[0], 1024, 256, 64, 32):
        m0 = min((flat // m) * m, A2.shape[0] - m)
        for n in (HID, 256, 64, 32):
            n0 = (ch // n) * n
            A = A2[m0:m0 + m]; B = W2[:, n0:n0 + n]
            At = ttnn.from_torch(A.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            Bt = ttnn.from_torch(B.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            want = A @ B
            for grid in (True, False):
                y = lin(At, Bt, "HiFi4", grid=grid)
                d = (y - want).abs()
                bis.append({"m": m, "n": n, "grid": grid, "at_px": float(y[flat - m0, ch - n0]),
                            "n_over_0.5": int((d > 0.5).sum()), "max": float(d.max()),
                            "worst": [int(i) for i in divmod(int(d.argmax()), n)]})
                print(json.dumps(bis[-1]), flush=True)
            ttnn.deallocate(At); ttnn.deallocate(Bt)
    res["bisect"] = bis
    torch.save({"x": x.to(torch.bfloat16), "w": w.to(torch.bfloat16), "res": res}, a.out)
    res["x_hex"] = x.to(torch.bfloat16).view(torch.int16).numpy().astype("uint16").tobytes().hex()
    res["w_hex"] = w.to(torch.bfloat16).view(torch.int16).numpy().astype("uint16").tobytes().hex()
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_hex")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
