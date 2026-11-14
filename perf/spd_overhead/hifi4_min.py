"""WH HiFi4 + fp32-acc matmul returns -4.0 for a dot product whose value is 0.00039. Plain ttnn, no tt-bio.

    TT_VISIBLE_DEVICES=N python perf/spd_overhead/hifi4_min.py AB.pt [--out OUT.json]

AB.pt holds a = [64, 256] and b = [256, 64] bf16 (hifi4_repro.py --ab writes it: the pair Transition's
layer-norm rows and fc2 weight columns around the bad element, row 45 col 38). The script runs
ttnn.linear(a, b, core_grid=CoreGrid(y, x)) at HiFi4 and HiFi3 with fp32_dest_acc_en, prints the
element next to float64, then tries the variants that say what the trigger is: other rows or
columns zeroed, the row moved into the first M tile, other core grids, ttnn.matmul.
"""
import argparse, json
import torch, ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ab")
    ap.add_argument("--out")
    ap.add_argument("--grid", default="8x8", help="core grid YxX the failing call used")
    a = ap.parse_args()
    d = torch.load(a.ab)
    A, B, (i, j) = d["a"].double(), d["b"].double(), d["ij"]
    dev = ttnn.open_device(device_id=0)
    gy, gx = map(int, a.grid.split("x"))

    def run(A, B, fid="HiFi4", grid=(gy, gx), op="linear", f32=True):
        k = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid),
                                             math_approx_mode=True, fp32_dest_acc_en=f32, packer_l1_acc=True)
        At = ttnn.from_torch(A.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        Bt = ttnn.from_torch(B.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        kw = dict(compute_kernel_config=k, dtype=ttnn.float32)
        if grid:
            kw["core_grid"] = ttnn.CoreGrid(y=grid[0], x=grid[1])
        y = (ttnn.linear if op == "linear" else ttnn.matmul)(At, Bt, **kw)
        return ttnn.to_torch(y).double()

    res = {"f64": float(A[i] @ B[:, j]), "ij": [i, j], "grid": a.grid}
    rows = lambda keep: A * torch.tensor([float(r in keep) for r in range(A.shape[0])], dtype=A.dtype)[:, None]
    cols = lambda keep: B * torch.tensor([float(c in keep) for c in range(B.shape[1])], dtype=B.dtype)[None]
    sw = torch.arange(A.shape[0]); sw[i], sw[i % 32] = i % 32, i
    var = {"as_is": (A, B, i, j, {}),
           "hifi3": (A, B, i, j, {"fid": "HiFi3"}),
           "hifi4_no_fp32acc": (A, B, i, j, {"f32": False}),
           "no_core_grid": (A, B, i, j, {"grid": None}),
           "matmul": (A, B, i, j, {"op": "matmul"}),
           "only_row_i": (rows({i}), B, i, j, {}),
           "only_col_j": (A, cols({j}), i, j, {}),
           "only_row_i_col_j": (rows({i}), cols({j}), i, j, {}),
           "row_swapped_into_tile0": (A[sw], B, i % 32, j, {})}
    for g in ((1, 1), (2, 2), (1, 2), (2, 1), (4, 4), (8, 1), (1, 8)):
        var[f"grid_{g[0]}x{g[1]}"] = (A, B, i, j, {"grid": g})
    for name, (Av, Bv, ii, jj, kw) in var.items():
        y = run(Av, Bv, **kw)
        want = Av @ Bv
        res[name] = {"at": float(y[ii, jj]), "want": float(want[ii, jj]),
                     "n_over_0.5": int(((y - want).abs() > 0.5).sum())}
        print(name, json.dumps(res[name]), flush=True)
    ttnn.close_device(dev)
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
