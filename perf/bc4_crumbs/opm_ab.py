#!/usr/bin/env python3
"""bc4-land-crumbs (from bc4-bet-layout): `AF2MaskedOuterProductMean._sum_rows` per call, the per-row form main shipped under
`rows_in_k` (inline copy below) against `_sum_rows_joined`, on the tape, real OPM weights, S=2.
Bit-equality of out, da, db between the arms, rel L2 of each against float64, and the median of
`--bursts` bursts of `--reps` taped forward+backward calls with one synchronize per burst.
    python perf/bc4_crumbs/opm_ab.py out.json [--ns 224,288,864]
"""
import argparse, json, pathlib, statistics, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def per_row(opm, a, b):
    """main 1764bf7a4's `_sum_rows` with `rows_in_k`: per-row matmul, row-major round trip, permute,
    then a concat per operand. `af2.ttnn` is the name the tape shims, so read it at call time."""
    from tt_bio import af2
    ttnn, CORE_GRID_MAIN = af2.ttnn, af2.CORE_GRID_MAIN
    S, I, C = (int(d) for d in a.shape)
    _, J, D = (int(d) for d in b.shape)
    w = opm._proj_o_folded(C, D)
    c_z = int(w.shape[1]) // D
    As, bts = [], []
    for s_i in range(S):
        a_s = ttnn.reshape(a[s_i:s_i + 1], (I, C))
        A = ttnn.matmul(a_s, w, compute_kernel_config=opm.compute_kernel_config, core_grid=CORE_GRID_MAIN)
        A = ttnn.to_layout(A, ttnn.ROW_MAJOR_LAYOUT)
        A = ttnn.reshape(A, (I, D, c_z))
        A = ttnn.to_layout(A, ttnn.TILE_LAYOUT)
        As.append(ttnn.permute(A, (0, 2, 1)))
        b_s = ttnn.reshape(b[s_i:s_i + 1], (J, D))
        bts.append(ttnn.permute(b_s, (1, 0)))
    out = ttnn.matmul(ttnn.concat(As, dim=-1), ttnn.concat(bts, dim=0),
                      compute_kernel_config=opm.compute_kernel_config)
    out = ttnn.permute(out, (0, 2, 1))
    out = ttnn.add(out, opm.o_bias)
    return ttnn.reshape(out, (1, *tuple(out.shape)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--ns", default="224,288,864")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--bursts", type=int, default=7)
    A = p.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import bindcraft2, af2
    tr = bindcraft2._Trunk(pathlib.Path("/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"))
    ag, T = tr.ag, tr.taped
    opm = tr.model.device_evoformer[0].opm
    C = int(opm.a_weight.shape[-1]); D = int(opm.b_weight.shape[-1]); S = 2
    card = int((__import__("os").environ.get("TT_VISIBLE_DEVICES") or "0").split(",")[0])
    aiclk = lambda: int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read())
    rel = lambda x, r: float((x - r).norm() / r.norm())
    res = {"S": S, "C": C, "D": D, "card": card, "ns": {}}
    with bindcraft2.fast_round():
        assert af2.AF2MaskedOuterProductMean.rows_in_k
        for n in (int(x) for x in A.ns.split(",")):
            torch.manual_seed(n)
            a0, b0 = torch.randn(S, n, C), torch.randn(S, n, D)
            rd = lambda t: t.to(torch.bfloat16).double()
            A64 = rd(a0).requires_grad_(); B64 = rd(b0).requires_grad_()
            W = ttnn.to_torch(opm.o_weight).double().reshape(C, D, -1)
            bias = ttnn.to_torch(opm.o_bias).double().reshape(-1)
            ref = torch.einsum("sic,sjd,cdk->ijk", A64, B64, W) + bias
            g0 = torch.randn(ref.shape) * 1e-2
            ref.backward(rd(g0))
            mk = lambda t: ag.Tensor(ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                                     device=tr.device, dtype=ttnn.bfloat16), requires_grad=True)
            gseed = None
            leaves = []

            def step(fn):
                nonlocal gseed
                al, bl = mk(a0), mk(b0)
                leaves[:] = [al, bl]
                with T.tape():
                    y = fn(opm, al, bl)
                yv = y.value if hasattr(y, "value") else y
                if gseed is None:
                    gseed = g0.reshape([int(d) for d in yv.shape])
                ag.backward([y], [tr.seed(gseed, y)])
                ag.release_pins()
                return yv

            arms = {"per_row": per_row, "joined": lambda o, x, y: o._sum_rows(x, y)}
            got, row = {}, {}
            for name, fn in arms.items():
                yv = step(fn); tr.sync()
                al, bl = leaves
                got[name] = [ttnn.to_torch(t).double().reshape(r.shape) for t, r in
                             ((yv, ref), (al.grad, A64.grad), (bl.grad, B64.grad))]
                row[name] = {"out": rel(got[name][0], ref.detach()), "da": rel(got[name][1], A64.grad),
                             "db": rel(got[name][2], B64.grad)}
            row["bit_equal"] = {arm: {k: bool(torch.equal(x, y)) for k, x, y in
                                      zip(("out", "da", "db"), got["per_row"], got[arm])}
                                for arm in arms if arm != "per_row"}
            # interleaved bursts, so a clock or load change lands on both arms
            ms = {k: [] for k in arms}
            clk = []
            for _ in range(A.bursts):
                for name, fn in arms.items():
                    tr.sync(); t0 = time.perf_counter()
                    for _ in range(A.reps):
                        step(fn)
                    tr.sync(); ms[name].append((time.perf_counter() - t0) * 1e3 / A.reps)
                    clk.append(aiclk())
            for name in arms:
                row[name]["ms"] = round(statistics.median(ms[name]), 4)
                row[name]["spread_ms"] = round(max(ms[name]) - min(ms[name]), 4)
            row["aiclk_min_max"] = [min(clk), max(clk)]
            res["ns"][n] = row
            print(n, json.dumps(row), flush=True)
    pathlib.Path(A.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
