#!/usr/bin/env python3
"""C4 float64 grade at S=2: `AF2MaskedOuterProductMean._sum_rows` on the tape, real OPM weights,
leaves a [S,I,C] and b [S,J,D], one cotangent; out, da, db of each arm against torch float64
autograd of `sum_s a_sic b_sjd W_cdk + bias` on the same bf16 operands. Arms: shipped (rows summed
after S products) and rows_in_k (rows joined along the contraction)."""
import json, pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def rel(x, r):
    return float((x - r).norm() / r.norm())


def main():
    out_path = sys.argv[1]
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import bindcraft2, af2
    tr = bindcraft2._Trunk(pathlib.Path("/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"))
    ag, T = tr.ag, tr.taped
    opm = tr.model.device_evoformer[0].opm
    n = 288
    C = int(opm.a_weight.shape[-1]); D = int(opm.b_weight.shape[-1])
    torch.manual_seed(0)
    a0, b0 = torch.randn(S, n, C), torch.randn(S, n, D)
    rd = lambda t: t.to(torch.bfloat16).double()
    A64 = rd(a0).requires_grad_(); B64 = rd(b0).requires_grad_()
    W = ttnn.to_torch(opm.o_weight).double().reshape(C, D, -1)
    bias = ttnn.to_torch(opm.o_bias).double().reshape(-1)
    ref = torch.einsum("sic,sjd,cdk->ijk", A64, B64, W) + bias
    g0 = torch.randn(ref.shape) * 1e-2
    ref.backward(rd(g0))
    res = {"S": S, "C": C, "D": D}
    with bindcraft2.fast_round():
        for arm, v in (("shipped", False), ("rows_in_k", True)):
            af2.AF2MaskedOuterProductMean.rows_in_k = v
            mk = lambda t: ag.Tensor(ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=tr.device, dtype=ttnn.bfloat16), requires_grad=True)
            al, bl = mk(a0), mk(b0)
            with T.tape():
                y = opm._sum_rows(al, bl)
            tr.sync()
            yv = ttnn.to_torch(y.value if hasattr(y, "value") else y).double().reshape(ref.shape)
            ag.backward([y], [tr.seed(g0.reshape([int(d) for d in (y.value if hasattr(y, "value") else y).shape]), y)])
            tr.sync()
            da = ttnn.to_torch(al.grad).double().reshape(A64.shape)
            db = ttnn.to_torch(bl.grad).double().reshape(B64.shape)
            res[arm] = {"out": rel(yv, ref.detach()), "da": rel(da, A64.grad), "db": rel(db, B64.grad),
                        "da_cos": float(torch.nn.functional.cosine_similarity(da.flatten(), A64.grad.flatten(), 0)),
                        "db_cos": float(torch.nn.functional.cosine_similarity(db.flatten(), B64.grad.flatten(), 0))}
            ag.release_pins()
            print(arm, json.dumps(res[arm]), flush=True)
    pathlib.Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
