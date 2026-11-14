#!/usr/bin/env python3
"""bc4-bet-opm step 1: `AF2MaskedOuterProductMean._sum_rows_joined`'s op sequence, replayed in float64
torch with ttnn's logical reshape/permute semantics, against a direct einsum of AF2's masked OPM
numerator `sum_s a_sic b_sjd W_cdk + o_bias_k`. Values and both input gradients. CPU only.

`rows_joined_blocks` is taken from tt_bio/af2.py's source (that module imports ttnn), so the check
reads the weight layout the card builds, not a copy of it.

    python perf/bc4_opm/form_f64.py out.json
"""
import ast, json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
src = (ROOT / "tt_bio/af2.py").read_text()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "rows_joined_blocks")
ns = {"torch": torch}
exec(compile(ast.Module([fn], []), "af2.py", "exec"), ns)
rows_joined_blocks = ns["rows_joined_blocks"]


def form(a, b, o, bias, flat):
    """The ttnn sequence. `flat` is this bet's [1, I*c_z, S*D] view of the second operand A."""
    S, I, C = a.shape
    _, J, D = b.shape
    w = rows_joined_blocks(o, S, C, D)
    c_z = w.shape[0] // (S * D)
    at = a.permute(0, 2, 1).reshape(S * C, I)
    T = w @ at
    A = T.reshape(c_z, S * D, I).permute(2, 0, 1)
    bt = b.permute(0, 2, 1).reshape(S * D, J)
    if flat:
        out = (A.reshape(1, I * c_z, S * D) @ bt).reshape(I, c_z, J)
    else:
        out = A @ bt
    out = out.permute(0, 2, 1) + bias
    return out.reshape(1, *out.shape)


def ref(a, b, o, bias):
    S, I, C = a.shape
    D = b.shape[-1]
    W = o.reshape(C, D, -1)
    return (torch.einsum("sic,sjd,cdk->ijk", a, b, W) + bias)[None]


def rel(x, y):
    return float(((x - y).norm() / y.norm()).detach())


def case(S, I, J, C=32, D=32, c_z=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    a, b = r(S, I, C).requires_grad_(), r(S, J, D).requires_grad_()
    o, bias = r(C * D, c_z) * 0.05, r(c_z)
    cot = r(1, I, J, c_z)
    y0 = ref(a, b, o, bias)
    ga0, gb0 = torch.autograd.grad((y0 * cot).sum(), (a, b))
    res = {}
    for name, flat in (("main", False), ("opm", True)):
        y = form(a, b, o, bias, flat)
        ga, gb = torch.autograd.grad((y * cot).sum(), (a, b))
        res[name] = {"out_rel": rel(y, y0), "da_rel": rel(ga, ga0), "db_rel": rel(gb, gb0)}
    return {"S": S, "I": I, "J": J, **res}


if __name__ == "__main__":
    cases = [case(2, 224, 224), case(2, 192, 192, seed=1), case(3, 96, 160, seed=2), case(4, 64, 128, seed=3)]
    worst = max(v for c in cases for k in ("main", "opm") for v in c[k].values())
    out = {"cases": cases, "worst_rel": worst, "pass": worst < 1e-12}
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    sys.exit(0 if out["pass"] else 1)
