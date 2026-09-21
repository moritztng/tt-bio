"""The two attentions under one `PairformerLayer` want opposite pair-bias conventions.

`AttentionPairBias` adds the bias inside its own score scale, `(q@k^T + z) * head_dim**-0.5`, so
a reference that adds `z` unscaled to a pre-scaled `q` needs `z` pre-baked by sqrt(d):
`scale_pair_bias=True`. `TriangleAttention` scales `q@k^T` alone and divides the bake back out
before adding, so the same reference convention wants False. One shared flag served both, and the
OpenFold3 trunk took the triangle value, which left its token pair bias at 1/sqrt(24) = 0.204 of
the reference value in all 48 blocks of every fold. Measured against a float64 transcription of
AF3 Alg 24: relative L2 5.7577e-02 -> 1.7795e-02, PCC 0.998522 -> 0.999977
(`perf/of3t_pairbias/attn_f64.json`).

Three properties to hold. The split exists and is wired the way round it is supposed to be. The
default is transparent, so a caller that does not name `tri_att_scale_pair_bias` builds exactly
the sub-modules it built before. And the OpenFold3 trunk is the one site that names it, with the
value the float64 reference picked -- a silent revert there is a silent 0.204x, which is the
failure this file exists to catch.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TS = ROOT / "tt_bio" / "tenstorrent.py"
TRUNK = ROOT / "tt_bio" / "openfold3_trunk.py"

TREE = ast.parse(TS.read_text())


def _class(tree, name):
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == name)


def _init(tree, name):
    return next(n for n in ast.walk(_class(tree, name))
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")


def _calls(fn, callee):
    return [n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == callee]


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _defaults(fn):
    args = fn.args.args[len(fn.args.args) - len(fn.args.defaults):]
    return {a.arg: d for a, d in zip(args, fn.args.defaults)}


def test_both_constructors_take_the_argument_and_default_it_to_none():
    for cls in ("PairformerLayer", "Pairformer"):
        d = _defaults(_init(TREE, cls))
        assert "tri_att_scale_pair_bias" in d, cls
        assert d["tri_att_scale_pair_bias"].value is None, cls


def test_pairformer_forwards_it_to_every_layer():
    call = _calls(_init(TREE, "Pairformer"), "PairformerLayer")
    assert len(call) == 1
    fwd = _kw(call[0], "tri_att_scale_pair_bias")
    assert isinstance(fwd, ast.Name) and fwd.id == "tri_att_scale_pair_bias"


def test_the_layer_sends_tri_scale_to_the_triangles_and_scale_pair_bias_to_the_token_site():
    fn = _init(TREE, "PairformerLayer")
    tri = _calls(fn, "TriangleAttention")
    assert len(tri) == 2, "both the starting and the ending node must take the split value"
    for c in tri:
        v = _kw(c, "scale_pair_bias")
        assert isinstance(v, ast.Name) and v.id == "tri_scale", ast.dump(v)
    apb = _calls(fn, "AttentionPairBias")
    assert len(apb) == 1
    v = _kw(apb[0], "scale_pair_bias")
    assert isinstance(v, ast.Name) and v.id == "scale_pair_bias", ast.dump(v)


def test_the_default_is_transparent():
    """`None` has to mean "whatever scale_pair_bias says", or every other model moves."""
    fn = _init(TREE, "PairformerLayer")
    assign = next(n for n in ast.walk(fn)
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "tri_scale")
    e = assign.value
    assert isinstance(e, ast.IfExp)
    assert isinstance(e.test, ast.Compare) and isinstance(e.test.ops[0], ast.Is)
    assert e.test.left.id == "tri_att_scale_pair_bias"
    assert e.test.comparators[0].value is None
    assert e.body.id == "scale_pair_bias"          # None -> follow the shared flag
    assert e.orelse.id == "tri_att_scale_pair_bias"


def test_the_openfold3_trunk_is_the_one_site_that_names_it_and_names_it_right():
    call = next(n for n in ast.walk(ast.parse(TRUNK.read_text()))
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Pairformer")
    assert _kw(call, "scale_pair_bias").value is True
    assert _kw(call, "tri_att_scale_pair_bias").value is False

    named = [p for p in sorted(ROOT.glob("tt_bio/**/*.py"))
             if p != TRUNK and "tri_att_scale_pair_bias=" in p.read_text()
             and p != TS]
    assert named == [], f"another site now names the split: {named}"
