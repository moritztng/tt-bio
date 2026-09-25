#!/usr/bin/env python3
"""Per-op VJP check for J4's nine substitutions, against a float64 reference.

A46 clause 1: a backward is graded on its vector-Jacobian product, never on its forward.
So for each of the nine ops this

  1. builds a float64 host reference VJP in numpy,
  2. VALIDATES that reference with central finite differences -- <g, J v> computed two ways,
     once through the reference Jacobian-vector contraction and once through
     (f(x + h v) - f(x - h v)) / 2h, in float64, for random probe directions v,
  3. runs the COMPOSED arm (the expression `tt_bio/autograd.py` shipped before this row) and
     the WHEEL arm (the `ttnn.*_bw` substitution) on the card, both at the dtype the tape
     actually carries, and
  4. reports the worst-case elementwise relative error of each arm against the reference,
     located by parameter path.

Nine ops, nine readings, no sampling. The two arms run on the same inputs in the same
process, so the composed reading is the control the wheel reading is judged against: a wheel
op no worse than the expression it replaces is accepted whether or not it is bit-identical.
"""
import argparse
import json
import pathlib

import numpy as np
import ttnn

# Relative-error floor: below this magnitude a relative error is a division artefact, not a
# disagreement. Fixed here, before any number exists, at the scale of a bf16 unit roundoff on
# an O(1) activation.
FLOOR = 1e-3
FD_H = 1e-5          # central-difference step in float64
FD_PROBES = 8        # probe directions per op
FD_TOL = 1e-6        # relative agreement required of the reference against the differences


def _np(t):
    return np.array(ttnn.to_torch(t).float())


def _dev(v, dt, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.Tensor(np.ascontiguousarray(v.astype(np.float32)), dt).to(layout).to(device)


def _rel(got, ref):
    """Worst-case elementwise relative error with a magnitude floor, and the rel L2."""
    got = got.astype(np.float64).ravel()
    ref = ref.astype(np.float64).ravel()
    den = np.maximum(np.abs(ref), FLOOR)
    err = np.abs(got - ref) / den
    i = int(np.argmax(err))
    l2 = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-300))
    return {"worst_rel": float(err[i]), "worst_index": i,
            "worst_ref": float(ref[i]), "worst_got": float(got[i]), "rel_l2": l2}


# ---------------------------------------------------------------------------------------
# float64 host references. Each is (forward, vjp) over a tuple of inputs.
# ---------------------------------------------------------------------------------------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


REF = {
    "relu_in": (lambda x: np.maximum(x, 0.0),
                lambda g, x: (g * (x > 0.0),)),
    "mul":     (lambda a, b: a * b,
                lambda g, a, b: (g * b, g * a)),
    "scale":   (lambda x, f: x * f,
                lambda g, x, f: (g * f,)),
    "add":     (lambda a, b: a + b,
                lambda g, a, b: (g, g)),
    "relu":    (lambda x: np.maximum(x, 0.0),
                lambda g, x: (g * (x > 0.0),)),
    "sigmoid": (lambda x: _sig(x),
                lambda g, x: (g * _sig(x) * (1.0 - _sig(x)),)),
    "silu":    (lambda x: x * _sig(x),
                lambda g, x: (g * _sig(x) * (1.0 + x * (1.0 - _sig(x))),)),
    "reshape": (lambda x, s: x.reshape(s),
                lambda g, x, s: (g.reshape(x.shape),)),
    "narrow":  (lambda x, st, ln: x[st:st + ln],
                None),      # built below, needs the source shape
    "concat":  (lambda *xs: np.concatenate(xs, axis=0),
                None),
}


def fd_kink_free(name, inputs):
    """The inputs the FINITE DIFFERENCES run on, which are not always the device arm's.

    relu's derivative does not exist at 0 and the device arm puts exact zeros in on purpose:
    that is the boundary where `gtz` and `relu_bw` could disagree and the only place the
    substitution could be wrong. A central difference across that point straddles the kink
    and disagrees with either one-sided answer by construction -- measured at 1.0e-01, which
    is the discontinuity's size and not an error in the reference. So the reference is
    validated away from the kink and the arms are still compared at it.
    """
    if name not in ("relu", "relu_in"):
        return inputs, None
    x = inputs[0].copy()
    near = np.abs(x) < 100 * FD_H
    x[near] = np.sign(x[near] + 1e-30) * (100 * FD_H)
    return (x,) + tuple(inputs[1:]), int(near.sum())


def fd_validate(f, vjp, inputs, g, rng, name):
    """Central finite differences over random probe directions, in float64.

    The identity checked is <g, f(x + h v) - f(x - h v)> / 2h  ==  <vjp(g), v>, which is the
    definition of the VJP contracted against v. Every input that is a float array gets its
    own probes; a shape or a python scalar is held fixed.
    """
    worst = 0.0
    detail = []
    for i, x in enumerate(inputs):
        if not isinstance(x, np.ndarray):
            continue
        for _ in range(FD_PROBES):
            v = rng.standard_normal(x.shape)
            v /= np.linalg.norm(v)
            up = list(inputs); dn = list(inputs)
            up[i] = x + FD_H * v
            dn[i] = x - FD_H * v
            fd = float(np.sum(g * (f(*up) - f(*dn))) / (2.0 * FD_H))
            an = float(np.sum(vjp(g, *inputs)[i] * v))
            rel = abs(fd - an) / max(abs(an), abs(fd), 1e-12)
            worst = max(worst, rel)
            detail.append({"input": i, "fd": fd, "analytic": an, "rel": rel})
    return {"worst_rel": worst, "tol": FD_TOL, "pass": worst <= FD_TOL,
            "probes": len(detail), "op": name}


# ---------------------------------------------------------------------------------------
# device arms
# ---------------------------------------------------------------------------------------
def arms(name, dev, dt, rng):
    """Returns (inputs_f64, upstream_grad_f64, {arm_name: [device grads as numpy]}, paths)."""
    D = lambda v: _dev(v, dt, dev)

    if name == "mul":
        a = rng.standard_normal((2, 256, 128)); b = rng.standard_normal((2, 256, 128))
        g = rng.standard_normal((2, 256, 128))
        ta, tb, tg = D(a), D(b), D(g)
        composed = [_np(ttnn.multiply(tg, tb)), _np(ttnn.multiply(tg, ta))]
        w = ttnn.mul_bw(tg, ta, tb)
        wheel = [_np(w[0]), _np(w[1])]
        return (a, b), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.autograd.mul.grad_a", "tt_bio.autograd.mul.grad_b"]

    if name == "scale":
        x = rng.standard_normal((2, 256, 128)); f = 0.35355339059327373
        g = rng.standard_normal((2, 256, 128))
        tx, tg = D(x), D(g)
        composed = [_np(ttnn.multiply(tg, f))]
        wheel = [_np(ttnn.mul_bw(tg, tx, f)[0])]
        return (x, f), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.autograd.scale.grad_x"]

    if name == "add":
        a = rng.standard_normal((2, 256, 128)); b = rng.standard_normal((2, 256, 128))
        g = rng.standard_normal((2, 256, 128))
        ta, tb, tg = D(a), D(b), D(g)
        composed = [_np(tg), _np(tg)]                 # the shipped backward moves no data
        w = ttnn.add_bw(tg, ta, tb)
        wheel = [_np(w[0]), _np(w[1])]
        return (a, b), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.autograd.add.grad_a", "tt_bio.autograd.add.grad_b"]

    if name == "relu_in":
        # `_VERBS["relu"]` in taped_ttnn reads the INPUT (`gtz(xv)`), so its substitution is
        # relu_bw on the input. Same function, different retained tensor, own reading.
        x = rng.standard_normal((2, 256, 128))
        x.ravel()[:64] = 0.0
        g = rng.standard_normal((2, 256, 128))
        tx, tg = D(x), D(g)
        composed = [_np(ttnn.multiply(tg, ttnn.gtz(tx)))]
        wheel = [_np(ttnn.relu_bw(tg, tx)[0])]
        return (x,), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.taped_ttnn.relu.grad_x"]

    if name == "relu":
        x = rng.standard_normal((2, 256, 128))
        x.ravel()[:64] = 0.0                          # the kink, on purpose
        g = rng.standard_normal((2, 256, 128))
        tx, tg = D(x), D(g)
        ty = ttnn.relu(tx)
        composed = [_np(ttnn.multiply(tg, ttnn.gtz(ty)))]
        wheel = [_np(ttnn.relu_bw(tg, ty)[0])]        # reads the OUTPUT, as the tape retains it
        return (x,), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.autograd.relu.grad_x"]

    if name == "sigmoid":
        # Both sites end up here: `autograd.sigmoid` composed off the retained output and
        # `_VERBS["sigmoid"]` composed off the same, and both now call sigmoid_bw on the input.
        x = rng.standard_normal((2, 256, 128)) * 4.0  # saturating tails, where y*(1-y) is small
        g = rng.standard_normal((2, 256, 128))
        tx, tg = D(x), D(g)
        ty = ttnn.sigmoid(tx)
        composed = [_np(ttnn.multiply(tg, ttnn.multiply(ty, ttnn.rsub(ty, 1.0))))]
        wheel = [_np(ttnn.sigmoid_bw(tg, tx)[0])]     # reads the INPUT, and recomputes
        return (x,), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.autograd.sigmoid.grad_x"]

    if name == "silu":
        x = rng.standard_normal((2, 256, 128)) * 4.0
        g = rng.standard_normal((2, 256, 128))
        tx, tg = D(x), D(g)
        sig = ttnn.sigmoid(tx)
        d = ttnn.multiply(sig, ttnn.add(ttnn.multiply(tx, ttnn.rsub(sig, 1.0)), 1.0))
        composed = [_np(ttnn.multiply(tg, d))]
        wheel = [_np(ttnn.silu_bw(tg, tx)[0])]
        return (x,), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.autograd.silu.grad_x"]

    if name == "reshape":
        x = rng.standard_normal((2, 256, 128))
        g = rng.standard_normal((512, 128))
        tx, tg = D(x), D(g)
        composed = [_np(ttnn.reshape(tg, [2, 256, 128]))]
        wheel = composed                              # no reshape_bw in the wheel
        return (x, (512, 128)), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.autograd.reshape.grad_x"]

    if name == "narrow":
        x = rng.standard_normal((256, 128)); st, ln = 32, 128
        g = rng.standard_normal((ln, 128))
        tg = D(g)
        before, after = st, 256 - st - ln
        zt = lambda n: _dev(np.zeros((n, 128)), dt, dev)
        composed = [_np(ttnn.concat([zt(before), tg, zt(after)], dim=0))]
        # REJECTED: `ttnn.pad` cannot front-pad a tile-layout tensor on device
        # (`pad.cpp:278`, `front_padding_is_zero`). The composed form is what ships, so the
        # wheel arm here IS the composed one and the error is recorded beside it.
        try:
            wheel = [_np(ttnn.pad(tg, padding=[(before, after), (0, 0)], value=0.0))]
            rejected = None
        except Exception as e:
            wheel = composed
            rejected = f"ttnn.pad: {type(e).__name__}: {str(e).splitlines()[0]}"[:200]
        arms.rejected = rejected
        return (x, st, ln), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.autograd.narrow.grad_x"]

    if name == "concat":
        a = rng.standard_normal((128, 128)); b = rng.standard_normal((128, 128))
        g = rng.standard_normal((256, 128))
        ta, tb, tg = D(a), D(b), D(g)
        composed = [_np(ttnn.slice(tg, [0, 0], [128, 128])),
                    _np(ttnn.slice(tg, [128, 0], [256, 128]))]
        # REJECTED: `ttnn.concat_bw` indexes as if every tensor were rank 4 and throws
        # "ShapeBase[] index out of range. 2 not in [-4, 2)" on this rank-2 pair. Probed at
        # rank 4 as well, so the rejection names the boundary rather than one shape.
        try:
            w = ttnn.concat_bw(tg, ta, tb, 0)
            wheel = [_np(w[0]), _np(w[1])]
            arms.rejected = None
        except Exception as e:
            r4a = _t4 = None
            try:
                ta4 = D(a.reshape(1, 1, 128, 128)); tb4 = D(b.reshape(1, 1, 128, 128))
                tg4 = D(g.reshape(1, 1, 256, 128))
                w4 = ttnn.concat_bw(tg4, ta4, tb4, 2)
                r4a = "rank 4, dim 2: OK"
            except Exception as e4:
                r4a = f"rank 4, dim 2: {type(e4).__name__}: {str(e4).splitlines()[0]}"[:160]
            wheel = composed
            arms.rejected = (f"ttnn.concat_bw rank 2, dim 0: "
                             f"{type(e).__name__}: {str(e).splitlines()[0]}"[:200]
                             + " | " + r4a)
        return (a, b), g, {"composed": composed, "wheel": wheel}, [
            "tt_bio.autograd.concat.grad_0", "tt_bio.autograd.concat.grad_1"]

    raise KeyError(name)


def reference(name, inputs, g):
    f, vjp = REF[name]
    if name == "narrow":
        x, st, ln = inputs
        f = lambda x, st=st, ln=ln: x[st:st + ln]
        vjp = lambda g, x, st=st, ln=ln: (
            np.concatenate([np.zeros((st,) + x.shape[1:]), g,
                            np.zeros((x.shape[0] - st - ln,) + x.shape[1:])], axis=0),)
        return f, vjp, (x,)
    if name == "concat":
        return (lambda *xs: np.concatenate(xs, axis=0),
                lambda g, *xs: tuple(np.split(g, np.cumsum([len(x) for x in xs])[:-1], axis=0)),
                inputs)
    if name == "reshape":
        x, s = inputs
        return (lambda x, s=s: x.reshape(s),
                lambda g, x, s=s: (g.reshape(x.shape),), (x,))
    if name == "scale":
        x, fa = inputs
        return (lambda x, fa=fa: x * fa, lambda g, x, fa=fa: (g * fa,), (x,))
    return f, vjp, inputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default="/tmp/of3t/of3t-wheelbw/vjp.json")
    ap.add_argument("--ops",
                    default="mul,scale,add,relu,relu_in,sigmoid,silu,reshape,narrow,concat")
    a = ap.parse_args()
    dt = {"float32": ttnn.float32, "bfloat16": ttnn.bfloat16}[a.dtype]

    dev = ttnn.open_device(device_id=0)
    out = {"dtype": a.dtype, "floor": FLOOR, "fd_h": FD_H, "fd_tol": FD_TOL, "ops": {}}
    try:
        for name in a.ops.split(","):
            rng = np.random.default_rng(abs(hash(name)) % (2 ** 31))
            arms.rejected = None
            try:
                inputs, g, got, paths = arms(name, dev, dt, rng)
            except Exception as e:
                out["ops"][name] = {"arm_error": f"{type(e).__name__}: {e}"[:400]}
                print(name, "ARM ERROR", out["ops"][name]["arm_error"], flush=True)
                continue
            f, vjp, ref_inputs = reference(name, inputs, g)
            fd_inputs, moved = fd_kink_free(name, ref_inputs)
            fd = fd_validate(f, vjp, fd_inputs, g, np.random.default_rng(7), name)
            if moved is not None:
                fd["kink_points_moved_off_zero"] = moved
            ref = vjp(g, *ref_inputs)
            rec = {"fd": fd, "paths": paths, "shape": [list(np.shape(r)) for r in ref],
                   "rejected": getattr(arms, "rejected", None), "arms": {}}
            for arm, grads in got.items():
                per = []
                for k, (gg, rr) in enumerate(zip(grads, ref)):
                    r = _rel(gg, rr)
                    r["path"] = paths[k]
                    per.append(r)
                w = max(per, key=lambda r: r["worst_rel"])
                rec["arms"][arm] = {"per_output": per, "worst_rel": w["worst_rel"],
                                    "worst_path": w["path"], "worst_rel_l2":
                                    max(p["rel_l2"] for p in per)}
            rec["identical"] = bool(
                all(np.array_equal(x, y) for x, y in zip(got["composed"], got["wheel"])))
            out["ops"][name] = rec
            print(f"{name:8s} fd_pass={fd['pass']} fd_worst={fd['worst_rel']:.3e} "
                  f"composed={rec['arms']['composed']['worst_rel']:.3e} "
                  f"wheel={rec['arms']['wheel']['worst_rel']:.3e} "
                  f"bit_identical={rec['identical']} @ {rec['arms']['wheel']['worst_path']}",
                  flush=True)
    finally:
        ttnn.close_device(dev)
    p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
