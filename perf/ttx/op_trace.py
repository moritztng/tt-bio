#!/usr/bin/env python3
"""Where do two ttnn stacks first disagree inside a real fold?

The 298 aa control folds identically on 0.68.0 and 0.69.0 and wrong on 0.73.1 and 0.78.0, so the
break is a real change in the op library and not anything above it. A probe of 20 ops at
production-ish shapes (`perf/b2z_ttnn/stack_numerics.py`) found 16 bit-identical and none broken,
which means the failing call is one the probe does not make: the shape, the config or the memory
layout matters. So stop guessing at the call and take it from the fold itself.

Every `ttnn.<op>` tt-bio calls is wrapped. After each call the output tensor is read back and
hashed, and one JSON line per call records the op, its operand shapes and dtypes, and that hash.
The fold is aborted once --calls calls are traced, so both arms trace the same prefix. Diffing two
traces gives the FIRST call index whose output differs, with the op and the shapes that reached it.

    <venv>/bin/python op_trace.py --out trace_069.jsonl --calls 1200
    op_trace.py --diff trace_069.jsonl trace_072.jsonl

A hash is bytes of the fp32 widening of whatever the op returned, so bf16 in bf16 out compares
exactly and nothing is hidden by the readback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))


class StopTrace(Exception):
    pass


def _shape_of(x):
    s = getattr(x, "shape", None)
    if s is None:
        return None
    try:
        return list(s)
    except Exception:
        return str(s)


def _dtype_of(x):
    d = getattr(x, "dtype", None)
    return None if d is None else str(d)


def _ref_linear(torch, a, b, bias):
    """float64 reference for `ttnn.linear`/`ttnn.matmul`, from the operands the device got."""
    out = a.to(torch.float64) @ b.to(torch.float64)
    return out if bias is None else out + bias.to(torch.float64)


def install(ttnn, sink: list, cap: int, names: list[str],
            dump: set[int] | None = None, dump_dir: Path | None = None,
            score: set[int] | None = None) -> None:
    """Wrap ttnn ops in place. tt-bio calls them as `ttnn.<name>`, so a module attribute is the
    whole interception point: no model code is touched and the wrapper is invisible to it."""
    Tensor = ttnn.Tensor

    def wrap(name, fn):
        def inner(*a, **kw):
            out = fn(*a, **kw)
            if len(sink) >= cap:
                raise StopTrace
            rec = {
                "i": len(sink), "op": name,
                "in": [{"shape": _shape_of(x), "dtype": _dtype_of(x)} for x in a
                       if isinstance(x, Tensor)],
                "kw": sorted(k for k in kw if kw[k] is not None),
                "out_shape": _shape_of(out), "out_dtype": _dtype_of(out),
            }
            if isinstance(out, Tensor):
                try:
                    t = ttnn.to_torch(out).float().contiguous()
                    rec["sha"] = hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]
                    rec["absmax"] = round(float(t.abs().max()), 6)
                    rec["mean"] = round(float(t.mean()), 8)
                    # the mean cancels a sign-symmetric difference; the L2 norm does not, so a
                    # cross-stack comparison of these two readings separates a coherent bias from
                    # a random one without dumping the tensor
                    rec["l2"] = round(float(t.pow(2).sum().sqrt()), 6)
                    if dump is not None and len(sink) in dump:
                        np.save(dump_dir / f"call{len(sink)}.npy", t.numpy())
                except Exception as e:                       # a tensor a readback cannot reach
                    rec["sha"] = f"unreadable:{type(e).__name__}"
            else:
                rec["sha"] = "not-a-tensor"
            if score and len(sink) in score and name in ("linear", "matmul"):
                # Score the call the fold actually made -- its own shapes, its own
                # program_config, its own memory_config -- against float64 built from the same
                # operands. No config is reconstructed and no reference build is needed, so
                # nothing can differ between the arms except the op.
                import torch as _t
                try:
                    ta = [ttnn.to_torch(x) for x in a if isinstance(x, Tensor)]
                    bias = kw.get("bias")
                    tb = ttnn.to_torch(bias) if isinstance(bias, Tensor) else None
                    ref = _ref_linear(_t, ta[0], ta[1], tb)
                    got = ttnn.to_torch(out).to(_t.float64)
                    if got.shape != ref.shape:
                        ref = ref.reshape(got.shape)
                    err = got - ref
                    rec["vs_float64"] = {
                        "rel_l2": float(err.norm() / ref.norm()),
                        "max_abs": float(err.abs().max()),
                        "ref_absmax": float(ref.abs().max()),
                        "pcc": float(_t.corrcoef(_t.stack(
                            [got.flatten(), ref.flatten()]))[0, 1]),
                        "kw": {k: str(v)[:300] for k, v in kw.items()
                               if not isinstance(v, Tensor)},
                    }
                except Exception as e:
                    rec["vs_float64"] = {"raised": f"{type(e).__name__}: {e}"}
            sink.append(rec)
            return out
        return inner

    for name in names:
        fn = getattr(ttnn, name, None)
        if callable(fn):
            setattr(ttnn, name, wrap(name, fn))


# The ops a Boltz-2 trunk step actually issues, plus every reduction (the four ops the default
# probe found differing all reduce across the last axis). `from_torch`/`to_torch` are deliberately
# absent: they move bytes, they do not compute, and wrapping to_torch would recurse.
OPS = [
    "matmul", "linear", "softmax", "layer_norm", "rms_norm", "add", "sub", "mul", "div",
    "sigmoid", "silu", "gelu", "exp", "rsqrt", "sqrt", "reciprocal", "tanh", "relu",
    "mean", "sum", "max", "min", "permute", "transpose", "concat", "reshape", "pad",
    "typecast", "clone", "to_layout", "to_memory_config", "reshard", "interleaved_to_sharded",
    "sharded_to_interleaved", "multiply", "subtract", "where", "sigmoid_accurate",
]


def run(out: Path, cap: int, dump: set[int] | None, dump_dir: Path | None,
        score: set[int] | None) -> int:
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this checkout")
    import ab_flag_levers as AB

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="ttx-trace-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    AB._seed_msa(AB.FIX / "cdk2x2_298.yaml", (AB.FIX / "cdk2x2_298.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    cfg["seed"] = 0
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("ttx-trace", cfg)

    sink: list = []
    os.environ["TT_BIO_SHARED_DRAW_SEED"] = "0"
    install(ttnn, sink, cap, OPS, dump, dump_dir, score)
    try:
        state.predict_one(AB.FIX / "cdk2x2_298.yaml", cfg)
    except StopTrace:
        pass
    except Exception as e:
        print(f"fold raised after {len(sink)} traced calls: {type(e).__name__}: {e}", flush=True)

    hdr = {"ttnn": md.version("ttnn"), "torch": torch.__version__,
           "arch": str(dev.arch()), "grid": list(map(int, (
               dev.compute_with_storage_grid_size().x, dev.compute_with_storage_grid_size().y))),
           "calls": len(sink), "cap": cap}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"header": hdr}) + "\n")
        for r in sink:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(sink)} calls -> {out}  ttnn={hdr['ttnn']}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return 0


def diff(a: Path, b: Path) -> int:
    ra = [json.loads(l) for l in a.read_text().splitlines() if l.strip()]
    rb = [json.loads(l) for l in b.read_text().splitlines() if l.strip()]
    ha, hb = ra[0]["header"], rb[0]["header"]
    ra, rb = ra[1:], rb[1:]
    print(f"A {a.name}: ttnn {ha['ttnn']}, {ha['calls']} calls")
    print(f"B {b.name}: ttnn {hb['ttnn']}, {hb['calls']} calls")
    n = min(len(ra), len(rb))
    first = None
    for i in range(n):
        x, y = ra[i], rb[i]
        if x["op"] != y["op"] or x["in"] != y["in"]:
            print(f"\ncall sequences diverge at {i}: A={x['op']}{[d['shape'] for d in x['in']]} "
                  f"B={y['op']}{[d['shape'] for d in y['in']]}")
            return 0
        if x["sha"] != y["sha"]:
            first = i
            break
    if first is None:
        print(f"\nfirst {n} calls identical")
        return 0
    print(f"\nFIRST DIVERGENCE at call {first}")
    for tag, r in (("A", ra[first]), ("B", rb[first])):
        print(f"  {tag} {r['op']:22s} in={[ (d['shape'], d['dtype']) for d in r['in']]} "
              f"kw={r['kw']} out={r['out_shape']} {r['out_dtype']} "
              f"sha={r['sha']} absmax={r.get('absmax')} mean={r.get('mean')}")
    print("\n  context, the 6 calls before it (identical on both):")
    for r in ra[max(0, first - 6):first]:
        print(f"    {r['i']:5d} {r['op']:22s} out={r['out_shape']} {r['out_dtype']} sha={r['sha']}")
    same = sum(1 for i in range(first, n) if ra[i]["sha"] == rb[i]["sha"])
    print(f"\n  of the {n - first} calls from there on, {same} still agree")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--calls", type=int, default=1200)
    ap.add_argument("--diff", nargs=2, type=Path)
    ap.add_argument("--dump", default="", help="comma separated call indices to save as .npy")
    ap.add_argument("--dump-dir", type=Path)
    ap.add_argument("--score", default="", help="call indices to score against float64")
    a = ap.parse_args()
    if a.diff:
        return diff(*a.diff)
    if not a.out:
        ap.error("--out or --diff")
    dump = {int(x) for x in a.dump.split(",") if x.strip()} or None
    dd = a.dump_dir or a.out.parent
    if dump:
        dd.mkdir(parents=True, exist_ok=True)
    scored = {int(x) for x in a.score.split(",") if x.strip()} or None
    return run(a.out, a.calls, dump, dd, scored)


if __name__ == "__main__":
    raise SystemExit(main())
