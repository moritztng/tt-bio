#!/usr/bin/env python3
"""How much of OpenFold3's diffusion path the shared tape follows.

D20 puts 88.54 % of the reference's squared gradient norm in `diffusion_module` and says
none of it is on a taped training path. That is a claim about a path **no census has ever
run**: `of3t-tape` measured trunk (11003/11003), MSA and template and stopped, and
`of3t-gradients` listed `diffusion_rollout` as NOT COVERED for want of a model forward.
The trunk turned out to need an R11-class fix before it could be differentiated at all, so
measuring first is cheap insurance against three days of porting into a wall.

This REUSES `perf/of3t_tape`'s instrument rather than writing a second one. `tapecount` is
imported unchanged -- same four buckets, same two seams into the tape, same `_on_tape`
predicate -- and so is `of3_coverage._device_weights` for the denominator. What is new is
the assembly, and only because it had to be: `OF3DiffusionModule` takes twenty-six
operands, twelve of them mask-derived gather indices and block masks, and the golden that
carries them (`diffusion_module_xlout_real`) is **not in qb2's copy of
`~/of3_ref_out.pkl`** -- that copy is a seven-key trunk-only capture. They come instead
from `capture_operands.py`, which records what the shipped fold passes.

**The shipped diffusion module runs in fp32, not bf16** (`openfold3_fold.py:205`,
`OF3_DIFFUSION_FP32_DEVICE` default-on), so the module is built under the same
`device_dtype_override` the fold uses. `--bf16` measures the opt-out arm instead. A census
taken at the wrong dtype is a census of a path production does not run.

Which operands are taped is a decision and it is recorded here. The seven that carry a
gradient in upstream's `_train_diffusion` are taped: si_trunk, si, zij, cl0, plm0,
rl_noisy, xl_noisy. The rest -- atom masks, gather indices, the block-mask bias, the
atom-to-token mean matrix, the token masks -- are host-precomputed constants with no
gradient in anyone's training step, so they stay raw and a call carrying only those is
honestly a `raw` entry rather than a gap.

    python3 perf/of3t_diffusion/of3_diffusion_coverage.py --survey
    python3 perf/of3t_diffusion/of3_diffusion_coverage.py            # strict
    python3 perf/of3t_diffusion/of3_diffusion_coverage.py --backward # closures run too
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf", "of3t_tape"))
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
OUT = "perf/of3t_diffusion"
BUNDLE = "perf/of3t_diffusion/operands_ubq.pt"

# The modules the diffusion forward reaches. Named rather than discovered, because the
# import-inside-a-function check has to be a file-level statement over a fixed set.
DIFFUSION_MODULES = [
    "openfold3_diffusion_module.py",
    "openfold3_diffusion_transformer.py",
    "openfold3_atom_transformer.py",
    "openfold3_diffusion_decoder.py",
    "openfold3_diffusion.py",
    "openfold3_sample_diffusion.py",
]

# The differentiable operands, in `OF3DiffusionModule.__call__` order. Everything else in
# the bundle is host-precomputed and has no gradient on either stack.
TAPED = ("si_trunk", "si", "zij", "cl0", "plm0", "rl_noisy", "xl_noisy")


def function_local_ttnn_imports(root="tt_bio"):
    """Every `import ttnn` that is NOT at module scope, in the diffusion path.

    PROTOCOL §6/A2: `tape()` installs itself by rebinding the module-global name `ttnn`,
    so an `import ttnn` inside a function binds the REAL module at call time and those
    sites run untaped with nothing raising. A runtime census cannot see them -- they never
    reach the proxy -- so this is not redundant with the count, it is the only thing that
    can find this shape of gap.
    """
    hits = []
    for name in DIFFUSION_MODULES:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            hits.append({"file": name, "line": 0, "what": "FILE MISSING"})
            continue
        tree = ast.parse(open(path).read(), filename=path)
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Import):
                    for al in node.names:
                        if al.name.split(".")[0] == "ttnn":
                            hits.append({"file": name, "line": node.lineno,
                                         "fn": fn.name, "what": "import ttnn"})
                elif (isinstance(node, ast.ImportFrom)
                      and (node.module or "").split(".")[0] == "ttnn"):
                    hits.append({"file": name, "line": node.lineno, "fn": fn.name,
                                 "what": "from ttnn import ..."})
    return hits


def load_operands(dev, path, force_dtype=None):
    """Push the captured bundle back to device, dtype and layout as recorded."""
    import torch
    import ttnn
    rec = torch.load(path, weights_only=False)
    dtypes = {"DataType.FLOAT32": ttnn.float32, "DataType.BFLOAT16": ttnn.bfloat16,
              "DataType.UINT32": ttnn.uint32, "DataType.INT32": ttnn.int32}
    layouts = {"Layout.TILE": ttnn.TILE_LAYOUT, "Layout.ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT}
    out = {}
    for n, r in rec.items():
        if r["kind"] == "scalar":
            out[n] = r["v"]
            continue
        dt = dtypes[r["dtype"]]
        if force_dtype is not None and dt in (ttnn.float32, ttnn.bfloat16):
            dt = force_dtype
        t = r["t"]
        if dt in (ttnn.float32, ttnn.bfloat16):
            t = t.float()
        out[n] = ttnn.from_torch(t, layout=layouts[r["layout"]], device=dev, dtype=dt)
    order = [n for n in rec]
    return out, order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", action="store_true")
    p.add_argument("--backward", action="store_true")
    p.add_argument("--bf16", action="store_true",
                   help="the OF3_DIFFUSION_FP32_DEVICE=0 opt-out arm")
    p.add_argument("--l1", action="store_true")
    p.add_argument("--bundle", default=BUNDLE)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    stat = {"function_local_ttnn_imports": function_local_ttnn_imports()}
    print("§6 file check: %d function-local `import ttnn` in the diffusion path"
          % len(stat["function_local_ttnn_imports"]), flush=True)
    for h in stat["function_local_ttnn_imports"]:
        print("   %s:%d in %s -- %s" % (h["file"], h["line"], h.get("fn", "?"), h["what"]))

    import torch
    import ttnn
    import tapecount
    from of3_coverage import _device_weights

    import tt_bio.tenstorrent as T
    loaded = []
    orig_load = T.Module.torch_to_tt

    def recording(self, key, *args, **kw):
        t = orig_load(self, key, *args, **kw)
        loaded.append((key, t))
        return t

    T.Module.torch_to_tt = recording

    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion_module import OF3DiffusionModule
    from tt_bio.openfold3_weights import _sub

    t0 = time.perf_counter()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    act = ttnn.bfloat16 if a.bf16 else ttnn.float32
    ops_in, order = load_operands(dev, a.bundle, force_dtype=act)
    print("[%ds] operands loaded from %s: %d tokens, %d atoms, %d blocks, act=%s"
          % (time.perf_counter() - t0, a.bundle, ops_in["n_token"], ops_in["n_atom"],
             ops_in["nb"], act), flush=True)

    with device_dtype_override(act):
        mod = OF3DiffusionModule(_sub(sd, "diffusion_module"), cfg)
    T.Module.torch_to_tt = orig_load

    # The denominator, same discipline as of3_coverage: what the loader returned is not
    # what the forward multiplies. `OF3DiffusionModule._w_tt` transposes every weight on
    # HOST and caches the result, so the tensor a matmul actually sees never passed
    # through the loader and the tape would see a constant. Both numbers are reported.
    walked = _device_weights(mod)
    by_loader = {id(t) for _, t in loaded}
    derived = {k: v for k, v in walked.items() if id(v) not in by_loader}
    for t in walked.values():
        ag.parameter(t)
    stat.update(weights_loaded=len(loaded), weights_reachable=len(walked),
                weights_derived_in_init=len(derived), derived_names=sorted(derived)[:40])
    print("[%ds] diffusion_module built: %d device weights reachable, %d from the loader, "
          "%d derived in __init__ and invisible to it"
          % (time.perf_counter() - t0, len(walked), len(loaded), len(derived)), flush=True)

    tapecount.install(survey=a.survey, l1=a.l1)

    args = [ag.Tensor(ops_in[n], requires_grad=True) if n in TAPED else ops_in[n]
            for n in order]

    err = None
    ag._TOUCHED.clear()
    with device_dtype_override(act), ag.tape():
        try:
            out = mod(*args)
            if a.backward:
                ag.backward([out])
        except Exception as e:                       # a gap is a finding, not a crash
            err = e

    touched = sum(1 for t in walked.values() if id(t) in ag._TOUCHED)
    with_grad = sum(1 for t in walked.values()
                    if getattr(ag._PARAMS.get(id(t)), "grad", None) is not None)
    nograd = sorted(n for n, t in walked.items()
                    if getattr(ag._PARAMS.get(id(t)), "grad", None) is None)
    stat.update(weights_touched=touched, weights_with_grad=with_grad,
                weights_without_grad=len(nograd), nograd_names=nograd)
    print("  weights: %d reachable and registered as leaves, %d resolved during the "
          "forward, %d carrying a gradient" % (len(walked), touched, with_grad), flush=True)

    mode = "survey" if a.survey else "strict"
    suffix = ("_bf16" if a.bf16 else "") + ("_bw" if a.backward else "") + a.tag
    name = "diffusion_%s%s" % (mode, suffix)
    tot = tapecount.report("diffusion_module %s%s" % (mode, suffix),
                           path=os.path.join(OUT, "coverage_%s.json" % name))
    stat.update(totals=tot, params=len(loaded), survey=a.survey, backward=a.backward,
                act_dtype=str(act), tokens=ops_in["n_token"], atoms=ops_in["n_atom"],
                error=None if err is None else "%s: %s" % (type(err).__name__, err))
    json.dump(stat, open(os.path.join(OUT, "run_%s.json" % name), "w"), indent=1)
    if err is not None:
        print("\nFORWARD RAISED: %s: %s" % (type(err).__name__, err), flush=True)
        import traceback
        traceback.print_exception(type(err), err, err.__traceback__)
        return 1
    print("\n[%ds] done" % (time.perf_counter() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
