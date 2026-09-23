"""Stage and op hashes for a boltz2 fold, so repeats of one input can be diffed to where they part.

On the PYTHONPATH only when run.sh puts it there. `tt-bio predict` folds in a spawned worker and
CPython imports sitecustomize in every interpreter, so the hook reaches the process that opens the
device. Adapted from mgx-combos' perf/mgx_combos/probe/sitecustomize.py (HASH mode).

DET_HASH=<file>   append one line per wrapped call: "<pid>:<fold>/<qual>#<n> name=sha1 ...". <fold> counts
                  TrunkModule.forward calls, so the folds of one in-process run separate cleanly.
DET_LEVEL=stage   trunk inputs, each recycle's s/z, conditioning, sampled coords, confidence (default).
DET_LEVEL=step    stage, plus the denoiser output at every diffusion step.
DET_LEVEL=op      step, plus every trunk op class in DET_OPS (default: the pairformer and MSA ops).
DET_DUMP=<dir>    also save the tensor of every call whose "<qual>#<n>" is listed in DET_DUMP_AT.

Reading a device tensor back synchronises the device, which can hide a race: a hashed run that
stops diverging is itself a result, and the stage level exists to perturb the fold as little as
possible.
"""
import functools
import hashlib
import importlib.abc
import importlib.util
import os
import sys
from collections import Counter

HASH = os.environ.get("DET_HASH")
LEVEL = os.environ.get("DET_LEVEL", "stage")
DUMP = os.environ.get("DET_DUMP")
DUMP_AT = set(filter(None, os.environ.get("DET_DUMP_AT", "").split(",")))
OPS = os.environ.get("DET_OPS", "TrunkRecycle,TemplateRecycle,MSALayer,OuterProductMean,"
                     "PairWeightedAveraging,PairformerLayer,TriangleMultiplication,"
                     "TriangleAttention,AttentionPairBias,Transition").split(",")

STAGES = {"tt_bio.tenstorrent": [("TrunkModule", "forward"), ("TrunkModule", "_iteration")],
          "tt_bio.boltz2": [("DiffusionConditioning", "forward_atoms"), ("AtomDiffusion", "sample"),
                            ("ConfidenceModule", "forward")]}
if LEVEL in ("step", "op"):
    STAGES["tt_bio.boltz2"].append(("AtomDiffusion", "preconditioned_network_forward"))
if LEVEL == "op":
    STAGES["tt_bio.tenstorrent"] += [(c, "__call__") for c in OPS]

_calls = Counter()
_fold = [-1]


def _torch(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if type(x).__name__ == "Tensor" and type(x).__module__.startswith("ttnn"):
        import ttnn
        return ttnn.to_torch(x)
    return None


def _digests(x, name, out, key):
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            _digests(x[k], f"{name}.{k}", out, key)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _digests(v, f"{name}[{i}]", out, key)
    else:
        t = _torch(x)
        if t is None:
            return
        t = t.contiguous()
        if DUMP and key in DUMP_AT:
            import torch
            os.makedirs(DUMP, exist_ok=True)
            torch.save(t, os.path.join(DUMP, f"{os.getpid()}_{_fold[0]}_{key}_{name}.pt"))
        b = t.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
        out.append(f"{name}={hashlib.sha1(b).hexdigest()[:10]}")


def _wrap(qual, fn):
    @functools.wraps(fn)
    def w(*a, **kw):
        if qual == "TrunkModule.forward":
            _fold[0] += 1
            _calls.clear()
        n = _calls[qual]
        _calls[qual] += 1
        key = f"{qual}#{n}"
        ins = []
        if qual == "TrunkModule.forward":        # s_inputs, s_init, z_init, feats: host inputs
            _digests(a[1:5], "in", ins, key)
        r = fn(*a, **kw)
        outs = []
        try:
            _digests(r, "out", outs, key)
        except Exception as exc:                  # never let the probe fail a fold
            outs.append(f"hash_error={type(exc).__name__}")
        with open(HASH, "a") as fh:
            fh.write(f"{os.getpid()}:{_fold[0]}/{key} " + " ".join(ins + outs) + "\n")
        return r
    return w


class _After(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in STAGES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        run = spec.loader.exec_module

        def exec_module(mod):
            run(mod)
            for cls, meth in STAGES[name]:
                c = getattr(mod, cls)
                setattr(c, meth, _wrap(f"{cls}.{meth}", c.__dict__[meth] if meth in c.__dict__
                                       else getattr(c, meth)))
        spec.loader.exec_module = exec_module
        return spec


if HASH:
    sys.meta_path.insert(0, _After())
