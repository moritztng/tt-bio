#!/usr/bin/env python3
"""Is the re-ordered AdamW the same optimizer? Bit-identical or it is not.

An optimizer change is a training-result change until proven otherwise, so this runs the
version at `--before-rev` and the working-tree version side by side over the same parameters,
the same gradients and the same seed, and compares every byte either of them can produce:
the fp32 masters, both moments, the device weights the forward would read, the per-parameter
report `step()` returns, and `displacement()`. `np.array_equal` on the raw bytes, not a
tolerance -- a tolerance would pass the one failure worth catching.

Three loops are run because they are three different code paths through `step()`:

  * **batch**   -- gradients on the leaves, clipped once inside `step()`.
  * **sample**  -- `clip_and_accumulate()` per sample, `step()` consuming the accumulator.
                   This is `recipes.py`'s shape and it is upstream OpenFold3's.
  * **disabled** -- a parameter excluded from the step, which is the path that synthesises a
                   zero gradient and decays the moments on momentum alone.

CARD-FREE. `to_host` and `to_device` are replaced, in both modules, by a host stand-in that
does the bfloat16 round-to-nearest-even the PCIe write does and nothing else
(`tt_bio/train/tensors.py:42-47`). Every line of arithmetic under test is the real one; only
the bus is absent, and it is absent identically on both sides of the comparison.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


class _DType:
    """Enough of a ttnn dtype for the two places the optimizer reads one."""

    def __init__(self, label):
        self.label = label

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label


BF16 = _DType("DataType.BFLOAT16")


def _bf16_round(a: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 precision, round-to-nearest-even, returned as float32.

    What `ttnn.from_torch(..., dtype=ttnn.bfloat16)` does to the value. Exact for every
    finite float32 whose rounded exponent does not overflow, which is every weight here.
    """
    out = np.ascontiguousarray(a, dtype=np.float32).copy()
    u = out.view(np.uint32)
    u += np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    u &= np.uint32(0xFFFF0000)
    return out


class DevArray:
    """The device-side weight: bfloat16 values, a dtype that says so, and a device handle."""

    __slots__ = ("a", "dtype")

    def __init__(self, a):
        self.a = a
        self.dtype = BF16

    def device(self):
        return "host"


def _to_host(t, *, dtype=None):
    if isinstance(t, DevArray):
        out = np.ascontiguousarray(t.a, dtype=np.float32)
    elif isinstance(t, np.ndarray):
        out = np.ascontiguousarray(t)
    else:
        raise TypeError(type(t))
    return out.astype(dtype) if dtype is not None else out


def _to_device(arr, device, *, dtype=None, layout=None):
    return DevArray(_bf16_round(arr))


class HostTensor:
    __slots__ = ("value", "grad")

    def __init__(self, value):
        self.value = value
        self.grad = None


def load_before(rev: str, work: Path, path_override: str | None = None):
    """Import `tt_bio/train/optim.py` as it stands at `rev`, as a sibling module.

    Named inside the real package so its `from .mesh import ...` resolves against the
    installed one: the comparison is of this file's change, not of the package around it.
    """
    work.mkdir(parents=True, exist_ok=True)
    if path_override:
        # A box with the sources but no git checkout: qb2 runs this off an rsynced tree and the
        # before-arm's optim.py travels with it.
        path = Path(path_override)
    else:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:tt_bio/train/optim.py"],
                             capture_output=True, text=True, check=True).stdout
        path = work / "optim_before.py"
        path.write_text(src)
    name = "tt_bio.train._optim_before"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, path


SHAPES = {
    "trunk.w0": (8, 16),
    "trunk.b0": (16,),
    "pair.w": (12, 12),
    "conf.head": (5, 7),
    "tiny": (3,),
}


def make_params(seed: int):
    rng = np.random.default_rng(seed)
    return {n: HostTensor(DevArray(_bf16_round(rng.standard_normal(s, dtype=np.float32))))
            for n, s in SHAPES.items()}


def grads(rng, scale=1.0):
    return {n: (rng.standard_normal(s, dtype=np.float32) * np.float32(scale))
            for n, s in SHAPES.items()}


def snapshot(opt, params):
    """Every byte this optimizer holds or wrote, in a comparable form."""
    return {
        "master": {n: v.copy() for n, v in opt.master.items()},
        "exp_avg": {n: np.asarray(opt.exp_avg[n]).copy() for n in opt.master},
        "exp_avg_sq": {n: np.asarray(opt.exp_avg_sq[n]).copy() for n in opt.master},
        "device": {n: np.asarray(t.value.a).copy() for n, t in params.items()},
        "scalars": {"steps": opt.steps, "beta1_pow": opt.beta1_pow,
                    "beta2_pow": opt.beta2_pow, "last_lr": getattr(opt, "last_lr", None),
                    "last_clip": getattr(opt, "last_clip", None),
                    "last_grad_norm": getattr(opt, "last_grad_norm", None)},
        "displacement": opt.displacement(),
    }


def compare(a, b, where, diffs):
    for key in ("master", "exp_avg", "exp_avg_sq", "device"):
        for n in a[key]:
            x, y = a[key][n], b[key][n]
            if not (x.shape == y.shape and x.dtype == y.dtype
                    and np.array_equal(x, y, equal_nan=True)):
                bad = int(np.count_nonzero(x.view(np.uint8) != y.view(np.uint8))) \
                    if x.shape == y.shape else -1
                diffs.append({"where": where, "buffer": key, "param": n,
                              "bytes_differing": bad,
                              "max_abs": float(np.max(np.abs(x - y)))
                              if x.shape == y.shape else None})
    for k, v in a["scalars"].items():
        if repr(v) != repr(b["scalars"][k]):
            diffs.append({"where": where, "buffer": "scalars", "param": k,
                          "before": v, "after": b["scalars"][k]})
    for k, v in a["displacement"].items():
        if repr(v) != repr(b["displacement"][k]):
            diffs.append({"where": where, "buffer": "displacement", "param": k,
                          "before": v, "after": b["displacement"][k]})


def compare_reports(ra, rb, where, diffs):
    if sorted(ra) != sorted(rb):
        diffs.append({"where": where, "buffer": "report", "param": "keys",
                      "before": sorted(ra), "after": sorted(rb)})
        return
    for n in ra:
        for k in ra[n]:
            if repr(ra[n][k]) != repr(rb[n][k]):
                diffs.append({"where": where, "buffer": "report", "param": f"{n}.{k}",
                              "before": ra[n][k], "after": rb[n][k]})


def run_arm(mod, mode, steps, seed, sched):
    params = make_params(seed)
    opt = mod.AdamW(params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0.01, clip_norm=10.0,
                    schedule=(lambda s: mod.af3_lr(s, 1.8e-3, warmup_steps=4)) if sched else None)
    reports = []
    for k in range(steps):
        rng = np.random.default_rng(1000 + k)
        if mode == "batch":
            for n, t in params.items():
                t.grad = grads(rng)[n]
            reports.append(opt.step())
        elif mode == "sample":
            for s in range(3):
                g = grads(rng, scale=1.0 + 60.0 * (s == 1))  # one sample over the clip norm
                for n, t in params.items():
                    t.grad = g[n]
                opt.clip_and_accumulate()
                opt.zero_grad()
            reports.append(opt.step())
        elif mode == "disabled":
            for n, t in params.items():
                t.grad = grads(rng)[n]
            reports.append(opt.step(disabled=("conf.head",) if k else ()))
        else:
            raise ValueError(mode)
        for t in params.values():
            t.grad = None
    return opt, params, reports


def _guarded(fn):
    """Return the call's result, or the exact type and text of what it raised."""
    try:
        return fn()
    except Exception as exc:                                     # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    # Pinned, not HEAD: once the change lands, HEAD carries it and the comparison compares
    # a tree with itself. 975b6a172 is `of3t-restep`'s tip, the last commit before this row.
    ap.add_argument("--before-rev", default="975b6a172")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--before-file", default=None,
                    help="path to the before-arm's optim.py; for a box with no git tree")
    ap.add_argument("--work", default="/tmp/of3t/of3t-optorder")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out" / "optfid.json"))
    a = ap.parse_args()

    before, before_path = load_before(a.before_rev, Path(a.work), a.before_file)
    import tt_bio.train.optim as after

    for mod in (before, after):
        mod.to_host = _to_host
        mod.to_device = _to_device

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "before_rev": a.before_rev,
        "before_sha": (f"file:{a.before_file}" if a.before_file else subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", a.before_rev],
            capture_output=True, text=True, check=True).stdout.strip()),
        "after_commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "after_dirty": bool(os.popen(
            f"git -C {REPO} status --porcelain tt_bio/train/optim.py").read().strip()),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "card_opened": False, "ttnn_imported": "ttnn" in sys.modules}}

    diffs, arms = [], []
    for mode in ("batch", "sample", "disabled"):
        for sched in (False, True):
            where = f"{mode}/sched={sched}"
            ob, pb, rb = run_arm(before, mode, a.steps, a.seed, sched)
            oa, pa, ra = run_arm(after, mode, a.steps, a.seed, sched)
            n0 = len(diffs)
            compare(snapshot(ob, pb), snapshot(oa, pa), where, diffs)
            for i, (x, y) in enumerate(zip(rb, ra)):
                compare_reports(x, y, f"{where}/step{i}", diffs)
            # The control has to agree on the FAILING case too: a short bf16 run legitimately
            # sits under the band, and the raise is as much a result as the dict is.
            cb, ca = _guarded(ob.check_displacement), _guarded(oa.check_displacement)
            if repr(cb) != repr(ca):
                diffs.append({"where": where, "buffer": "check_displacement",
                              "before": cb, "after": ca})
            d = ob.displacement()
            arms.append({"arm": where, "steps": a.steps, "params": len(SHAPES),
                         "differences": len(diffs) - n0,
                         "master_displacement": round(d["master"], 12),
                         "ratio": round(d["ratio"], 12),
                         "check_displacement": cb if isinstance(cb, str) else "inside band"})

    # The ordering claim itself, asserted rather than described: the moments do not exist
    # before the first step, and they do after it.
    probe = make_params(a.seed)
    opt = after.AdamW(probe, lr=3e-4)
    order = {"moments_len_at_construction": [len(opt.exp_avg), len(opt.exp_avg_sq)],
             "init_len_at_construction": len(opt.init),
             "has_init_device_attr": hasattr(opt, "init_device"),
             "has_init_master_attr": hasattr(opt, "init_master")}
    for n, t in probe.items():
        t.grad = np.zeros(SHAPES[n], np.float32)
    opt.step()
    order["moments_len_after_one_step"] = [len(opt.exp_avg), len(opt.exp_avg_sq)]
    out["ordering"] = order

    # `init` really is what BOTH of the old copies were, on the old code's own arrays. Over a
    # wide shape sweep rather than the five above, because the claim is that the SECOND copy's
    # `.reshape(master[n].shape)` is a no-op and a reshape is exactly where a shape could
    # matter: 1-D, trailing ones, a singleton, and a rank-4 pair tensor are all in here.
    rng = np.random.default_rng(a.seed + 1)
    wide = {f"p{i}": HostTensor(DevArray(_bf16_round(
        rng.standard_normal(sh, dtype=np.float32))))
        for i, sh in enumerate(
            [(1,), (7,), (1, 1), (3, 1), (1, 5), (2, 3, 4), (2, 1, 3, 1), (16, 16),
             (4, 4, 4, 4), (33,), (1, 64), (64, 1)] * 20)}
    ob2 = before.AdamW(wide, lr=3e-4)
    dup = [n for n in ob2.master
           if not (ob2.init_master[n].shape == ob2.init_device[n].shape
                   and np.array_equal(ob2.init_master[n], ob2.init_device[n]))]
    same_as_after = [n for n in ob2.master
                     if not np.array_equal(ob2.init_master[n], after.AdamW(wide).init[n])]
    out["duplicate_check"] = {
        "params": len(ob2.master),
        "distinct_shapes": len({tuple(v.shape) for v in ob2.master.values()}),
        "init_master_differs_from_init_device_on": dup,
        "new_init_differs_from_old_init_master_on": same_as_after,
        "identical": not dup and not same_as_after}

    # The one behaviour a lazily built moment could break: a checkpoint written before the
    # first step. `save_adapter` reads `opt.exp_avg[name]` by name, which materialises it.
    try:
        import safetensors  # noqa: F401
        from tt_bio.train import checkpoint as ck
        ck.to_device = _to_device
        ckres = {}
        for label, mod in (("before", before), ("after", after)):
            pr = make_params(a.seed)
            o = mod.AdamW(pr, lr=3e-4)
            path = Path(a.work) / f"adapter_{label}.safetensors"
            ck.save_adapter(path, o)
            pr2 = make_params(a.seed + 99)
            o2 = mod.AdamW(pr2, lr=3e-4)
            ck.load_adapter(path, pr2, "host", opt=o2)
            ckres[label] = {
                "master": {n: v.copy() for n, v in o2.master.items()},
                "exp_avg": {n: np.asarray(o2.exp_avg[n]).copy() for n in o2.master},
                "exp_avg_sq": {n: np.asarray(o2.exp_avg_sq[n]).copy() for n in o2.master},
                "device": {n: np.asarray(t.value.a).copy() for n, t in pr2.items()},
                "scalars": {}, "displacement": o2.displacement()}
            path.unlink(missing_ok=True)
        compare(ckres["before"], ckres["after"], "checkpoint/save-before-any-step", diffs)
        out["checkpoint_roundtrip"] = "save before any step, then load: compared"
    except ImportError:
        out["checkpoint_roundtrip"] = ("safetensors absent on this interpreter; run this script "
                                       "on qb2's bcx_e2e_venv for the checkpoint arm")

    out["arms"] = arms
    out["differences"] = diffs
    out["VERDICT"] = ("BIT-IDENTICAL" if not diffs else f"DIFFERS in {len(diffs)} places")
    out["env"]["ttnn_imported_after"] = "ttnn" in sys.modules
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    if not a.before_file:
        before_path.unlink(missing_ok=True)
    print(json.dumps({k: out[k] for k in ("VERDICT", "arms", "ordering", "duplicate_check")},
                     indent=1, default=str))
    if diffs:
        print(json.dumps(diffs[:10], indent=1, default=str))
    return 0 if not diffs else 1


if __name__ == "__main__":
    raise SystemExit(main())
