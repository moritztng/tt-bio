#!/usr/bin/env python3
"""bc4-bet-mm2 step 1: every call of the two matmul sites in one warm BindCraft 2 iteration, by signature.

The census (bc4-bet-unnamed u2) names `autograd.py:_matmul` and `pair_mm.py:or_linear` by seconds but not
by shape. This runs the PD-L1 campaign, hooks both sites for the gradient iterations in
[--at, --at + --keep), and writes one row per distinct (shapes, dtypes, placement, plan) with its call
count and the tt_bio frames that issued it. No profiler, no timing: the micro A/B prices each row.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 python perf/bc4_mm2/shapes.py --out OUT.json
"""
import argparse
import collections
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--bc2", type=Path, default=Path("~/bcx_e2e/bc2").expanduser())
ap.add_argument("--binder", type=int, default=80)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--at", type=int, default=4)
ap.add_argument("--keep", type=int, default=2)
a = ap.parse_args()
sys.path.insert(0, str(a.bc2))

import ttnn  # noqa: E402,F401
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
from tt_bio import bindcraft2 as bc2  # noqa: E402
from tt_bio import autograd as AG, pair_mm as PM  # noqa: E402

st = dict(n=0)
rows = collections.Counter()
SKIP = ("autograd.py", "taped_ttnn.py", "pair_mm.py", "ops.py", "shapes.py")


def kept():
    return a.at <= st["n"] < a.at + a.keep


def tens(t):
    try:
        mc = t.memory_config()
        place = f"{str(mc.buffer_type).split('.')[-1]}/{str(mc.memory_layout).split('.')[-1]}"
    except Exception:                                                    # noqa: BLE001
        place = "?"
    return dict(shape=[int(d) for d in t.shape], dtype=str(t.dtype).split(".")[-1], place=place)


def cfg(c):
    if c is None:
        return None
    return {k: str(getattr(c, k)) for k in ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode")
            if hasattr(c, k)}


def plan(p):
    if p is None:
        return None
    out = {"type": type(p).__name__}
    for k in ("in0_block_w", "out_subblock_h", "out_subblock_w", "per_core_M", "per_core_N",
              "compute_with_storage_grid_size", "fuse_batch", "mcast_in0"):
        if hasattr(p, k):
            out[k] = str(getattr(p, k))
    return out


def opt(kw, k):
    return str(kw[k]) if kw.get(k) is not None else None


def where():
    f, names = sys._getframe(2), []
    while f is not None and len(names) < 3:
        fn = f.f_code.co_filename
        if "tt_bio" in fn and not fn.endswith(SKIP):
            names.append(f"{Path(fn).name}:{f.f_code.co_name}:{f.f_lineno}")
        f = f.f_back
    return " < ".join(names)


real_mm = AG._matmul


def _matmul(x, y, **kw):
    if kept():
        key = dict(site="_matmul", a=tens(x), b=tens(y), transpose_a=bool(kw.get("transpose_a")),
                   transpose_b=bool(kw.get("transpose_b")), plan=plan(kw.get("program_config")),
                   core_grid=opt(kw, "core_grid"), out_dtype=opt(kw, "dtype"), out_mem=opt(kw, "memory_config"),
                   cfg=cfg(kw.get("compute_kernel_config")), where=where())
        rows[json.dumps(key, sort_keys=True)] += 1
    return real_mm(x, y, **kw)


AG._matmul = _matmul
real_or = PM.or_linear


def or_linear(linear, x, w, bias=None, **kw):
    if not kept():
        return real_or(linear, x, w, bias, **kw)
    s0 = PM.STATS[0]
    out = real_or(linear, x, w, bias, **kw)
    xv, wv = getattr(x, "value", x), getattr(w, "value", w)
    key = dict(site="or_linear", a=tens(xv), b=tens(wv),
               bias=None if bias is None else tens(getattr(bias, "value", bias)),
               taped=isinstance(x, AG.Tensor) or isinstance(w, AG.Tensor), served=PM.STATS[0] != s0,
               activation=str(kw.get("activation")), core_grid=opt(kw, "core_grid"), out_dtype=opt(kw, "dtype"),
               out_mem=opt(kw, "memory_config"), cfg=cfg(kw.get("compute_kernel_config")), where=where(),
               extra=sorted(k for k in kw if k not in ("activation", "core_grid", "dtype", "memory_config",
                                                        "compute_kernel_config")))
    rows[json.dumps(key, sort_keys=True)] += 1
    return out


PM.or_linear = or_linear
TTS = sys.modules.get("tt_bio.taped_ttnn")
if TTS is not None:
    TTS.forget_shim_bindings()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def write():
    out = [dict(calls_per_it=n / a.keep, **json.loads(k)) for k, n in rows.most_common()]
    a.out.write_text(json.dumps(dict(
        sha=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
        host=socket.gethostname(), chip=os.environ.get("TT_VISIBLE_DEVICES"),
        env={k: v for k, v in sorted(os.environ.items()) if k.startswith("TT_BIO_")},
        binder=a.binder, seed=a.seed, at=a.at, keep=a.keep, pair_mm_armed=PM.armed(), rows=out), indent=1))


cls = bc2.design_model_class()
real_grads = cls.sequence_gradients


def sequence_gradients(self, *args, **kwargs):
    st["n"] += 1
    try:
        return real_grads(self, *args, **kwargs)
    finally:
        print(json.dumps(dict(ev="iter", n=st["n"], rows=len(rows))), flush=True)
        if st["n"] >= a.at + a.keep - 1:
            write()
            os._exit(0)


cls.sequence_gradients = sequence_gradients
project = a.out.parent / f"campaign_s{a.seed}"
settings = cleaned_campaign_settings(read_settings(str(a.bc2 / "examples" / "pdl1.json"), parse_setting_overrides(
    [f"campaign_seed={a.seed}", "max_trajectories=1", f"project_folder={project}", f"binder_lengths=[{a.binder}]"])))
weights = {k: os.environ[v] for k, v in (("af2_weights", "JAPANFOLD_BC2_AF2_WEIGHTS"),) if os.environ.get(v)}
with bc2.campaign_predictor(**({"checkpoints": weights["af2_weights"]} if weights else {})):
    bc2.run_campaign(settings, str(project), trajectories_per_card=1, **weights)
write()
os._exit(1)
