#!/usr/bin/env python3
"""Warm BindCraft 2 iterations in process: the shapes `autograd._join_slices` concatenates, per-iteration wall
and a digest of every sequence gradient, with this row's levers on (default) or off (--off).

    TT_VISIBLE_DEVICES=<chip> python perf/bc4_sites/shapes.py --out DIR [--off] --iters 6

Digests of an --off run and a default run on the same seed say which levers are bit-exact: the permute VJP
must be, split-K is graded in float64 and on designs.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--bc2", type=Path, default=Path("~/bcx_e2e/bc2").expanduser())
ap.add_argument("--binder", type=int, default=80)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--iters", type=int, default=6)
ap.add_argument("--off", action="store_true", help="split-K off and the stock (1,2,0) permute VJP")
ap.add_argument("--no-split", action="store_true", help="split-K off, permute VJP lever on")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.bc2))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import numpy as np  # noqa: E402
import ttnn  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
from tt_bio import bindcraft2 as bc2, autograd as ag, taped_ttnn as tt  # noqa: E402

CHIP = os.environ.get("TT_VISIBLE_DEVICES", "?")
log = open(a.out / "shapes.jsonl", "a")


def emit(r):
    log.write(json.dumps(r) + "\n"); log.flush(); print(json.dumps(r)[:400], flush=True)


if a.off or a.no_split:
    ag.SPLIT_K_OUT_TILES = 0
if a.off:
    _pb = tt._permute_back
    tt._permute_back = lambda g, inv: ttnn.permute(g, inv) if list(inv) == [1, 2, 0] else _pb(g, inv)

st = dict(n=0)
_join = ag._join_slices


def join(parts, shape):
    if st["n"] >= 2 and st["n"] <= 3:
        dev = parts[0][2].device()
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        out = _join(parts, shape)
        ttnn.synchronize_device(dev); ms = (time.perf_counter() - t0) * 1e3
        if ms > 0.5:
            emit(dict(ev="join", it=st["n"], ms=ms, shape=list(shape),
                      parts=[[list(s), list(e), str(g.dtype), list(g.shape)] for s, e, g in parts]))
        return out
    return _join(parts, shape)


ag._join_slices = join


def aiclk():
    try:
        return int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read().split()[0])
    except Exception:
        return None


def digest(r):
    h = hashlib.sha256()
    for x in (r if isinstance(r, (tuple, list)) else [r]):
        try:
            h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
        except Exception:
            h.update(repr(type(x)).encode())
    return h.hexdigest()[:16]


cls = bc2.design_model_class()
real = cls.sequence_gradients


def sequence_gradients(self, *args, **kw):
    st["n"] += 1
    t0, c0 = time.monotonic(), aiclk()
    r = real(self, *args, **kw)
    emit(dict(ev="iter", n=st["n"], wall_s=time.monotonic() - t0, aiclk=[c0, aiclk()], load=os.getloadavg()[0],
              digest=digest(r), split_k=dict(ag.SPLIT_K_STATS), off=a.off))
    if st["n"] >= a.iters:
        emit(dict(ev="done"))
        os._exit(0)
    return r


cls.sequence_gradients = sequence_gradients
project = a.out / f"campaign_s{a.seed}"
settings = cleaned_campaign_settings(read_settings(str(a.bc2 / "examples" / "pdl1.json"), parse_setting_overrides(
    [f"campaign_seed={a.seed}", "max_trajectories=1", f"project_folder={project}", f"binder_lengths=[{a.binder}]"])))
weights = {k: os.environ[v] for k, v in (("af2_weights", "JAPANFOLD_BC2_AF2_WEIGHTS"),) if os.environ.get(v)}
with bc2.campaign_predictor(**({"checkpoints": weights["af2_weights"]} if weights else {})):
    bc2.run_campaign(settings, str(project), trajectories_per_card=1, **weights)
os._exit(1)
