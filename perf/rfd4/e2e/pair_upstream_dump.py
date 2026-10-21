"""Upstream side of the pairing bisect: dump the seams `capture.py` does not reach.

`capture.py`'s sampler hook stores the batch through `_detach`, which returns None for the
dataloader's sample object, so the batch that goes into the trunk never made it to disk. This
runs the same pinned upstream generation with the same reseed at `full_simulation`'s entry and
dumps, as plain tensors:

* ``x_0``   - the initial noise, right out of `fm.sample_noise`
* ``batch`` - every tensor field of the sample object at the first trunk call
* ``x_k``   - the state after simulation step k, for the first few steps
* ``ts``/``gt``/``recycle_depths`` - the schedule those steps ran on

    PROJECT_ROOT=<upstream src> /home/moritz/rfd4-ref-env/venv/bin/python \
        perf/rfd4/e2e/pair_upstream_dump.py --seed 0 --steps 2 --out <dir> <hydra overrides>
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch


class Enough(Exception):
    """Raised to end generation once the dumps are in."""


def _tensors(obj, depth=0):
    """Every tensor reachable from a batch-like object, keyed by path."""
    out = {}
    if isinstance(obj, torch.Tensor):
        return {"": obj.detach().cpu().clone()}
    if depth > 2:
        return out
    items = None
    if isinstance(obj, dict):
        items = obj.items()
    elif hasattr(obj, "keys"):
        try:
            items = [(k, obj[k]) for k in obj.keys()]
        except Exception:
            items = None
    elif hasattr(obj, "__dict__"):
        items = vars(obj).items()
    for k, v in items or []:
        for sub, t in _tensors(v, depth + 1).items():
            out[f"{k}{'.' + sub if sub else ''}"] = t
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2, help="dump the state after this many steps")
    ap.add_argument("--out", default="/home/moritz/rfd4-pair-runs/upstream/dump")
    ap.add_argument("--ckpt", default="nvidia/RFD4-Proteina")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(torch.float32)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    import rfd4_proteina as rfd4
    from capture import force_cpu_fp32_trainer  # same directory

    model = rfd4.Proteina.from_pretrained(a.ckpt)
    fm = model.fm
    blob: dict = {"seed": a.seed, "data_modes": list(fm.data_modes)}

    inner_sim, inner_noise, inner_step = fm.full_simulation, fm.sample_noise, fm.simulation_step
    inner_predict = model.predict_for_sampling
    state = {"step": 0}

    def full_simulation(*args, **kwargs):
        torch.manual_seed(a.seed)
        blob["rng_state_at_entry"] = torch.get_rng_state().clone()
        return inner_sim(*args, **kwargs)

    def sample_noise(*args, **kwargs):
        x = inner_noise(*args, **kwargs)
        blob["x_0"] = {k: v.detach().cpu().clone() for k, v in x.items()}
        blob["sample_noise_kwargs"] = {k: v for k, v in kwargs.items()
                                       if isinstance(v, (int, float, bool, str, type(None)))}
        blob["emit_masks"] = {k: v.detach().cpu().clone()
                              for k, v in (kwargs.get("emit_masks") or {}).items()}
        groups = kwargs.get("groups") or {}
        blob["groups"] = {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else None)
                          for k, v in groups.items()}
        ccm = kwargs.get("coord_cond_mask")
        blob["coord_cond_mask"] = None if ccm is None else ccm.detach().cpu().clone()
        return x

    def predict_for_sampling(batch, mode, recycle_depth):
        i = state["step"]
        if i == 0:
            blob["batch"] = _tensors(batch)
            blob["batch_repr"] = type(batch).__name__
        blob[f"recycle_depth{i}"] = int(recycle_depth)
        return inner_predict(batch, mode, recycle_depth)

    def simulation_step(*args, **kwargs):
        x = inner_step(*args, **kwargs)
        i = state["step"]
        blob[f"x_{i + 1}"] = {k: v.detach().cpu().clone() for k, v in x.items()}
        t = kwargs.get("t") or {}
        blob[f"t{i}"] = {k: float(v.reshape(-1)[0]) for k, v in t.items()}
        state["step"] = i + 1
        if state["step"] >= a.steps:
            torch.save(blob, out / "seams.pt")
            raise Enough
        return x

    fm.full_simulation, fm.sample_noise, fm.simulation_step = (
        full_simulation, sample_noise, simulation_step)
    model.predict_for_sampling = predict_for_sampling
    force_cpu_fp32_trainer()

    def caused_by_enough(exc):
        seen = set()
        while exc is not None and id(exc) not in seen:
            if isinstance(exc, Enough):
                return True
            seen.add(id(exc))
            exc = exc.__cause__ or exc.__context__
        return False

    try:
        rfd4.generate(model, overrides=list(a.overrides))
    except BaseException as exc:
        if not caused_by_enough(exc):
            raise
    if not (out / "seams.pt").exists():
        torch.save(blob, out / "seams.pt")
    print(f"[pair] wrote {out / 'seams.pt'}: {sorted(blob)}", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    raise SystemExit(main())
