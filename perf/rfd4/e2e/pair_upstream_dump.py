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


def reseed_at_featurize(seed: int) -> None:
    """Seed the ambient streams at the entry of each example's featurization.

    Upstream's featurizer draws from whatever the process stream holds when the DataLoader
    reaches the example: `datapipes/pipeline.py` permutes the chain and molecule labels with
    `np.random.permutation`, and `materialize_internal_coords` rotates each internal-coordinate
    group with a torch draw. Both are training augmentations that also run at inference, and
    neither is a function of the design seed: in a worker the numpy seed is
    `_generate_state(base_seed, worker_id)` and `base_seed` comes off the main process's torch
    stream after the model constructor has drawn from it.

    tt-bio's `prepare_cif` seeds python, numpy and torch to the design seed at exactly this
    point. This makes upstream do the same, which is the featurization-side counterpart of
    `capture.py::reseed_at_sampler`. The DataLoader runs in-process so the seeding applies to
    the stream the transforms actually read.
    """
    import torch.utils.data as D

    class Seeded(D.Dataset):
        def __init__(self, inner):
            self.inner = inner

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, idx):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            return self.inner[idx]

    base_loader_init = D.DataLoader.__init__

    def loader_init(self, dataset, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs.pop("prefetch_factor", None)
        base_loader_init(self, Seeded(dataset), *args, **kwargs)

    D.DataLoader.__init__ = loader_init


def log_featurize_draws() -> list:
    """Record every `np.random.permutation` the featurizer takes, with the array it indexes.

    The chain and molecule labels are `permutation(n)[inverse_of_unique(annotation)]`, so both
    the draw and the annotation's sort order decide the label. The caller's frame carries the
    atom array, which is the only place the annotation can be read at that moment.
    """
    import inspect

    from rfproteina.datapipes import pipeline as P

    draws: list = []
    real = P.np.random.permutation

    def spy(n):
        result = real(n)
        frame = inspect.currentframe().f_back
        aa = frame.f_locals.get("aa")
        entry = {"n": int(n), "draw": [int(v) for v in np.atleast_1d(result)],
                 "line": frame.f_lineno}
        if aa is not None:
            for name in ("chain_iid", "molecule_iid"):
                if name in aa.get_annotation_categories():
                    v = aa.get_annotation(name)
                    entry[name] = {"unique": [str(x) for x in np.unique(v)],
                                   "first": str(v[0]), "last": str(v[-1])}
        draws.append(entry)
        return result

    P.np.random.permutation = spy
    return draws


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2, help="dump the state after this many steps")
    ap.add_argument("--out", default="/home/moritz/rfd4-pair-runs/upstream/dump")
    ap.add_argument("--ckpt", default="nvidia/RFD4-Proteina")
    ap.add_argument("--reseed-at-featurize", action="store_true",
                    help="seed python/numpy/torch to --seed at the entry of each example's "
                         "featurization, the way tt-bio's prepare_cif does, and run the "
                         "dataloader in-process so that is the stream the transforms draw from")
    ap.add_argument("--log-featurize-draws", action="store_true",
                    help="record the pipeline's chain/molecule permutation draws and the "
                         "annotations they index, which is where the labels are decided")
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
    if a.reseed_at_featurize:
        reseed_at_featurize(a.seed)
    if a.log_featurize_draws:
        blob["featurize_draws"] = log_featurize_draws()

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
