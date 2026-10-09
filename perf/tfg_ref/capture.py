"""Run `opendde pred` with TFG's hook points recorded, for parity fixtures.

Wraps, without changing what they compute or the RNG stream they draw from:
  TFGEngine.step            x_t in, x_{t-1} out, step_i            (every guided step)
  TFGEngine._project        denoiser x0 in, projection delta out   (every guided step)
  TFGEngine._logp_and_grad_x0  log p (= -energy) per refinement iteration
  epitope_guidance.guide_x0    early rigid pass on x0, in and out  (scheduled steps only)
  late pass on x_l: search_rigid_contact / refine_rigid_contact / search_epitope / refine_epitope, in and out
  (the same solvers called from inside guide_x0 are tagged x0_* instead of late_*)
plus the guidance input features (user_* and the TFG-only features) of each job.
Writes <FIXTURE_DIR>/<job name>.pt, one dict per job: {"feats": ..., "events": [(kind, step_i, {tensors})...]}.

usage: FIXTURE_DIR=/path python capture.py pred -i in.json -o out ...   (same arguments as `opendde`)
"""
import os
import sys
from pathlib import Path

import torch

import opendde.model.generator as gen
import opendde.tfg.engine as eng
import opendde.tfg.epitope_guidance as epi
import opendde.tfg.rigid_contact as rc
from runner.cli import opendde_cli

OUT = Path(os.environ["FIXTURE_DIR"])
OUT.mkdir(parents=True, exist_ok=True)
STATE = {"feats_id": None, "events": [], "feats": None, "name": None, "step": None}


def cpu(x):
    return x.detach().float().cpu().clone() if torch.is_tensor(x) else x


def flush():
    if STATE["name"] and STATE["events"]:
        torch.save({"feats": STATE["feats"], "events": STATE["events"]}, OUT / f"{STATE['name']}.pt")
    STATE.update(events=[], feats=None, name=None)


def note_feats(feats):
    if STATE["feats_id"] == id(feats):
        return
    flush()
    STATE["feats_id"] = id(feats)
    keep = getattr(eng, "_TFG_ONLY_FEATURES", ())
    STATE["feats"] = {k: cpu(v) for k, v in feats.items()
                      if torch.is_tensor(v) and (k.startswith("user_") or k in keep or k in (
                          "atom_to_token_idx", "ref_element", "ref_pos", "asym_id", "residue_index", "atom_name_chars",
                          "is_protein", "ref_mask"))}
    STATE["name"] = os.environ.get("FIXTURE_NAME_PREFIX", "") + f"job{len(list(OUT.glob('*.pt')))}"


def wrap_method(cls, name, record):
    orig = getattr(cls, name)

    def f(self, *a, **k):
        out = orig(self, *a, **k)
        record(a, k, out)
        return out

    setattr(cls, name, f)


def rec_step(a, k, out):
    note_feats(k["input_feature_dict"])
    STATE["step"] = k["step_i"]
    STATE["events"].append(("engine_step", k["step_i"], {"x_in": cpu(k["x"]), "x_out": cpu(out),
                                                         "t_hat": cpu(k["t_hat"]), "c_tau": cpu(k["c_tau"])}))


def rec_project(a, k, out):
    x0 = a[0] if a else k.get("x")
    STATE["events"].append(("project", k.get("step_i"), {"x0_denoised": cpu(x0), "delta": cpu(out)}))


def rec_logp(a, k, out):
    STATE["events"].append(("logp", k.get("step_i"), {"logp": cpu(out[0])}))


wrap_method(eng.TFGEngine, "step", rec_step)
wrap_method(eng.TFGEngine, "_project", rec_project)
wrap_method(eng.TFGEngine, "_logp_and_grad_x0", rec_logp)


def wrap_fn(mod, name, kind, step_arg=None):
    orig = getattr(mod, name)

    def f(*a, **k):
        outer = STATE.get("in_x0", False)
        STATE["in_x0"] = outer or kind == "guide_x0"
        try:
            out = orig(*a, **k)
        finally:
            STATE["in_x0"] = outer
        step = a[step_arg] if step_arg is not None else STATE["step"]
        tag = kind.replace("late_", "x0_") if outer else kind
        if kind != "guide_x0" or not torch.equal(a[0], out):
            STATE["events"].append((tag, step, {"in": cpu(a[0]), "out": cpu(out)}))
        return out

    setattr(mod, name, f)
    return f


wrap_fn(epi, "guide_x0", "guide_x0", step_arg=2)
for mod, name in ((epi, "search_epitope"), (epi, "refine_epitope")):
    wrap_fn(mod, name, "late_" + name)
for name in ("search_rigid_contact", "refine_rigid_contact"):
    f = wrap_fn(rc, name, "late_" + name)
    if hasattr(gen, name):
        setattr(gen, name, f)

try:
    opendde_cli(sys.argv[1:], standalone_mode=False)
finally:
    flush()
