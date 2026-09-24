"""Where does the loop exit? Log every round's loss terms, predictions and logits.

trajectory.py:132-136 has TWO finiteness checks -- the gradient's scalar, then a recompute
from the PREDICTIONS -- and the break needs both to fail. So the culprit is in the
predictions dict, not a scalar that went bad on the way out. device_calls backward 2 with
one round recorded says fold 2's predictions are the ones that came back non-finite, i.e.
the state after exactly ONE optimiser update.

This wraps sequence_gradients and records, per call: finiteness of every gradient array,
of every metric and atom array in the predictions, the design loss, and the input logits.
Nothing in the loop is modified -- the wrapper observes and forwards.
"""
import json, os, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, jax
import bc2_state as B
import afgrad as A, stack as S
import bindcraft.campaign as campaign
from bindcraft.af2 import campaign_length_bucket
from bindcraft.settings import parse_setting_overrides, read_settings
from bindcraft.preflight import cleaned_campaign_settings
from splice import EvoformerOnDevice, evoformer_on_device, NANLOG
import ttbio_predictor as T

LOG = []

def finite_report(obj, tag):
    bad = {}
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif hasattr(o, "shape") or isinstance(o, (int, float)):
            a = np.asarray(o, dtype=np.float64)
            if a.size and not np.isfinite(a).all():
                bad[path] = {"nan": int(np.isnan(a).sum()), "inf": int(np.isinf(a).sum()),
                             "size": int(a.size)}
    walk(obj, tag)
    return bad

_real = T.TTBioAlphaFoldDesignModel.sequence_gradients

def traced(self, protein_states, losses, *a, **kw):
    call = {"round": len(LOG) + 1}
    seqs = {n: np.asarray(p.sequence, np.float64)
            for st in protein_states.values() for n, p in st.items()}
    call["input_logits_bad"] = {n: {"nan": int(np.isnan(v).sum()), "inf": int(np.isinf(v).sum())}
                                for n, v in seqs.items()
                                if not np.isfinite(v).all()}
    call["input_logits_absmax"] = {n: float(np.abs(v).max()) for n, v in seqs.items()}
    preds, grads, loss = _real(self, protein_states, losses, *a, **kw)
    call["design_loss"] = float(loss)
    call["design_loss_finite"] = bool(np.isfinite(float(loss)))
    call["grad_bad"] = {k: {"nan": int(np.isnan(np.asarray(v)).sum()),
                            "inf": int(np.isinf(np.asarray(v)).sum())}
                        for k, v in grads.items() if not np.isfinite(np.asarray(v)).all()}
    pb = {}
    for st, sp in preds.items():
        pb.update(finite_report(dict(sp.metrics), f"{st}.metrics"))
        for cn, pr in sp.protein_complex.items():
            pb.update(finite_report({"atoms": pr.atoms, "atom_mask": pr.atom_mask},
                                    f"{st}.{cn}"))
    call["predictions_bad"] = pb
    LOG.append(call)
    print(f"round {call['round']}: loss={call['design_loss']} finite={call['design_loss_finite']} "
          f"grad_bad={list(call['grad_bad'])} pred_bad={list(pb)[:4]}", flush=True)
    return preds, grads, loss

T.TTBioAlphaFoldDesignModel.sequence_gradients = traced

BUCKET = os.environ.get("BCX_BUCKET", "")
OUT = HERE / "runs" / ("trace_seam_seed0_b" + (BUCKET or "32"))
OUT.mkdir(parents=True, exist_ok=True)
ov = [f"campaign_seed=0", "max_trajectories=1", "validation_model=monomer",
      'design_models=["model_1_ptm"]', 'validation_models=["model_2_ptm"]',
      "autotune=false", "compile_next_length=false",
      *([("length_bucket_size=" + BUCKET)] if BUCKET else []),
      "save_failed_trajectories=true", "save_design_sequences=true",
      f"project_folder={OUT}"]
settings = cleaned_campaign_settings(read_settings(
    os.path.join(B.BC2, "examples", "pdl1.json"), parse_setting_overrides(ov)))
campaign.MULTIMER_POOL = T.__dict__.get("MONOMER", ("model_1_ptm", "model_2_ptm"))
campaign.AlphaFoldDesignModel = lambda *aa, **kk: T.TTBioAlphaFoldDesignModel(
    *aa, trunk="device", **kk)

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
mpnn = os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn", "weights_neutral")
try:
    with evoformer_on_device(evo):
        campaign.run_campaign(settings, str(OUT),
                              af2_weights="/home/ttuser/bcx_e2e/af2_params",
                              mpnn_weights=mpnn, max_trajectories=1)
finally:
    blob = {"bucket": campaign_length_bucket(settings), "rounds": LOG,
            "seam": NANLOG,
            "first_seam_bad": next((i for i, e in enumerate(NANLOG)
                                    if any(e.get(k) for k in e if k not in ("op", "call"))), None),
            "device_calls": dict(evo.calls),
            "first_bad_round": next((c["round"] for c in LOG
                                     if c["predictions_bad"] or c["grad_bad"]
                                     or not c["design_loss_finite"]), None)}
    (HERE / ("trace_seam_b" + (BUCKET or "32") + ".json")).write_text(json.dumps(blob, indent=1, default=str))
    print(json.dumps({"rounds": len(LOG), "first_bad_round": blob["first_bad_round"],
                      "device_calls": blob["device_calls"]}, indent=1))
