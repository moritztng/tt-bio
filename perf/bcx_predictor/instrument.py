"""Do the gradient stage's metrics and `predict`'s metrics agree on the same sequence?

post_seed100 read i_pTM 0.74 at harden round 5 and 0.11 at mutate round 1. Those two
numbers come from different instruments: the gradient stages read the predictions
`sequence_gradients` returns (trajectory.py:130), at the optimiser's own softmax_weight,
one_hot_weight, temperature and logit_scale and with design_dropout True; mutate reads
`predict`, at the defaults 1.0 / 1.0 / 0.01 / 2.0 with dropout False.

`mutate_predict.json` has already shown that `predict` itself agrees with BindCraft 2's own
JAX to 0.002 i_pTM, padded or not. So this asks the other half: hold the state and the
sequence parameters fixed, turn dropout off on both, and read i_pTM through each
instrument, on each arm.

  same value both instruments  -> the harden-to-mutate step is the sequence parameters or
                                  the state, not the port, and the device tracks BindCraft
                                  2 through both.
  instruments disagree         -> the stage grades and the mutate grade are not on the same
                                  scale, and whichever arm disagrees more owns it.
"""
import json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)

import numpy as np, jax
import bc2_state as B
import afgrad as A, stack as S
from splice import EvoformerOnDevice, evoformer_on_device
from ttbio_predictor import TTBioAlphaFoldDesignModel
from bindcraft.target_schedule import losses_for_active_states

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
# The parameters `predict` uses by default, which is what the mutate stage calls it with.
SP = dict(softmax_weight=1.0, one_hot_weight=1.0, temperature=0.01, logit_scale=2.0)

s = B.campaign_settings()
_ds, states, losses = B.design_state(s)
active_losses = losses_for_active_states(losses, states)


def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=32,
                                     max_cache_size=4, dropout=False, trunk="jax")


def read(pred):
    sp = pred["hPDL1"]
    plddt = np.asarray(sp.metrics["plddt"]).reshape(-1)
    return {"n": int(plddt.shape[0]),
            "binder_plddt": round(float(plddt[:77].mean()), 4),
            "plddt": round(float(plddt.mean()), 4),
            "ptm": round(float(np.asarray(sp.metrics["ptm"])), 4),
            "iptm": round(float(np.asarray(sp.metrics["iptm"])), 4)}


out = {"why": "harden round 5 read i_pTM 0.74 through sequence_gradients; mutate round 1 read 0.11 through predict",
       "sequence_parameters": SP, "dropout": False, "num_recycle": 1,
       "losses": sorted(active_losses), "result": {}}
print(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=1), flush=True)


def both_instruments(tag, res):
    m = model()
    t0 = time.time()
    preds, _grads, loss = m.sequence_gradients(states, active_losses, **SP)
    res["sequence_gradients"] = read(preds)
    res["sequence_gradients"]["design_loss"] = round(float(loss), 6)
    res["sequence_gradients_s"] = round(time.time() - t0, 1)
    print(tag, "sequence_gradients", json.dumps(res["sequence_gradients"]),
          res["sequence_gradients_s"], "s", flush=True)
    t0 = time.time()
    res["predict"] = read(m.predict(states, **SP))
    res["predict_s"] = round(time.time() - t0, 1)
    print(tag, "predict", json.dumps(res["predict"]), res["predict_s"], "s", flush=True)


out["result"]["jax"] = {}
both_instruments("JAX", out["result"]["jax"])
(HERE / "instrument.json").write_text(json.dumps(out, indent=1, default=str))

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock(); w0 = time.time()
out["result"]["device"] = {}
with evoformer_on_device(evo):
    both_instruments("DEV", out["result"]["device"])
w1 = time.time(); clock.stop()
out["aiclk"] = clock.window([(w0, w1)])
out["device_calls"] = dict(evo.calls)

for arm in ("jax", "device"):
    r = out["result"][arm]
    out.setdefault("instrument_gap", {})[arm] = {
        m: round(r["sequence_gradients"][m] - r["predict"][m], 4)
        for m in ("iptm", "ptm", "plddt", "binder_plddt")}
for m in ("iptm", "ptm", "plddt", "binder_plddt"):
    out.setdefault("arm_gap", {})[m] = {
        inst: round(out["result"]["device"][inst][m] - out["result"]["jax"][inst][m], 4)
        for inst in ("sequence_gradients", "predict")}
out["being_explained"] = {"harden_round_5_iptm": 0.74, "mutate_round_1_iptm": 0.11}
out["stamp"] = A.stamp(3)
(HERE / "instrument.json").write_text(json.dumps(out, indent=1, default=str))
print("instrument_gap", json.dumps(out["instrument_gap"], indent=1), flush=True)
print("arm_gap", json.dumps(out["arm_gap"], indent=1), flush=True)
print("AICLK", json.dumps(out["aiclk"], default=str), flush=True)
