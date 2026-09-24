"""Is the residue numbering the CARRIER of the gap between the two instruments?

`relpos_fix.py` tried to answer this by patching the offsets and re-reading. It tested
nothing: the patched legs came back bit-identical to the unpatched ones and in 84.3 s against
546.2 s, which is the jit compile cache being reused. `TTBioAlphaFoldDesignModel` keeps
`gradient_compile_cache` and `prediction_compile_cache` per instance and neither key mentions
the offsets, so the already-traced function was handed back with the old numbering baked in.
A patch applied after the trace cannot move the answer.

So this drives the test from the cheap instrument and forces a retrace by building a FRESH
model per leg:

  L1  predict, shipped                  -> the chain break is applied (af2.py:305-306)
  L2  predict, chain break made identity -> the numbering the gradient path passes

If L2 lands on the gradient path's own reading (i_pTM 0.2843, pLDDT 0.4162, measured on this
card) then the numbering carries the whole instrument gap. If it stays on L1, something else
in the gradient path does and the numbering is a bystander.

L1 is also the fresh-process half of CARRIED: this process opens the card and its first model
call is that read.
"""
import json, os, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_predictor"),
          str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)

import numpy as np, jax
import bc2_state as B
import afgrad as A, stack as S
from splice import EvoformerOnDevice, evoformer_on_device
from ttbio_predictor import TTBioAlphaFoldDesignModel
from bindcraft import af2 as BAF2

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
SP = dict(softmax_weight=1.0, one_hot_weight=1.0, temperature=0.01, logit_scale=2.0)

_real_break = BAF2.monomer_chain_break_indices
BREAK = {"on": True}


def _maybe_break(chain_lengths, residue_index):
    return _real_break(chain_lengths, residue_index) if BREAK["on"] else residue_index


BAF2.monomer_chain_break_indices = _maybe_break

s = B.campaign_settings()
_ds, states, _losses = B.design_state(s)
n_res = len(states["hPDL1"]["binder"])


def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=32,
                                     max_cache_size=6, dropout=False, trunk="jax")


def read(pred):
    sp = pred["hPDL1"]
    plddt = np.asarray(sp.metrics["plddt"]).reshape(-1)
    ptm, iptm = float(np.asarray(sp.metrics["ptm"])), float(np.asarray(sp.metrics["iptm"]))
    return {"n": int(plddt.shape[0]), "ptm": round(ptm, 6), "iptm": round(iptm, 6),
            "binder_plddt": round(float(plddt[:n_res].mean()), 4),
            "plddt": round(float(plddt.mean()), 4)}


out = {"why": "does the monomer chain break carry the whole gap between the two instruments",
       "against": {"grad_unpatched_on_this_card": {"iptm": 0.284293, "ptm": 0.452477,
                                                   "plddt": 0.4162, "binder_plddt": 0.3448},
                   "predict_shipped_on_this_card": {"iptm": 0.084367, "ptm": 0.569915,
                                                    "plddt": 0.7172, "binder_plddt": 0.4005}},
       "invalidates": "relpos_fix.json grad_patched/predict_patched -- compile cache reuse, "
                      "bit-identical at 84.3 s against 546.2 s",
       "result": {}}

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock(); w0 = time.time()
with evoformer_on_device(evo):
    for tag, on in (("predict_break_on_fresh_process", True), ("predict_break_off", False)):
        BREAK["on"] = on
        m = model()                      # fresh instance -> empty compile caches -> retrace
        t0 = time.time()
        res = read(m.predict(states, **SP))
        res["s"] = round(time.time() - t0, 1)
        res["chain_break_applied"] = on
        out["result"][tag] = res
        print(tag, json.dumps(res), flush=True)
        (HERE / "relpos_carrier.json").write_text(json.dumps(out, indent=1, default=str))
w1 = time.time(); clock.stop()
out["aiclk_during"] = clock.window([(w0, w1)])
out["device_calls"] = dict(evo.calls)
out["sysfs_node"] = list(S.sysfs_node())
out["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]))

on_, off = out["result"]["predict_break_on_fresh_process"], out["result"]["predict_break_off"]
grad = out["against"]["grad_unpatched_on_this_card"]
out["verdict"] = {
    "break_off_minus_break_on": {k: round(off[k] - on_[k], 6)
                                 for k in ("iptm", "ptm", "plddt", "binder_plddt")},
    "break_off_minus_gradient_instrument": {k: round(off[k] - grad[k], 6)
                                            for k in ("iptm", "ptm", "plddt", "binder_plddt")},
    "retrace_happened": off["s"] > 3 * on_["s"] or on_["s"] > 60,
}
print(json.dumps(out["verdict"], indent=1), flush=True)
print("AICLK", json.dumps(out["aiclk_during"], default=str), "calls", out["device_calls"], flush=True)
(HERE / "relpos_carrier.json").write_text(json.dumps(out, indent=1, default=str))
