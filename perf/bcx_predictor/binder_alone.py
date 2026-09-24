"""Is it the single chain or the size? Fold the BINDER alone, as the gate path does.

The target alone (115) failed: device pLDDT 0.534 against BindCraft 2's 0.950. The
211-residue complex agreed. This folds the binder alone through BindCraft 2's OWN
binder_alone_state -- the exact call trajectory.py:71 makes to produce the
Unbound_Binder_pLDDT the screen gate reads -- so the answer applies to the gate and not to
an analogy. A second single-chain size is included to separate chain count from n.
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
from bindcraft.af2 import campaign_length_bucket
from bindcraft.trajectory import binder_alone_state, primary_target_state

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
s = B.campaign_settings()
bucket = campaign_length_bucket(s)
ds, states, _ = B.design_state(s)
alone = binder_alone_state(states, primary_target_state(states), ds.target_chain_prefix)

def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                     max_cache_size=4, trunk="jax")

def summarise(pred):
    o = {}
    for st, sp in pred.items():
        pl = np.asarray(sp.metrics["plddt"]).reshape(-1)
        o[st] = {"chains": {k: len(v) for k, v in sp.protein_complex.items()},
                 "n": int(pl.shape[0]), "plddt": round(float(pl.mean()), 4),
                 "ptm": round(float(sp.metrics["ptm"]), 4)}
    return o

out = {"bucket": bucket, "state_chains": {k: sorted(v) for k, v in alone.items()}}
t0 = time.time(); out["jax"] = summarise(model().predict(alone))
out["jax_s"] = round(time.time()-t0, 1)
print("jax:", json.dumps(out["jax"]), flush=True)

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock(); t0 = time.time()
with evoformer_on_device(evo):
    out["device"] = summarise(model().predict(alone))
t1 = time.time(); clock.stop()
out["device_s"] = round(t1-t0, 1); out["aiclk"] = clock.window([(t0, t1)])
print("device:", json.dumps(out["device"]), flush=True)
for st in out["jax"]:
    j, d = out["jax"][st], out["device"][st]
    out.setdefault("ratio", {})[st] = {"plddt": round(d["plddt"]/j["plddt"], 4),
                                       "ptm": round(d["ptm"]/j["ptm"], 4), "n": j["n"]}
out["target_alone_reference"] = {"jax_plddt": 0.9501, "device_plddt": 0.5340, "n": 115}
out["complex_reference"] = {"jax_plddt": 0.7349, "device_plddt": 0.7465, "n": 211}
out["stamp"] = A.stamp(3)
(HERE / "binder_alone.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out.get("ratio"), indent=1))
