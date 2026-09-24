"""Is our pLDDT BindCraft 2's pLDDT in the regime the trajectories actually ran?

splice_check.json compared the two arms at bucket 1, n=192, mask all ones. Every completed
trajectory ran at BindCraft 2's default bucket 32: n=211 with 19 pad residues masked. That
is a different regime and the earlier control says nothing about it, so this repeats the
comparison there -- same state, same class, same code, the only difference being which
Evoformer ran.

No af2.py mask site is touched: bcx-mask owns those now.
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
from bindcraft.af2 import campaign_length_bucket, pad_design_chains, protein_state_shapes
from bindcraft.protein import real_residue_weights
from bindcraft.af2 import concatenate_chain_arrays

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
s = B.campaign_settings()                      # no override: BindCraft 2's own default 32
bucket = campaign_length_bucket(s)
_, states, _ = B.design_state(s)
padded = pad_design_chains(states, bucket, 0)
shp = protein_state_shapes(padded)[0]
flags = concatenate_chain_arrays(shp[1], padded[shp[0]], "flags")["flags"]
mask = np.asarray(real_residue_weights(flags))

def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                     max_cache_size=2, trunk="jax")

def metrics(pred):
    m = pred["hPDL1"].metrics
    return {"plddt_mean": float(np.asarray(m["plddt"]).mean()),
            "plddt_min": float(np.asarray(m["plddt"]).min()),
            "plddt_max": float(np.asarray(m["plddt"]).max()),
            "ptm": float(m["ptm"]), "iptm": float(m["iptm"])}

out = {"bucket": bucket, "n": int(sum(shp[2])), "chains": list(shp[2]),
       "masked_residues": int((mask == 0).sum()),
       "mask_all_ones": bool((mask == 1).all())}
t0 = time.time(); out["jax"] = metrics(model().predict(states)); out["jax_s"] = round(time.time()-t0, 1)
print("jax:", json.dumps(out["jax"]), flush=True)

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock(); t0 = time.time()
with evoformer_on_device(evo):
    out["device"] = metrics(model().predict(states))
t1 = time.time(); clock.stop()
out["device_s"] = round(t1-t0, 1); out["aiclk"] = clock.window([(t0, t1)])
out["plddt_ratio_device_over_jax"] = round(out["device"]["plddt_mean"] / out["jax"]["plddt_mean"], 4)
out["screen_bar_min_plddt"] = 0.60
out["trajectories_rejected_at"] = [0.41, 0.38, 0.38, 0.38, 0.38]
out["stamp"] = A.stamp(3)
(HERE / "metric_masked.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "stamp"}, indent=1, default=str))
