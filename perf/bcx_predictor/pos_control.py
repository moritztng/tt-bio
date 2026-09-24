"""A positive control: fold inputs whose answer is known, both arms, pLDDT split by chain.

Five trajectories rejected at pLDDT ~0.38 almost independently of n, which is the shape of
a path that has stopped carrying its input. This tests that directly and cheaply -- folds,
not trajectories:

  A  the PD-L1 target ALONE, a natural 115-residue protein with a known fold. Both arms
     must read high. If the device does not, the trunk or the confidence head is broken
     and no optimiser will ever clear a 0.60 filter.
  B  the seed-0 starting complex, pLDDT split into the binder and target residues, because
     the screen filter is bound to the BINDER role and a whole-complex mean is dominated by
     the template-conditioned target.

Bucket 32 throughout, the regime the trajectories ran. No af2.py mask site touched.
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

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
s = B.campaign_settings()
bucket = campaign_length_bucket(s)
_, states, _ = B.design_state(s)
target_only = {"hPDL1": {k: v for k, v in states["hPDL1"].items() if k != "binder"}}

def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                     max_cache_size=4, trunk="jax")

def split_plddt(pred, state="hPDL1"):
    sp = pred[state]
    plddt = np.asarray(sp.metrics["plddt"]).reshape(-1)
    out, i = {}, 0
    for name in sorted(sp.protein_complex):
        n = len(sp.protein_complex[name])
        out[name] = round(float(plddt[i:i + n].mean()), 4)
        i += n
    out["_all"] = round(float(plddt.mean()), 4)
    out["_ptm"] = round(float(sp.metrics["ptm"]), 4)
    return out

def both(states_in, tag, res):
    m = model()
    t0 = time.time(); res[tag + "_jax"] = split_plddt(m.predict(states_in))
    res[tag + "_jax_s"] = round(time.time() - t0, 1)
    print(tag, "jax", res[tag + "_jax"], flush=True)

out = {"bucket": bucket}
both(target_only, "A_target_alone", out)
both(states, "B_start_complex", out)

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock(); t0 = time.time()
with evoformer_on_device(evo):
    out["A_target_alone_device"] = split_plddt(model().predict(target_only))
    print("A device", out["A_target_alone_device"], flush=True)
    out["B_start_complex_device"] = split_plddt(model().predict(states))
    print("B device", out["B_start_complex_device"], flush=True)
t1 = time.time(); clock.stop()
out["device_s"] = round(t1 - t0, 1); out["aiclk"] = clock.window([(t0, t1)])
out["screen_bar"] = 0.60
out["trajectories_rejected_at_plddt"] = [0.41, 0.38, 0.38, 0.38, 0.38]
out["stamp"] = A.stamp(3)
(HERE / "pos_control.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "stamp"}, indent=1, default=str))
