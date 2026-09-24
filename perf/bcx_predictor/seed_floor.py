"""The seed floor at this state: how far does the JAX arm move on its own?

A 24.9 A binder deviation means nothing without the variation the model already shows at
the same state. Both runs here are BindCraft 2's own trunk; only the PRNG key differs, and
the key drives dropout, which BindCraft 2 has ON for design (dropout=True at af2.py:209).
No device is opened.
"""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, jax
import bc2_state as B
from ttbio_predictor import TTBioAlphaFoldDesignModel

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
s = B.campaign_settings(overrides=["length_bucket_size=1"])
_, states, _ = B.design_state(s)

def model(key):
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1, key=key,
                                     length_bucket_size=1, max_cache_size=2, trunk="jax")

def ca(pred, chain):
    return np.asarray(pred["hPDL1"].protein_complex[chain].atoms)[:, 1, :].astype(np.float64)

def kabsch(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(-1).mean()))

t0 = time.time(); p0 = model(jax.random.PRNGKey(0)).predict(states)
p0b = model(jax.random.PRNGKey(0)).predict(states)
p1 = model(jax.random.PRNGKey(1)).predict(states)
out = {
  "same_key_binder_rmsd_A": float(np.sqrt(((ca(p0,"binder")-ca(p0b,"binder"))**2).sum(-1).mean())),
  "same_key_target_rmsd_A": float(np.sqrt(((ca(p0,"target_hPDL1")-ca(p0b,"target_hPDL1"))**2).sum(-1).mean())),
  "seed_floor_binder_unsuperposed_A": float(np.sqrt(((ca(p0,"binder")-ca(p1,"binder"))**2).sum(-1).mean())),
  "seed_floor_binder_kabsch_A": kabsch(ca(p0,"binder"), ca(p1,"binder")),
  "seed_floor_target_unsuperposed_A": float(np.sqrt(((ca(p0,"target_hPDL1")-ca(p1,"target_hPDL1"))**2).sum(-1).mean())),
  "plddt_key0": float(np.asarray(p0["hPDL1"].metrics["plddt"]).mean()),
  "plddt_key1": float(np.asarray(p1["hPDL1"].metrics["plddt"]).mean()),
  "device_binder_unsuperposed_A": 36.6509605844655,
  "device_binder_kabsch_A": 24.944468734975832,
  "device_target_unsuperposed_A": 0.106021473387675,
  "seconds": round(time.time()-t0, 1),
}
out["device_over_seed_floor_binder_kabsch"] = round(
    out["device_binder_kabsch_A"] / max(out["seed_floor_binder_kabsch_A"], 1e-9), 2)
(HERE / "seed_floor.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
