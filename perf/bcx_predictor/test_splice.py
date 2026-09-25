"""Does BindCraft 2's own predict run with tt-bio's Evoformer inside it?

Graded against the same class on BindCraft 2's trunk -- same call path, same bucket, same
seed -- so the only difference is where the 48 blocks ran. The bar is structural agreement,
not bit-exactness: quoted in Angstrom of C-alpha RMSD on the designed binder.
"""
import json, os, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np
import bc2_state as B
import afgrad as A, stack as S
from splice import EvoformerOnDevice, evoformer_on_device
from ttbio_predictor import TTBioAlphaFoldDesignModel

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
s = B.campaign_settings(overrides=["length_bucket_size=1"])
_, states, _ = B.design_state(s)

def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     length_bucket_size=1, max_cache_size=2, trunk="jax")

def ca(pred, chain="binder"):
    p = pred["hPDL1"].protein_complex[chain]
    return np.asarray(p.atoms)[:, 1, :].astype(np.float64)   # atom 1 is C-alpha in atom37


def kabsch_rmsd(a, b):
    """Internal RMSD after optimal superposition: does the binder FOLD the same, as
    distinct from being docked on a different face of the target?"""
    a = a - a.mean(0); b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(-1).mean()))

out = {}
t0 = time.time(); jax_pred = model().predict(states); out["jax_predict_s"] = round(time.time()-t0, 1)

lv = S.Levers(); dm, _ref = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock()
t0 = time.time()
with evoformer_on_device(evo) as swapped:
    dev_pred = model().predict(states)
t1 = time.time()
out["device_predict_s"] = round(t1-t0, 1)
clock.stop()

a, b = ca(jax_pred), ca(dev_pred)
out.update({
  "stacks_swapped": swapped,
  "device_calls": evo.calls,
  "live_tapes": EvoformerOnDevice.live_tapes(),
  "binder_residues": int(a.shape[0]),
  "binder_rmsd_unsuperposed_A": float(np.sqrt(((a - b) ** 2).sum(-1).mean())),
  "binder_rmsd_kabsch_A": kabsch_rmsd(a, b),
  "binder_max_dev_unsuperposed_A": float(np.sqrt(((a - b) ** 2).sum(-1)).max()),
  "target_rmsd_unsuperposed_A": float(np.sqrt(
      ((ca(jax_pred, "target_hPDL1") - ca(dev_pred, "target_hPDL1")) ** 2).sum(-1).mean())),
  "target_rmsd_kabsch_A": kabsch_rmsd(ca(jax_pred, "target_hPDL1"),
                                      ca(dev_pred, "target_hPDL1")),
  "plddt_jax": float(np.asarray(jax_pred["hPDL1"].metrics["plddt"]).mean()),
  "plddt_device": float(np.asarray(dev_pred["hPDL1"].metrics["plddt"]).mean()),
  "ptm_jax": float(jax_pred["hPDL1"].metrics["ptm"]),
  "ptm_device": float(dev_pred["hPDL1"].metrics["ptm"]),
  "device_atoms_finite": bool(np.isfinite(b).all()),
  "aiclk": clock.window([(t0, t1)]),
  "stamp": A.stamp(3),
})
(HERE / "splice_check.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "stamp"}, indent=1, default=str))
