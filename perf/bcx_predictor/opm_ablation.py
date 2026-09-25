"""What is the OUTER PRODUCT MEAN's mask worth on its own?

The attention masks landed and pair moved the wrong way, 0.181 -> 0.199 against a ~0.145
bf16 floor. Before writing a masked OPM in ttnn -- the class is shared with Boltz,
OpenFold3, Protenix, OpenDDE, RF3 and AF2-IG, so it cannot just be edited -- price the one
site. Three arms of identical torch code on BindCraft 2's own captured inputs, fp32:

  A  true mask everywhere                 what AF2 computes
  B  true mask in the attentions, ones in the OPM   what the device does today
  C  ones everywhere                      what the device did before this pass

B against A is the OPM mask's own worth. If it is small, the remaining pair gap is not the
OPM and building a masked one would be the wrong move.
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad")):
    sys.path.insert(0, p)
import numpy as np, torch, jax
import bc2_state as B
import afgrad as A
from bindcraft.af.alphafold.model import modules
from ttbio_predictor import TTBioAlphaFoldDesignModel

cap = {}
_real = modules.EvoformerIteration.__call__
def store(mi, pi, mm):
    cap.setdefault("msa", np.asarray(mi, np.float32))
    cap.setdefault("pair", np.asarray(pi, np.float32))
    cap.setdefault("msa_mask", np.asarray(mm, np.float32))
def spy(self, activations, masks, safe_key, use_dropout, **kw):
    if not self.is_extra_msa:
        jax.debug.callback(store, activations["msa"], activations["pair"], masks["msa"])
    return _real(self, activations=activations, masks=masks, safe_key=safe_key,
                 use_dropout=use_dropout, **kw)
modules.EvoformerIteration.__call__ = spy
try:
    s = B.campaign_settings(overrides=["length_bucket_size=1"])
    _, states, _ = B.design_state(s)
    TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir="/home/ttuser/bcx_e2e/af2_params",
                              models=("model_1_ptm",), num_recycle=1, key=jax.random.PRNGKey(0),
                              length_bucket_size=1, max_cache_size=2, trunk="jax").predict(states)
finally:
    modules.EvoformerIteration.__call__ = _real

_, ref = A.load_models(A.DEFAULT_PARAMS, device_arm=False)
m0 = torch.from_numpy(cap["msa"]).float(); z0 = torch.from_numpy(cap["pair"]).float()
mm = torch.from_numpy(cap["msa_mask"]).float()
ones = torch.ones_like(mm)
pm = torch.ones(z0.shape[0], z0.shape[0])

def stack(att_mask, opm_mask):
    m, z = m0.clone(), z0.clone()
    for i in range(48):
        blk = ref["f32"].evoformer[i]
        m = blk._msa_track(m, z, att_mask)
        z = z + blk.opm(m, opm_mask)
        z = blk._pair_track(z, pm)
    return m, z

def cmp(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return {"rel_l2": float((a-b).norm()/a.norm()), "cos": float(a@b/(a.norm()*b.norm()))}

with torch.no_grad():
    mA, zA = stack(mm, mm)         # AF2
    mB, zB = stack(mm, ones)       # device today
    mC, zC = stack(ones, ones)     # device before
out = {
  "B_vs_A_opm_mask_alone": {"pair": cmp(zA, zB), "msa_row1": cmp(mA[1], mB[1]),
                            "msa_row0": cmp(mA[0], mB[0])},
  "C_vs_A_all_three_sites": {"pair": cmp(zA, zC), "msa_row1": cmp(mA[1], mC[1]),
                             "msa_row0": cmp(mA[0], mC[0])},
  "device_measured_today_vs_bc2": {"pair": 0.19930998100162226, "msa_row1": 0.16031306535643963},
  "device_measured_before_vs_bc2": {"pair": 0.18134394600449874, "msa_row1": 0.5417278626362723},
}
(HERE / "opm_ablation.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
