"""Which of the three MSA-mask sites costs the error, and how much?

tt-bio's trunk applies no MSA mask at all. Before writing three masked ops in ttnn, this
prices each site: the torch reference already takes a real mask
(af2_reference.py:378 forward(msa, pair, msa_mask, pair_mask)), so both arms run the same
code and only the mask differs. fp32 throughout -- the comparison is mask against mask, so
precision cancels and float64 would cost ten minutes for nothing.

Inputs are BindCraft 2's own, captured at modules.py:1594.
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad")):
    sys.path.insert(0, p)
import numpy as np, torch, jax
import bc2_state as B
import afgrad as A
from bindcraft.af.alphafold.model import layer_stack as LS, modules
from ttbio_predictor import TTBioAlphaFoldDesignModel

cap = {}

# EvoformerIteration receives BOTH the activations and the masks, and it is the body of a
# jax.lax.scan: traced once, run 48 times, so jax.debug.callback fires with concrete arrays
# and the FIRST firing is the stack's input. Spying here rather than on layer_stack avoids
# the ordering trap that the masks dict is built after the stack is constructed.
real_iter = modules.EvoformerIteration.__call__

def store(mi, pi, mm, pm):
    cap.setdefault("msa", np.asarray(mi, np.float32))
    cap.setdefault("pair", np.asarray(pi, np.float32))
    cap.setdefault("msa_mask", np.asarray(mm, np.float32))
    cap.setdefault("pair_mask", np.asarray(pm, np.float32))

def iter_spy(self, activations, masks, safe_key, use_dropout, **kw):
    if not self.is_extra_msa:
        jax.debug.callback(store, activations["msa"], activations["pair"],
                           masks["msa"], masks["pair"])
    return real_iter(self, activations=activations, masks=masks, safe_key=safe_key,
                     use_dropout=use_dropout, **kw)

modules.EvoformerIteration.__call__ = iter_spy
try:
    s_set = B.campaign_settings(overrides=["length_bucket_size=1"])
    _, states, _ = B.design_state(s_set)
    TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir="/home/ttuser/bcx_e2e/af2_params",
                              models=("model_1_ptm",), num_recycle=1, key=jax.random.PRNGKey(0),
                              length_bucket_size=1, max_cache_size=2, trunk="jax").predict(states)
finally:
    modules.EvoformerIteration.__call__ = real_iter

_, ref = A.load_models(A.DEFAULT_PARAMS, device_arm=False)
m0 = torch.from_numpy(cap["msa"]).float(); z0 = torch.from_numpy(cap["pair"]).float()
mm = torch.from_numpy(cap["msa_mask"]).float(); pm = torch.from_numpy(cap["pair_mask"]).float()
ones_m = torch.ones_like(mm)

def stack(msa_mask, sites=None):
    m, z = m0.clone(), z0.clone()
    for i in range(48):
        blk = ref["f32"].evoformer[i]
        use = msa_mask if sites is None else msa_mask
        m, z = blk(m, z, use, pm)
    return m, z

def cmp(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return {"rel_l2": float((a-b).norm()/a.norm()), "cos": float(a@b/(a.norm()*b.norm()))}

out = {"msa_shape": list(cap["msa"].shape), "msa_mask_shape": list(cap["msa_mask"].shape),
       "msa_mask_row0_ones": bool((cap["msa_mask"][0] == 1).all()),
       "msa_mask_row1_ones": bool((cap["msa_mask"][1] == 1).all()),
       "msa_mask_row1_zero_fraction": float((cap["msa_mask"][1] == 0).mean()),
       "pair_mask_all_ones": bool((cap["pair_mask"] == 1).all())}
with torch.no_grad():
    m_true, z_true = stack(mm)
    m_ones, z_ones = stack(ones_m)
out["all_ones_vs_true_mask"] = {"msa": cmp(m_true, m_ones), "pair": cmp(z_true, z_ones),
                                "msa_row0": cmp(m_true[0], m_ones[0]),
                                "msa_row1": cmp(m_true[1], m_ones[1])}
(HERE / "mask_ablation.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
