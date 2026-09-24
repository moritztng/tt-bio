"""Grade the device BACKWARD at the splice seam, the way the forward was graded.

The full-loop gradient is 7.0x small on the binder and 5.6x on the target with a forward
that is good at the same seam, so the question is whether tt-bio's backward over 48 blocks
carries. This feeds both arms BindCraft 2's own (msa, pair) and mask, seeds the same
cotangent, and compares the leaf gradients against a float64 torch reference of the same
stack. No BindCraft 2 loss is involved, so nothing downstream can hide or cause it.
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch, jax
import bc2_state as B
import afgrad as A, stack as S
from bindcraft.af.alphafold.model import modules
from ttbio_predictor import TTBioAlphaFoldDesignModel

cap = {}
_real = modules.EvoformerIteration.__call__
def store(mi, pi, mm):
    cap.setdefault("msa", np.asarray(mi, np.float32))
    cap.setdefault("pair", np.asarray(pi, np.float32))
    cap.setdefault("mask", np.asarray(mm, np.float32))
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

m_np, z_np, mask_np = cap["msa"], cap["pair"], cap["mask"]
rng = np.random.default_rng(0)
gm_np = rng.standard_normal(m_np.shape).astype(np.float32)
gz_np = rng.standard_normal(z_np.shape).astype(np.float32)

lv = S.Levers(); dm, ref = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
mask_dev = dev.up(torch.from_numpy(mask_np).float())
ml = dev.leaf(torch.from_numpy(m_np).float()); zl = dev.leaf(torch.from_numpy(z_np).float())
with dev.tt.tape():
    mo, zo = dev.stack(ml, zl, 0, 48, ckpt=True, msa_mask=mask_dev)
dev.sync()
reach = A.node_census(dev.ag, [mo, zo])
dev.ag.backward([mo, zo], [dev.seed(torch.from_numpy(gm_np), mo),
                           dev.seed(torch.from_numpy(gz_np), zo)])
dev.sync()
d_gm = dev.grad(ml, m_np.shape); d_gz = dev.grad(zl, z_np.shape)
dev.ag.release_pins()

mr = torch.from_numpy(m_np).double().requires_grad_(True)
zr = torch.from_numpy(z_np).double().requires_grad_(True)
mk = torch.from_numpy(mask_np).double()
pm = torch.ones(z_np.shape[0], z_np.shape[0]).double()
m, z = mr, zr
for i in range(48):
    m, z = ref["f64"].evoformer[i](m, z, mk, pm)
torch.autograd.backward([m, z], [torch.from_numpy(gm_np).double(),
                                 torch.from_numpy(gz_np).double()])

def cmp(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return {"rel_l2": float((a-b).norm()/b.norm()), "cos": float(a@b/(a.norm()*b.norm())),
            "norm_device": float(a.norm()), "norm_f64": float(b.norm()),
            "norm_ratio": float(a.norm()/b.norm())}

out = {"tape_reach_nodes": reach, "msa_shape": list(m_np.shape),
       "d_msa": cmp(d_gm, mr.grad), "d_pair": cmp(d_gz, zr.grad), "stamp": A.stamp(3)}
(HERE / "bwd_boundary.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "stamp"}, indent=1, default=str))
