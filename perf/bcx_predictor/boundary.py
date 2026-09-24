"""Compare the device Evoformer against BindCraft 2's own, at the splice boundary.

The loss agreed to 0.28% while the gradient came back at cos 0.093. A loss can be
insensitive; the activations the stack actually emits cannot. This captures BindCraft 2's
real (msa, pair) at modules.py:1594, runs both stacks on exactly those inputs, and
compares the outputs.
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, jax
import bc2_state as B
import afgrad as A, stack as S
from bindcraft.af.alphafold.model import layer_stack as LS, modules
from ttbio_predictor import TTBioAlphaFoldDesignModel

cap = {}
MASKS = {}
_real_iter = modules.EvoformerIteration.__call__

def _iter_spy(self, activations, masks, safe_key, use_dropout, **kw):
    if not self.is_extra_msa:
        MASKS.setdefault("msa", masks["msa"])
    return _real_iter(self, activations=activations, masks=masks, safe_key=safe_key,
                      use_dropout=use_dropout, **kw)

modules.EvoformerIteration.__call__ = _iter_spy
real = LS.layer_stack
def factory(num_layers, *a, **kw):
    made = real(num_layers, *a, **kw)
    def choose(fn):
        if getattr(fn, "__name__", None) == "evoformer_fn":
            inner = made(fn)
            def store(mi, pi, mo, po, mm):
                # Under jit these are tracers at trace time; jax.debug.callback fires at
                # RUN time with concrete arrays. Only the first call is kept: the recycle
                # pass and the real pass both come through here.
                if "msa_in" not in cap:
                    cap["msa_in"] = np.asarray(mi, dtype=np.float32)
                    cap["pair_in"] = np.asarray(pi, dtype=np.float32)
                    cap["msa_out"] = np.asarray(mo, dtype=np.float32)
                    cap["pair_out"] = np.asarray(po, dtype=np.float32)
                    cap["msa_mask"] = np.asarray(mm, dtype=np.float32)

            def spy(x):
                act, sk = x
                out, sk2 = inner(x)
                jax.debug.callback(store, act["msa"], act["pair"], out["msa"], out["pair"],
                                   MASKS["msa"])
                return out, sk2
            return spy
        return made(fn)
    return choose

modules.layer_stack.layer_stack = factory
try:
    s = B.campaign_settings(overrides=["length_bucket_size=1"])
    _, states, _ = B.design_state(s)
    TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir="/home/ttuser/bcx_e2e/af2_params",
                              models=("model_1_ptm",), num_recycle=1, key=jax.random.PRNGKey(0),
                              length_bucket_size=1, max_cache_size=2, trunk="jax").predict(states)
finally:
    modules.layer_stack.layer_stack = real
    modules.EvoformerIteration.__call__ = _real_iter

import torch
lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
m = torch.from_numpy(cap["msa_in"]).float(); z = torch.from_numpy(cap["pair_in"]).float()
mask_t = torch.from_numpy(cap["msa_mask"]).float()
MASKED = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True
mask_dev = dev.up(mask_t) if MASKED else None
mo, zo = dev.stack(dev.up(m), dev.up(z), 0, 48, ckpt=False, msa_mask=mask_dev)
dev.sync()
d_msa = dev.down(mo, tuple(m.shape)).numpy(); d_pair = dev.down(zo, tuple(z.shape)).numpy()

def cmp(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return {"rel_l2": float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)),
            "cos": float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)),
            "norm_bc2": float(np.linalg.norm(a)), "norm_device": float(np.linalg.norm(b))}

out = {"masked": MASKED, "msa_mask_row1_zero_fraction": float((cap["msa_mask"][1] == 0).mean()),
       "msa_in_shape": list(cap["msa_in"].shape), "pair_in_shape": list(cap["pair_in"].shape),
       "msa_out": cmp(cap["msa_out"], d_msa), "pair_out": cmp(cap["pair_out"], d_pair),
       "msa_row0_out": cmp(cap["msa_out"][0], d_msa[0]),
       "msa_row1_out": cmp(cap["msa_out"][1], d_msa[1]) if cap["msa_out"].shape[0] > 1 else None,
       "stamp": A.stamp(3)}
(HERE / f"boundary_check_masked{int(MASKED)}.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "stamp"}, indent=1, default=str))
