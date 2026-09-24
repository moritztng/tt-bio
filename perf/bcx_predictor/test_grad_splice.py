"""Does BindCraft 2's sequence_gradients run with tt-bio's Evoformer inside it?

This is the half of the Protocol the design loop actually turns on: trajectory.py:131 calls
it 125 times a trajectory. Graded against the same class on BindCraft 2's trunk, same key,
same bucket, same state -- and after BindCraft 2's OWN normalisation, because
sequence_optimization.py:90 rescales to sqrt(designed_residue_count) and only DIRECTION
reaches the logits (bcx-e2e measured every arm's update norm at 1.0000001x the reference).
"""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, jax, jax.numpy as jnp
import bc2_state as B
import afgrad as A, stack as S
from splice import EvoformerOnDevice, evoformer_on_device
from ttbio_predictor import TTBioAlphaFoldDesignModel
from bindcraft.sequence_optimization import normalize_sequence_gradient

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
s = B.campaign_settings(overrides=["length_bucket_size=1"])
_, states, losses = B.design_state(s)

DROPOUT = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True

def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=1,
                                     max_cache_size=2, trunk="jax", dropout=DROPOUT)

def run(m):
    t0 = time.time()
    preds, grads, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                              one_hot_weight=0.0, temperature=1.0,
                                              logit_scale=2.0)
    return preds, {k: np.asarray(v) for k, v in grads.items()}, float(loss), time.time() - t0

out = {"design_dropout": DROPOUT}
_, g_jax, l_jax, t_jax = run(model())
out["jax"] = {"loss": l_jax, "seconds": round(t_jax, 1),
              "grad_shapes": {k: list(v.shape) for k, v in g_jax.items()}}
print("jax arm:", json.dumps(out["jax"]), flush=True)

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock(); t0 = time.time()
with evoformer_on_device(evo) as swapped:
    _, g_dev, l_dev, t_dev = run(model())
t1 = time.time(); clock.stop()
out["device"] = {"loss": l_dev, "seconds": round(t_dev, 1), "stacks_swapped": swapped,
                 "calls": dict(evo.calls), "live_tapes": EvoformerOnDevice.live_tapes(),
                 "aiclk": clock.window([(t0, t1)])}
print("device arm:", json.dumps(out["device"]), flush=True)

# The raw gradient, and then what BindCraft 2 actually applies.
def norm_update(g):
    flat = jnp.concatenate([jnp.asarray(g[k]) for k in sorted(g)], axis=0)
    return np.asarray(normalize_sequence_gradient(flat))

cmp = {}
for k in sorted(g_jax):
    a, b = g_jax[k].ravel(), g_dev[k].ravel()
    cmp[k] = {"rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)),
              "cos": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
              "norm_jax": float(np.linalg.norm(a)), "norm_device": float(np.linalg.norm(b))}
ua, ub = norm_update(g_jax).ravel(), norm_update(g_dev).ravel()
out["raw_gradient"] = cmp
out["bc2_normalised_update"] = {
    "rel_l2": float(np.linalg.norm(ua - ub) / max(np.linalg.norm(ua), 1e-30)),
    "cos": float(ua @ ub / max(np.linalg.norm(ua) * np.linalg.norm(ub), 1e-30)),
    "norm_ratio": float(np.linalg.norm(ub) / max(np.linalg.norm(ua), 1e-30))}
out["loss_rel"] = abs(l_dev - l_jax) / max(abs(l_jax), 1e-30)
out["stamp"] = A.stamp(3)
(HERE / f"grad_splice_check_dropout{int(DROPOUT)}.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "stamp"}, indent=1, default=str))
