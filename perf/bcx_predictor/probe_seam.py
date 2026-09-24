"""Where does the gradient go? Log both directions across the seam, in both arms.

The device backward is correct at the seam (bwd_boundary.json: cos 0.963/0.975, norms
1.13x/1.10x) and the full-loop gradient is 7x small, so the loss is in the glue. A
custom_vjp identity `tap` placed on the stack's inputs and outputs records the norm of the
value going forward and of the cotangent coming back, and the SAME tap runs in both arms,
so the two columns are comparable line for line.

  in/out fwd   : the activations crossing the seam
  out bwd      : the cotangent BindCraft 2 hands the stack
  in bwd       : the gradient the stack hands back to the embeddings
"""
import collections, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, jax, jax.numpy as jnp
import bc2_state as B
import afgrad as A, stack as S
from bindcraft.af.alphafold.model import layer_stack as LS, modules
from ttbio_predictor import TTBioAlphaFoldDesignModel
from splice import EvoformerOnDevice

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
LOG = collections.defaultdict(list)

def make_tap(tag):
    @jax.custom_vjp
    def tap(x):
        return x
    def f(x):
        jax.debug.callback(lambda v, t=tag: LOG[("fwd", t)].append(float(np.linalg.norm(
            np.asarray(v, np.float32)))), x)
        return x, None
    def b(_res, g):
        jax.debug.callback(lambda v, t=tag: LOG[("bwd", t)].append(float(np.linalg.norm(
            np.asarray(v, np.float32)))), g)
        return (g,)
    tap.defvjp(f, b)
    return tap

T = {k: make_tap(k) for k in ("msa_in", "pair_in", "msa_out", "pair_out")}

def model():
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=1,
                                     max_cache_size=2, trunk="jax")

s = B.campaign_settings(overrides=["length_bucket_size=1"])
_, states, losses = B.design_state(s)

# BindCraft 2's mask, for the device arm.
cap = {}
_ri = modules.EvoformerIteration.__call__
def _spy(self, activations, masks, safe_key, use_dropout, **kw):
    if not self.is_extra_msa:
        jax.debug.callback(lambda m: cap.setdefault("mask", np.asarray(m, np.float32)),
                           masks["msa"])
    return _ri(self, activations=activations, masks=masks, safe_key=safe_key,
               use_dropout=use_dropout, **kw)
modules.EvoformerIteration.__call__ = _spy
try:
    model().predict(states)
finally:
    modules.EvoformerIteration.__call__ = _ri

real = LS.layer_stack

def run(device_stack):
    LOG.clear()
    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)
        def choose(fn):
            if getattr(fn, "__name__", None) == "evoformer_fn":
                inner = made(fn)
                def wrapped(x):
                    act, sk = x
                    m = T["msa_in"](act["msa"]); z = T["pair_in"](act["pair"])
                    if device_stack is None:
                        out, sk2 = inner(({**act, "msa": m, "pair": z}, sk))
                        mo, zo = out["msa"], out["pair"]
                    else:
                        out, sk2 = act, sk
                        mo, zo = device_stack(m, z)
                    return ({**out, "msa": T["msa_out"](mo), "pair": T["pair_out"](zo)}, sk2)
                return wrapped
            return made(fn)
        return choose
    modules.layer_stack.layer_stack = factory
    try:
        _, grads, loss = model().sequence_gradients(
            states, losses, softmax_weight=1.0, one_hot_weight=0.0,
            temperature=1.0, logit_scale=2.0)
    finally:
        modules.layer_stack.layer_stack = real
    return ({f"{d}:{t}": [round(v, 4) for v in vs] for (d, t), vs in sorted(LOG.items())},
            {k: float(np.linalg.norm(np.asarray(v))) for k, v in grads.items()}, float(loss))

out = {}
out["jax"], out["jax_grad_norms"], out["jax_loss"] = run(None)
print("JAX arm:", json.dumps(out["jax"], indent=1), flush=True)

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48, msa_mask=cap["mask"])
out["device"], out["device_grad_norms"], out["device_loss"] = run(evo.as_jax())
out["device_calls"] = dict(evo.calls)
print("device arm:", json.dumps(out["device"], indent=1), flush=True)
print("grad norms:", out["jax_grad_norms"], out["device_grad_norms"], flush=True)
(HERE / "seam_probe.json").write_text(json.dumps(out, indent=1, default=str))
