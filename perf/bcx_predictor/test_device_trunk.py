"""Does the JAX primitive carry tt-bio's trunk AND its gradient correctly?

Graded against a float64 torch reference of the SAME stack on the SAME inputs -- never
against the other device arm. Small k for the plumbing, then the production 4+48 shape
once to show it runs where it has to.
"""
import json, os, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch, jax, jax.numpy as jnp
import afgrad as A, stack as S
from device_trunk import DeviceTrunk

PARAMS = os.environ.get("BCX_AF2", A.DEFAULT_PARAMS)
out = {}
lv = S.Levers(); dm, ref = A.load_models(PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")

def build(n, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(n, 20) * 2.0
    ridx = torch.arange(n)
    m0, z0 = A.embed(ref["bf16"], logits, ridx)
    return m0.detach(), z0.detach()

def probe(n, ke, kv, tag):
    m0, z0 = build(n)
    trunk = DeviceTrunk(dev, k_extra=ke, k_evo=kv, checkpoint=True)
    f = trunk.as_jax()
    mj, zj = jnp.asarray(m0.float().numpy()), jnp.asarray(z0.float().numpy())

    # A scalar the device gradient can be graded on: fixed random readout, same for both arms.
    rng = np.random.default_rng(0)
    ws = jnp.asarray(rng.standard_normal((n, trunk.c_single)).astype(np.float32))
    wp = jnp.asarray(rng.standard_normal((n, n, z0.shape[-1])).astype(np.float32))
    def loss(msa, pair):
        s, p = f(msa, pair)
        return (s * ws).sum() + (p * wp).sum()

    t0 = time.time(); val = float(jax.jit(loss)(mj, zj)); t_fwd = time.time() - t0
    t1 = time.time(); g_msa, g_pair = jax.grad(loss, argnums=(0, 1))(mj, zj); t_bwd = time.time() - t1
    g_msa, g_pair = np.asarray(g_msa), np.asarray(g_pair)

    # float64 torch reference of the same stack, same readout.
    mr = m0.double().clone().requires_grad_(True); zr = z0.double().clone().requires_grad_(True)
    m64, z64 = A.ref_stack(ref["f64"], mr, zr, ke, kv)
    s64 = ref["f64"].single_activations(m64[0])
    l64 = (s64 * torch.from_numpy(np.asarray(ws)).double()).sum() + \
          (z64 * torch.from_numpy(np.asarray(wp)).double()).sum()
    l64.backward()

    r = {"n": n, "k_extra": ke, "k_evo": kv,
         "loss_device": val, "loss_f64": float(l64),
         "loss_rel": abs(val - float(l64)) / max(abs(float(l64)), 1e-30),
         "g_msa_rel_l2": A.rel_l2(torch.from_numpy(g_msa), mr.grad),
         "g_msa_cos": A.cosine(torch.from_numpy(g_msa), mr.grad),
         "g_pair_rel_l2": A.rel_l2(torch.from_numpy(g_pair), zr.grad),
         "g_pair_cos": A.cosine(torch.from_numpy(g_pair), zr.grad),
         "fwd_s": round(t_fwd, 2), "bwd_s": round(t_bwd, 2),
         "live_tapes_after": DeviceTrunk.live_tapes()}
    out[tag] = r
    print(tag, json.dumps({k: r[k] for k in ("loss_rel", "g_msa_rel_l2", "g_msa_cos",
                                             "g_pair_rel_l2", "g_pair_cos", "fwd_s", "bwd_s",
                                             "live_tapes_after")}, indent=1), flush=True)

probe(64, 1, 2, "plumbing_n64_k1_2")

# bf16's own distance is measured by bf16_control.py, CPU-only, and is NOT repeated here:
# the first attempt at it in this file handed the bf16 model float32 activations and so
# measured a float32 arm (0.00074), which made the device look 166x worse than bf16 when it
# is 1.35x. A dtype-parameterised reference takes its precision from the ACTIVATIONS --
# load_models says so in its own comment: the parameters stay float32 and the Linear casts
# the weight to the activation dtype per call.

# Controls that must MOVE: a zero cotangent gives zero, and the tape must not leak.
m0, z0 = build(64)
tr = DeviceTrunk(dev, k_extra=1, k_evo=2); f = tr.as_jax()
mj, zj = jnp.asarray(m0.float().numpy()), jnp.asarray(z0.float().numpy())
gz = jax.grad(lambda a, b: (f(a, b)[0] * 0.0).sum(), argnums=(0, 1))(mj, zj)
out["zero_cotangent_gives_zero"] = bool(np.abs(np.asarray(gz[0])).max() == 0.0
                                        and np.abs(np.asarray(gz[1])).max() == 0.0)
out["live_tapes_at_end"] = DeviceTrunk.live_tapes()
assert out["live_tapes_at_end"] == 0, (
    f"{out['live_tapes_at_end']} tapes left live: a forward that banks a tape nobody frees is "
    "an OOM, at 5.33 GB an Evoformer block by bcx-ckpt's measurement")
print("zero-cotangent control:", out["zero_cotangent_gives_zero"],
      "live tapes:", out["live_tapes_at_end"], flush=True)

# Production shape, forward only, to show it runs where it must.
m0, z0 = build(192)
tr = DeviceTrunk(dev, k_extra=4, k_evo=48); f = tr.as_jax()
t0 = time.time()
s, p = f(jnp.asarray(m0.float().numpy()), jnp.asarray(z0.float().numpy()))
out["production_n192_k4_48"] = {"single_shape": list(s.shape), "pair_shape": list(p.shape),
                                "finite": bool(np.isfinite(np.asarray(s)).all()
                                               and np.isfinite(np.asarray(p)).all()),
                                "fwd_s": round(time.time() - t0, 2)}
print("production:", json.dumps(out["production_n192_k4_48"]), flush=True)
out["stamp"] = A.stamp(3)
(HERE / "device_trunk_check.json").write_text(json.dumps(out, indent=1, default=str))
print("wrote device_trunk_check.json")
