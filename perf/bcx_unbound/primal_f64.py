"""Grade the PRIMAL splice path against a float64 reference.

Every correctness result this campaign banked graded GRADIENTS -- bcx-afgrad's per-block VJP,
bcx-e2e's joined step, bcx-stack, bcx-bytes -- which is the taped path. `_primal`
(`splice.py:131-147`, `dev.up`, `ckpt=False`, no tape) has never been graded here, and it is a
different kernel route by construction: bcx-forward established that the fused triangle
attention serves untaped calls and declines under a tape, so `_primal` runs the fused route
and `_taped` does not.

Method is bcx-predictor's `boundary.py`, which captures BindCraft 2's real (msa, pair) at
`modules.py`'s evoformer_fn during a `predict` call, extended with the float64 arm:

  device  bf16 on card, untaped -> the fused route `_primal` actually takes
  f64     the reference promoted to float64, same bf16-rounded inputs
  bf16    the same reference at bf16 on the host -- the envelope, so the device number reads
          against what bf16 costs anyway rather than against zero

bucket 1, so BindCraft 2 pads nothing and the splice's pair mask is not in the comparison.
"""
import json, os, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_predictor"),
          str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, jax
import bc2_state as B
import afgrad as A, stack as S
from bindcraft.af.alphafold.model import layer_stack as LS, modules
from ttbio_predictor import TTBioAlphaFoldDesignModel

BUDGET_S = float(os.environ.get("BCX_F64_BUDGET_S", "600"))
K_EVO = 48

cap, MASKS = {}, {}
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
    # predict, not sequence_gradients: the primal path is what mutate calls.
    TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir="/home/ttuser/bcx_e2e/af2_params",
                              models=("model_1_ptm",), num_recycle=1, key=jax.random.PRNGKey(0),
                              length_bucket_size=1, max_cache_size=2, dropout=False,
                              trunk="jax").predict(states)
finally:
    modules.layer_stack.layer_stack = real
    modules.EvoformerIteration.__call__ = _real_iter

import torch
m_t = torch.from_numpy(cap["msa_in"]).float()
z_t = torch.from_numpy(cap["pair_in"]).float()
mask_t = torch.from_numpy(cap["msa_mask"]).float()
n = z_t.shape[0]
out = {"why": "the primal path has never been graded against float64 on this campaign",
       "bucket": 1, "n": int(n), "k_evo": K_EVO, "route": "untaped -> fused triangle attention",
       "msa_in_shape": list(cap["msa_in"].shape), "pair_in_shape": list(cap["pair_in"].shape),
       "capture": "bindcraft/af/alphafold/model/modules.py evoformer_fn, during predict"}


def cmp(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return {"rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)),
            "cos": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "max_abs": float(np.abs(a - b).max()),
            "norm_ref": float(np.linalg.norm(a)), "norm_other": float(np.linalg.norm(b))}


# ------------------------------------------------------------------ device, untaped, bf16
lv = S.Levers(); dm, ref = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
clock = S.Clock(); w0 = time.time()
mo, zo = dev.stack(dev.up(m_t), dev.up(z_t), 0, K_EVO, ckpt=False, msa_mask=dev.up(mask_t))
dev.sync()
w1 = time.time(); clock.stop()
d_msa = dev.down(mo, tuple(m_t.shape)).numpy()
d_pair = dev.down(zo, tuple(z_t.shape)).numpy()
out["device_s"] = round(w1 - w0, 1)
out["aiclk_during"] = clock.window([(w0, w1)])
print("device", out["device_s"], "s", json.dumps(out["aiclk_during"], default=str), flush=True)

# The device arm reads bf16-rounded inputs. The references get the SAME rounded inputs, so the
# comparison is about the stack and not about the upload.
m_b = m_t.to(torch.bfloat16).float()
z_b = z_t.to(torch.bfloat16).float()

def ref_stack_masked(model, msa, pair, k_evo):
    """`afgrad.ref_evo` hardcodes an all-ones MSA mask. The captured mask is not all ones --
    row 1 is 40% zero at bucket 1 -- and the device leg passes it, so the reference has to."""
    dt = pair.dtype
    mm = mask_t.to(dt)
    ones = torch.ones(pair.shape[0], pair.shape[0], dtype=dt)
    for i in range(k_evo):
        msa, pair = model.evoformer[i](msa, pair, mm, ones)
    return msa, pair


# ------------------------------------------------------------------ one f64 block, to price it
t0 = time.time()
ref_stack_masked(ref["f64"], m_b.double(), z_b.double(), 1)
block_s = time.time() - t0
k_f64 = K_EVO if block_s * K_EVO <= BUDGET_S else max(1, int(BUDGET_S // block_s))
out["f64_block_s"] = round(block_s, 1)
out["f64_blocks_graded"] = int(k_f64)
print("f64 block", round(block_s, 1), "s -> grading", k_f64, "blocks", flush=True)

for name, dt in (("f64", torch.float64), ("bf16", torch.bfloat16)):
    t0 = time.time()
    rm, rz = ref_stack_masked(ref[name], m_b.to(dt), z_b.to(dt), k_f64)
    out[f"{name}_s"] = round(time.time() - t0, 1)
    out[f"ref_{name}"] = {"msa_norm": float(rm.double().norm()), "pair_norm": float(rz.double().norm())}
    if name == "f64":
        f64_m, f64_z = rm.double().numpy(), rz.double().numpy()
    else:
        bf16_m, bf16_z = rm.double().numpy(), rz.double().numpy()
    print(name, out[f"{name}_s"], "s", flush=True)
    (HERE / "primal_f64.json").write_text(json.dumps(out, indent=1, default=str))

if k_f64 == K_EVO:
    out["device_vs_f64"] = {"msa": cmp(f64_m, d_msa), "pair": cmp(f64_z, d_pair)}
    out["host_bf16_vs_f64"] = {"msa": cmp(f64_m, bf16_m), "pair": cmp(f64_z, bf16_z)}
    out["bc2_jax_vs_f64"] = {"msa": cmp(f64_m, cap["msa_out"]), "pair": cmp(f64_z, cap["pair_out"])}
    out["device_vs_bc2_jax"] = {"msa": cmp(cap["msa_out"], d_msa), "pair": cmp(cap["pair_out"], d_pair)}
else:
    # A partial stack cannot be compared with the whole-stack device output. Grade the same
    # prefix on the device instead, so both arms stop at the same block.
    pm, pz = dev.stack(dev.up(m_t), dev.up(z_t), 0, k_f64, ckpt=False, msa_mask=dev.up(mask_t))
    dev.sync()
    p_msa = dev.down(pm, tuple(m_t.shape)).numpy(); p_pair = dev.down(pz, tuple(z_t.shape)).numpy()
    out["prefix_only"] = (f"{k_f64} of {K_EVO} blocks: the f64 arm did not fit the budget, so both "
                          "arms stop at the same block and the whole-stack device output is not "
                          "compared against a partial reference")
    out["device_vs_f64"] = {"msa": cmp(f64_m, p_msa), "pair": cmp(f64_z, p_pair)}
    out["host_bf16_vs_f64"] = {"msa": cmp(f64_m, bf16_m), "pair": cmp(f64_z, bf16_z)}

out["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]))
out["sysfs_node"] = list(S.sysfs_node())
(HERE / "primal_f64.json").write_text(json.dumps(out, indent=1, default=str))
for k in ("device_vs_f64", "host_bf16_vs_f64", "bc2_jax_vs_f64", "device_vs_bc2_jax"):
    if k in out:
        print(k, json.dumps({t: {m: round(v[m], 6) for m in ("rel_l2", "cos")}
                             for t, v in out[k].items()}), flush=True)
