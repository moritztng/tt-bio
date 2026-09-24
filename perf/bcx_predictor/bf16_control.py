"""What IS torch bf16's own distance on this stack? The earlier control cast it to float32.

`afgrad.embed` returns activations already in `model.trunk_dtype`, so the bf16 arm's inputs
are bf16. The first control did `m0.to(torch.bfloat16).float()`, handing the bf16 model
float32 activations -- and `load_models` notes the parameters stay float32 and the Linear
casts the weight to the ACTIVATION dtype per call, so float32 activations make a float32
arm. That is why it read 0.00074. No device is opened here; both arms are torch.
"""
import json, os, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad")):
    sys.path.insert(0, p)
import numpy as np, torch
import afgrad as A

PARAMS = os.environ.get("BCX_AF2", A.DEFAULT_PARAMS)
_, ref = A.load_models(PARAMS, device_arm=False)

def arm(name, n, ke, kv, as_float32):
    torch.manual_seed(0)
    logits = torch.randn(n, 20) * 2.0
    ridx = torch.arange(n)
    m0, z0 = A.embed(ref["bf16"], logits, ridx)          # bf16 activations
    m0, z0 = m0.detach(), z0.detach()
    mi = m0.clone().float() if as_float32 else m0.clone()
    zi = z0.clone().float() if as_float32 else z0.clone()
    mi.requires_grad_(True); zi.requires_grad_(True)
    m, z = A.ref_stack(ref["bf16"], mi, zi, ke, kv)
    s = ref["bf16"].single_activations(m[0])
    rng = np.random.default_rng(0)
    ws = torch.from_numpy(rng.standard_normal((n, 384)).astype(np.float32))
    wp = torch.from_numpy(rng.standard_normal((n, n, z0.shape[-1])).astype(np.float32))
    ((s.float() * ws).sum() + (z.float() * wp).sum()).backward()

    mr = m0.double().clone().requires_grad_(True); zr = z0.double().clone().requires_grad_(True)
    m64, z64 = A.ref_stack(ref["f64"], mr, zr, ke, kv)
    s64 = ref["f64"].single_activations(m64[0])
    ((s64 * ws.double()).sum() + (z64 * wp.double()).sum()).backward()
    return {"arm": name, "input_dtype": str(mi.dtype), "activations_float32": as_float32,
            "g_msa_rel_l2": A.rel_l2(mi.grad, mr.grad), "g_msa_cos": A.cosine(mi.grad, mr.grad),
            "g_pair_rel_l2": A.rel_l2(zi.grad, zr.grad), "g_pair_cos": A.cosine(zi.grad, zr.grad)}

out = {"bf16_activations_the_honest_control": arm("bf16", 64, 1, 2, False),
       "float32_activations_what_was_measured_before": arm("mislabelled", 64, 1, 2, True),
       "device_from_device_trunk_check": {
           "g_msa_rel_l2": 0.12259594010820524, "g_msa_cos": 0.9924605386448399,
           "g_pair_rel_l2": 0.08311615476329591, "g_pair_cos": 0.996540010736473}}
out["device_over_bf16_g_msa"] = round(
    out["device_from_device_trunk_check"]["g_msa_rel_l2"]
    / out["bf16_activations_the_honest_control"]["g_msa_rel_l2"], 2)
(HERE / "bf16_control.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
