"""Does padding the token axis to 32 change the answer?

The acceptance arm pads n up to a multiple of 32, masks what it added and slices back --
worth 3.29x on the forward, and an unverified change on the arm that produces the
campaign's number. This grades it at n=211, which is what the earlier 3.29x was measured
at and is not tile-aligned: padded against unpadded, and BOTH against a float64 torch
reference of the same 48 blocks, so the question is whether padding moves the answer
further than bf16 already does.

Runs on an IDLE SIBLING card so the acceptance arm keeps card 3.
"""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch
import afgrad as A, stack as S
from splice import _pad_inputs

N = 211
lv = S.Levers(); dm, ref = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")

torch.manual_seed(0)
logits = torch.randn(N, 20) * 2.0
m0, z0 = A.embed(ref["bf16"], logits, torch.arange(N))
m0, z0 = m0.detach().float(), z0.detach().float()
# Two MSA rows with a realistic partial mask on the template row, as BindCraft 2 has.
m0 = torch.cat([m0, m0 * 0.5], dim=0)
mask = torch.ones(2, N); mask[1, ::3] = 0.0

rng = np.random.default_rng(0)
gm_np = rng.standard_normal(tuple(m0.shape)).astype(np.float32)
gz_np = rng.standard_normal(tuple(z0.shape)).astype(np.float32)

def run(pad):
    if pad:
        m, z, mk, n, n32 = _pad_inputs(m0.clone(), z0.clone(), mask.clone())
    else:
        m, z, mk, n, n32 = m0.clone(), z0.clone(), mask.clone(), N, N
    ml, zl = dev.leaf(m), dev.leaf(z)
    t0 = time.time()
    with dev.tt.tape():
        mo, zo = dev.stack(ml, zl, 0, 48, ckpt=True, msa_mask=dev.up(mk))
    dev.sync(); t_f = time.time() - t0
    out_m = dev.down(mo.value, tuple(m.shape))[:, :N]
    out_z = dev.down(zo.value, tuple(z.shape))[:N, :N]
    gm = torch.zeros(tuple(m.shape)); gz = torch.zeros(tuple(z.shape))
    gm[:, :N] = torch.from_numpy(gm_np); gz[:N, :N] = torch.from_numpy(gz_np)
    dev.ag.backward([mo, zo], [dev.seed(gm, mo), dev.seed(gz, zo)])
    dev.sync()
    d_m = dev.grad(ml, tuple(m.shape))[:, :N]; d_z = dev.grad(zl, tuple(z.shape))[:N, :N]
    dev.ag.release_pins()
    return out_m, out_z, d_m, d_z, t_f, n32

mr = m0.double().clone().requires_grad_(True); zr = z0.double().clone().requires_grad_(True)
mk64 = mask.double(); pm = torch.ones(N, N).double()
m, z = mr, zr
for i in range(48):
    m, z = ref["f64"].evoformer[i](m, z, mk64, pm)
torch.autograd.backward([m, z], [torch.from_numpy(gm_np).double(),
                                 torch.from_numpy(gz_np).double()])

def cmp(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return {"rel_l2": round(float((a-b).norm()/b.norm()), 6),
            "cos": round(float(a@b/(a.norm()*b.norm())), 6)}

clock = S.Clock(); t0 = time.time()
up_m, up_z, up_dm, up_dz, t_un, _ = run(False)
pd_m, pd_z, pd_dm, pd_dz, t_pd, n32 = run(True)
clock.stop()

out = {"n": N, "n_padded": n32,
       "fwd_seconds": {"unpadded": round(t_un, 3), "padded": round(t_pd, 3),
                       "speedup": round(t_un / t_pd, 3)},
       "vs_float64": {
           "unpadded_msa": cmp(up_m, m), "padded_msa": cmp(pd_m, m),
           "unpadded_pair": cmp(up_z, z), "padded_pair": cmp(pd_z, z),
           "unpadded_dmsa": cmp(up_dm, mr.grad), "padded_dmsa": cmp(pd_dm, mr.grad),
           "unpadded_dpair": cmp(up_dz, zr.grad), "padded_dpair": cmp(pd_dz, zr.grad)},
       "padded_vs_unpadded": {"msa": cmp(pd_m, up_m), "pair": cmp(pd_z, up_z),
                              "dmsa": cmp(pd_dm, up_dm), "dpair": cmp(pd_dz, up_dz)},
       "aiclk": clock.window([(t0, time.time())]),
       "card": int(__import__("os").environ.get("TT_VISIBLE_DEVICES", -1))}
(HERE / "pad_grade.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
