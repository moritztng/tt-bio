import sys, time, numpy as np, torch
sys.path.insert(0, "perf/bcx_e2e")
import e2e, afgrad as A
torch.set_num_threads(8)
n_t, n_b, n_br = e2e.STATES[128]; n = 128
vg32, vg64, batch, meta = e2e.tail_arms(n_t, n_b, n_br)
t0 = time.time(); vgj = e2e.jax_program(A.DEFAULT_PARAMS, meta, n); print("jax built", round(time.time()-t0, 1), flush=True)
_, ref = A.load_models(A.DEFAULT_PARAMS, device_arm=False)
ridx = torch.from_numpy(np.array(meta["residue_index"])).long()
torch.manual_seed(0); logits = torch.randn(n, 20) * 2.0
t0 = time.time(); (lj, rj), gj = vgj(logits.double().numpy()); gj = e2e.to_t(gj)
print("jax f64 loss", float(lj), "dtype", np.asarray(rj["pair"]).dtype, "t", round(time.time()-t0, 1), flush=True)
t0 = time.time(); g64, l64, i64 = e2e.ref_step(ref, vg64, logits, ridx, 4, 48, "f64", torch.float64, np.float64)
print("torch f64 loss", l64, "t", round(time.time()-t0, 1))
print("torch_f64 vs jax_f64", A.cmp(g64, gj), "norms", float(g64.norm()), float(gj.norm()))
