"""Price upstream 0.4.3s own float64 DiffusionModule fwd+bwd at the accumulation-sample
granularity the trajectory actually uses (12 of the 48 noise levels per call), and at the
thread count this box can actually give it. SCOPE_LADDER priced 1 level at 4 threads."""
import json, os, sys, time, resource
NT = int(sys.argv[1]); NS = int(sys.argv[2])
os.environ["OMP_NUM_THREADS"] = str(NT); os.environ["MKL_NUM_THREADS"] = str(NT)
sys.path.insert(0, os.getcwd())
for _p in ("/home/ttuser/of3t_refprec/of3pkg043", "/home/ttuser/of3t_refprec/deps", "/home/ttuser/of3t_refprec/pylibs"):
    sys.path.insert(1, _p)
import torch
torch.set_num_threads(NT)
DIFFCAP = "/home/ttuser/of3t_softgrad/diffcap043/diffusion_boundary.pt"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
t0 = time.time()
D = torch.load(DIFFCAP, map_location="cpu", weights_only=False)
kw, cot = D["kwargs"], D["cot"]
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
del ck
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.core.model.structure.diffusion_module import DiffusionModule
cfg = OF3ProjectEntry().get_model_config_with_presets(presets=["train"])
t1 = time.time()
m = DiffusionModule(config=cfg.architecture.diffusion_module).to(torch.float64); m.train()
head = "diffusion_module."
m.load_state_dict({k[len(head):]: (v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v)
                   for k, v in sd.items() if k.startswith(head)}, strict=False)
build_s = time.time() - t1
nel = sum(p.numel() for p in m.parameters() if p.requires_grad)
call = {k: (v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()}
idx = list(range(NS))
call["t"] = call["t"][:, idx]
call["xl_noisy"] = call["xl_noisy"][:, idx]
t1 = time.time(); out = m(**call); fwd = time.time() - t1
first = out[0] if isinstance(out, (tuple, list)) else out
t1 = time.time(); torch.autograd.backward([first], [cot[:, idx].to(torch.float64).reshape(first.shape)]); bwd = time.time() - t1
hw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
row = dict(threads=NT, n_levels=NS, build_s=build_s, n_parameters=len(list(m.parameters())),
           n_elements=nel, forward_s=fwd, backward_s=bwd, one_fwd_bwd_s=fwd + bwd,
           per_level_s=(fwd + bwd) / NS, est_20_step_s=(fwd + bwd) * (48 / NS) * 20,
           est_20_step_h=(fwd + bwd) * (48 / NS) * 20 / 3600.0, peak_rss_gb=hw,
           out_shape=tuple(first.shape), load_at_start=os.getloadavg()[0])
print(json.dumps(row, indent=1), flush=True)
json.dump(row, open(f"/tmp/of3t/trajwide/price_ref_t{NT}_n{NS}.json", "w"), indent=1)
