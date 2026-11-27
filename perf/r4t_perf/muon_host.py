"""Host cost of tt_bio.train.optim.Muon.orthogonalize (upstream bf16 Newton-Schulz, 5 steps) over RFD4-Proteina's
denoiser 2-D weights (shapes from the release base/model.safetensors, nn.* only). Same op sequence as optim.py:247-261."""
import json, sys, time, torch
shapes = json.loads(open(sys.argv[1]).read().splitlines()[2])
a, b, c = 3.4445, -4.775, 2.0315
def ns(x):
    x = x[None].bfloat16()
    if x.shape[-2] > x.shape[-1]:
        x = x.transpose(-1, -2)
    x.div_(x.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-7))
    for _ in range(5):
        g = torch.bmm(x, x.transpose(-1, -2))
        x = torch.baddbmm(x, torch.baddbmm(g, g, g, beta=b, alpha=c), x, beta=a)
    return x
tot_s = tot_f = 0.0
for (r, co), cnt in shapes:
    w = torch.randn(r, co)
    ns(w)
    t = time.perf_counter(); k = 3
    for _ in range(k): ns(w)
    dt = (time.perf_counter() - t) / k
    m, n = min(r, co), max(r, co)
    fl = 5 * (4 * m * m * n + 2 * m ** 3)
    tot_s += dt * cnt; tot_f += fl * cnt
print(json.dumps(dict(threads=torch.get_num_threads(), torch=torch.__version__, ns_s_per_step=round(tot_s, 3),
                      tflop_per_step=round(tot_f / 1e12, 4), host_gflops=round(tot_f / tot_s / 1e9, 1),
                      device_roof_s=round(tot_f / 115.685e12, 5))))
