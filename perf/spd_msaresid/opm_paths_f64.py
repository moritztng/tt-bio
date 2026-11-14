"""spd-msaresid: OuterProductMean's three programs at one shape, each against a float64 reference.

  whole    one pass over all I token rows (what a c730 fold ran before the fix)
  blocked  the same input in row blocks, as the byte arm now asks for at c730 on Wormhole
  chunked  `x` handed over as a list of MSA_CHUNK_SIZE depth chunks (the streamed-`m` path)

Random bf16 operands and weights, the model's kernel config (HiFi3, fp32 dest acc, packer L1 acc).
Prints max |diff| between arms and each arm's error against float64 computed from the same bf16
inputs, so "differ in bits" and "which is closer to the true function" are separate numbers.

usage: TT_VISIBLE_DEVICES=<chip> python opm_paths_f64.py OUT TOKENS DEPTH C_M C_Z [SEED=0]
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
I, S, CM, CZ = (int(v) for v in sys.argv[2:6])
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
LOG = open(OUT / "opm_paths_f64.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                             math_approx_mode=False, fp32_dest_acc_en=True,
                                             packer_l1_acc=True)
g = torch.Generator().manual_seed(SEED)
bf = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).bfloat16()
w = {"norm.weight": 1 + 0.1 * bf(CM).float(), "norm.bias": 0.1 * bf(CM).float(),
     "proj_a.weight": bf(32, CM, sc=CM ** -0.5).float(), "proj_b.weight": bf(32, CM, sc=CM ** -0.5).float(),
     "proj_o.weight": bf(CZ, 1024, sc=1024 ** -0.5).float(), "proj_o.bias": 0.1 * bf(CZ).float()}
w = {k: v.bfloat16().float() for k, v in w.items()}
opm = T.OuterProductMean(w, ckc, scale_bias=True)
x = bf(1, S, I, CM)


def ref64():
    """Protenix/OpenDDE's outer_product_mean in float64 from the same bf16 numbers: layer norm,
    proj_a/b, mean over depth of a_i (x) b_j, then proj_o with its bias, all over the depth."""
    xd = x[0].double()
    m = torch.nn.functional.layer_norm(xd, (CM,), w["norm.weight"].double(), w["norm.bias"].double(), 1e-5)
    a, b = m @ w["proj_a.weight"].double().T, m @ w["proj_b.weight"].double().T  # (S, I, 32)
    o = torch.einsum("sic,sjd->ijcd", a, b).reshape(I, I, 1024)
    return (o @ w["proj_o.weight"].double().T + w["proj_o.bias"].double()) / S


def run(arm):
    T._OPM_DRAM_ROW_CAP.clear()
    old = (T.OPM_WHOLE_DRAM_DEN, T.SEQ_LEN_MORE_CHUNKING)
    if arm == "whole":
        T.OPM_WHOLE_DRAM_DEN, T.SEQ_LEN_MORE_CHUNKING = 0, 1 << 30
    elif arm == "blocked":
        T.SEQ_LEN_MORE_CHUNKING = 0
    else:
        T.OPM_WHOLE_DRAM_DEN, T.SEQ_LEN_MORE_CHUNKING = 0, 1 << 30
    try:
        if arm == "chunked":
            xin = [ttnn.from_torch(x[:, s:s + T.MSA_CHUNK_SIZE].contiguous(), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16) for s in range(0, S, T.MSA_CHUNK_SIZE)]
        else:
            xin = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        stats0 = dict(T.OPM_ROW_STATS)
        t0 = time.time()
        out = opm(xin, n_msa=float(S))
        ttnn.synchronize_device(dev)
        ms = (time.time() - t0) * 1e3
        y = ttnn.to_torch(out).float().reshape(I, I, CZ)
        return y, ms, {k: T.OPM_ROW_STATS[k] - stats0[k] for k in stats0}
    finally:
        T.OPM_WHOLE_DRAM_DEN, T.SEQ_LEN_MORE_CHUNKING = old


r = ref64()
ys = {}
for arm in ("whole", "blocked", "chunked", "whole"):
    y, ms, st = run(arm)
    assert torch.isfinite(y).all(), arm
    e = (y.double() - r).abs()
    log(arm=arm, tokens=I, depth=S, c_m=CM, c_z=CZ, seed=SEED, ms=round(ms, 1), stats=st,
        max_abs_vs_f64=e.max().item(), mean_abs_vs_f64=e.mean().item(), ref_rms=r.pow(2).mean().sqrt().item(),
        equal_to_whole=(None if "whole" not in ys else bool(torch.equal(y, ys["whole"]))),
        max_abs_vs_whole=(None if "whole" not in ys else (y - ys["whole"]).abs().max().item()))
    ys.setdefault(arm, y)
log(ev="end")
