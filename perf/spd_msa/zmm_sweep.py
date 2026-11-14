"""spd-msa: how often does each OPM contraction path put a WRONG element, across many (tokens, depth) shapes?

zmm_check.py found single elements off by 1, 2 or 4 (operands scaled so outputs are ~N(0,1)) in the padded paths
(OPM_KPAD's auto at 384/4097, OPM_CFG's plan at 896/9947, 1024/9947, 384/13602) and none in the unpadded auto path
the model runs without the levers. This runs, per shape, with the model's kernel config (HiFi3, fp32 dest acc,
packer_l1_acc on): `unpad` (auto on K = depth), `pad` (auto on K rounded up by opm_kpad_rows) and `cfg` (the
opm_contract_config plan on the padded K, if it returns one). Rows 0-1023 and the last 512 are compared in full
against a float64 product of the same bf16 operands; an element with |err| > 0.25 is wrong. One timed call per arm.

usage: TT_VISIBLE_DEVICES=<chip> python zmm_sweep.py OUT TOKENS(,..) DEPTHS(,..)
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TOKS = [int(x) for x in sys.argv[2].split(",")]
DEPTHS = [int(x) for x in sys.argv[3].split(",")]
LOG = open(OUT / "zmm_sweep.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

torch.set_num_threads(16)
dev = T.get_device()
grid = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
log(ev="start", toks=TOKS, depths=DEPTHS, arch=str(dev.arch()), grid=[grid.x, grid.y])
for tok in TOKS:
    for depth in DEPTHS:
        N = tok * 32
        kp = depth + T.opm_kpad_rows(depth)
        torch.manual_seed(depth * 7 + tok)
        a_h = (torch.randn(N, kp) / depth ** 0.5).bfloat16(); a_h[:, depth:] = 0
        b_h = torch.randn(N, kp).bfloat16(); b_h[:, depth:] = 0
        rows = torch.tensor(sorted(set(range(min(1024, N))) | set(range(max(0, N - 512), N))))
        ref = a_h[rows, :depth].double() @ b_h[:, :depth].double().T
        cfg = T.opm_contract_config(N // 32, N // 32, kp // 32, grid)
        arms = [("unpad", depth, None)] + ([("pad", kp, None)] if kp != depth else [])
        arms += [("cfg", kp, cfg)] if cfg is not None else []
        for name, k, pc in arms:
            a = ttnn.from_torch(a_h[:, :k].contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            b = ttnn.from_torch(b_h[:, :k].contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            z = ttnn.matmul(a, b, transpose_b=True, program_config=pc, compute_kernel_config=ckc)
            zt = ttnn.to_torch(z)[rows].double()
            ttnn.deallocate(z)
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            ttnn.deallocate(ttnn.matmul(a, b, transpose_b=True, program_config=pc, compute_kernel_config=ckc))
            ttnn.synchronize_device(dev); ms = (time.perf_counter() - t0) * 1e3
            ttnn.deallocate(a); ttnn.deallocate(b)
            e = (zt - ref).abs()
            bad = (e > 0.25).nonzero()
            log(ev="arm", tokens=tok, depth=depth, k=k, arm=name, ms=round(ms, 2), max_err=round(e.max().item(), 4),
                n_bad=len(bad), bad=[(rows[i].item(), j, round(e[i, j].item(), 3)) for i, j in bad.tolist()[:8]])
log(ev="end")
