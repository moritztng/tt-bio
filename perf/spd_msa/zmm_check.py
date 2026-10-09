"""spd-msa: is `opm_contract_config`'s plan RIGHT, element by element, at the shapes zmm_rule.py flagged?

zmm_rule.py saw max |rule - auto| of 0.999 (896 tokens, depth 9947) and 4.0 (384 tokens, depth 4097) where every
other size read 0.031, one bf16 step. This runs auto and the rule twice each and compares every one against a
float64 product of the same bf16 operands on row bands covering every core row of the grid (first 64 rows of each
per_core_M band, the first 256 and the last 512 rows of the matrix, which zmm_rule.py compared): max |err|, count of |err| > 0.25, and rule-vs-rule equality.

A case written TOKENS:DEPTH:u first runs ttnn's auto matmul on the UNPADDED operands, which is what zmm_rule.py
did before every padded pair.

usage: TT_VISIBLE_DEVICES=<chip> python zmm_check.py OUT [CASES=384:4097,896:9947,736:9947,512:9947]
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
CASES = [c.split(":") for c in (sys.argv[2] if len(sys.argv) > 2 else "384:4097,896:9947,736:9947,512:9947").split(",")]
LOG = open(OUT / "zmm_check.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
grid = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi3,
                                             math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
log(ev="start", cases=[":".join(c) for c in CASES], arch=str(dev.arch()), grid=[grid.x, grid.y])
for case in CASES:
    tok, depth, pre = int(case[0]), int(case[1]), case[2:] == ["u"]
    N = tok * 32
    if pre:
        torch.manual_seed(0)
        u = [ttnn.from_torch(torch.randn(N, depth).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16) for _ in range(2)]
        z = ttnn.matmul(u[0], u[1], transpose_b=True, compute_kernel_config=ckc)
        ttnn.synchronize_device(dev)
        for t in (z, *u):
            ttnn.deallocate(t)
    k = depth + T.opm_kpad_rows(depth, grid.x)
    torch.manual_seed(0)
    a_h = (torch.randn(N, k) / k ** 0.5).bfloat16()
    b_h = torch.randn(N, k).bfloat16()
    a = ttnn.from_torch(a_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(b_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    cfg = T.opm_contract_config(N // 32, N // 32, k // 32, grid)
    pcm = cfg.per_core_M * 32 if cfg else N
    rows = sorted({r for s in range(0, N, pcm) for r in range(s, min(s + 64, N))} | set(range(256))
                  | set(range(N - 512, N)))
    idx = torch.tensor(rows)
    ref = a_h[idx].double() @ b_h.double().T
    outs = {}
    for name, pc in (("auto", None), ("rule1", cfg), ("rule2", cfg), ("auto2", None)):
        z = ttnn.matmul(a, b, transpose_b=True, program_config=pc, compute_kernel_config=ckc)
        zt = ttnn.to_torch(z)[idx].double()
        ttnn.deallocate(z)
        e = (zt - ref).abs()
        bad = (e > 0.25).nonzero()
        outs[name] = zt
        log(ev="arm", tokens=tok, depth=depth, k=k, arm=name, max_err=e.max().item(), n_bad=len(bad),
            bad_tiles=sorted({(rows[i] // 32, j // 32) for i, j in bad.tolist()})[:20],
            bad_rows=sorted({rows[i] for i in bad[:, 0].tolist()})[:20],
            bad_cols=sorted(set(bad[:, 1].tolist()))[:20])
    log(ev="same", tokens=tok, rule_equal=torch.equal(outs["rule1"], outs["rule2"]),
        auto_equal=torch.equal(outs["auto"], outs["auto2"]))
    ttnn.deallocate(a); ttnn.deallocate(b)
log(ev="end")
