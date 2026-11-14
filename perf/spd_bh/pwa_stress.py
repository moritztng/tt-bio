"""spd-bh: hammer one Protenix-v2 MSA op path on one chip, unsynced as the fold runs it, to tell a bad
card from a bad op after a Blackhole hang.

  TT_VISIBLE_DEVICES=<chip> PYTHONPATH=<engine tree> timeout 1800 python pwa_stress.py OUT [ARM] [CALLS]

ARM is `unpadded` (PairWeightedAveraging with TT_BIO_PWA_UNPADDED's head layout, the path the qb1 card 1
hang of 2026-10-09 stopped in), `fused` (the padded fused-heads path) or `reshape` (only the row-major
to_layout / reshape / permute round trip the unpadded path adds). Calls are enqueued back to back with a
synchronize every 100, where progress is logged, so a hang shows as a stalled log and the outer `timeout`
ends it. Shapes are the c730 cell: one 512-row depth chunk, 736 tokens, c_m 128.
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ARM = sys.argv[2] if len(sys.argv) > 2 else "unpadded"
CALLS = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
LOG = open(OUT / f"stress_{ARM}.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time()
    LOG.write(json.dumps(kw) + "\n"); LOG.flush(); print(json.dumps(kw), flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T

torch.manual_seed(0)
dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
ROWS, TOK, C_M, C_Z, H, HD = 512, 736, 128, 256, 8, 8
bf = lambda t: t.to(torch.bfloat16)
up = lambda t: ttnn.from_torch(bf(t), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
lin = lambda o, i: bf(torch.randn(o, i) / i ** 0.5)
sd = {"norm_m.weight": bf(1 + 0.1 * torch.randn(C_M)), "norm_m.bias": bf(0.1 * torch.randn(C_M)),
      "norm_z.weight": bf(1 + 0.1 * torch.randn(C_Z)), "norm_z.bias": bf(0.1 * torch.randn(C_Z)),
      "proj_m.weight": lin(H * HD, C_M), "proj_g.weight": lin(H * HD, C_M),
      "proj_z.weight": lin(H, C_Z), "proj_o.weight": lin(C_M, H * HD)}
pwa = T.PairWeightedAveraging(HD, H, sd, ckc)
m_tt = up(torch.randn(1, ROWS, TOK, C_M))
ws = pwa.head_weights(up(torch.randn(1, TOK, TOK, C_Z)), None)
T._PWA_FUSED_HEADS, T._PWA_UNPADDED = True, ARM == "unpadded"

if ARM == "reshape":
    x = up(torch.randn(ROWS, TOK, H * HD))

    def call():
        vt = ttnn.to_layout(ttnn.permute(x, (1, 2, 0)), ttnn.ROW_MAJOR_LAYOUT)
        vh = ttnn.permute(ttnn.reshape(vt, (TOK, H, HD * ROWS)), (1, 0, 2))
        ttnn.deallocate(vt)
        return ttnn.to_layout(vh, ttnn.TILE_LAYOUT)
else:
    def call():
        return pwa(m_tt, None, weights=ws)

log(ev="start", arm=ARM, calls=CALLS, stats=list(T.PWA_UNPADDED_STATS))
t0 = time.perf_counter()
for i in range(1, CALLS + 1):
    ttnn.deallocate(call())
    if i % 100 == 0:
        ttnn.synchronize_device(dev)
        log(ev="progress", calls=i, s=round(time.perf_counter() - t0, 2))
log(ev="end", arm=ARM, calls=CALLS, s=round(time.perf_counter() - t0, 2), stats=list(T.PWA_UNPADDED_STATS))
