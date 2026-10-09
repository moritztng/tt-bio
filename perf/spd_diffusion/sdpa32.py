"""spd-diffusion lever 9 probe: the fp32 token-DiT attention as one SDPA program (tt_bio.sdpa_generic) vs the
shipped matmul / scale_add / softmax / matmul chain, both against a float64 reference.

usage: TT_VISIBLE_DEVICES=N python sdpa32.py OUT.jsonl [NT] [M] [LOGIT_SCALE]

ttnn's SDPA device op refuses fp32 operands; generic_op runs the same kernels without that check, so q, k, v,
the bias, the score and output accumulators and (optionally) the row statistics are all fp32 here. 736 padded
tokens is 23 tiles, prime, so no q/k chunk divides it: the SDPA arms pad the token axis to NP (default 768) and
the bias's padded key columns carry -1e4, which exp sends to exactly 0. SDPA adds the mask BEFORE the scale, so
the mask is bias / scale (the DiT's precomputed bias already carries that sqrt(head_dim) factor).
Per arm: device-synced time over REPS calls, max |o - o64| and its ratio to the shipped chain's, run-to-run
torch.equal. AICLK sampled around every timing.
"""
import json, sys, time
from pathlib import Path

import torch

OUT = Path(sys.argv[1]); NT = int(sys.argv[2]) if len(sys.argv) > 2 else 730
M = int(sys.argv[3]) if len(sys.argv) > 3 else 5
LS = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
H, D, REPS = 16, 48, 20
NP = -(-NT // 256) * 256 if NT > 512 else -(-NT // 128) * 128

try:
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
except Exception:
    pass
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import sdpa_generic as SG
from tt_bio.eltwise_fusion import scale_add

dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
g_ = dev.compute_with_storage_grid_size()
GRID = (g_.x, g_.y)
LOG = open(OUT, "a")


def aiclk():
    vals = []
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            vals.append(int((p / "tt_aiclk").read_text().split()[0]))
        except Exception:
            pass
    return max(vals) if vals else None


def log(**kw):
    kw.update(NT=NT, M=M, NP=NP, LS=LS, grid=GRID, t_unix=time.time())
    LOG.write(json.dumps(kw) + "\n"); LOG.flush(); print(json.dumps(kw), flush=True)


def up(t):
    return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)


def host(t):
    return torch.Tensor(ttnn.to_torch(t)).double()


def timed(name, fn):
    c0 = aiclk(); t0 = time.perf_counter()
    for _ in range(REPS):
        o = fn(); ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    return round((time.perf_counter() - t0) / REPS * 1e6, 1), [c0, aiclk()]


g = torch.Generator().manual_seed(0)
s = D ** -0.5
q_h, k_h, v_h = (torch.randn(M, H, NT, D, generator=g) * LS for _ in range(3))
bias_h = torch.randn(1, H, NT, NT, generator=g) * 3
o64 = torch.softmax(torch.einsum("mhid,mhjd->mhij", q_h.double(), k_h.double()) * s + bias_h.double(), -1) \
    @ v_h.double()
logits_absmax = float((torch.einsum("mhid,mhjd->mhij", q_h, k_h) * s + bias_h).abs().max())
q, k, v, bias = up(q_h), up(k_h), up(v_h), up(bias_h)


def shipped():
    sc = ttnn.matmul(q, k, transpose_b=True, core_grid=T.CORE_GRID_MAIN, compute_kernel_config=ckc)
    b = scale_add(sc, s, bias); ttnn.deallocate(sc)
    p = ttnn.softmax(b, dim=-1, compute_kernel_config=ckc); ttnn.deallocate(b)
    o = T.batched_matmul(p, v, compute_kernel_config=ckc); ttnn.deallocate(p)
    return o


o = shipped(); ttnn.synchronize_device(dev)
ref_err = float((host(o)[..., :NT, :D] - o64).abs().max())
us, clk = timed("shipped", shipped)
log(op="shipped", us=us, aiclk=clk, max_abs_vs_f64=ref_err, logits_absmax=logits_absmax)


def pad(t, rows):
    z = torch.zeros(*t.shape[:2], rows, t.shape[3]); z[:, :, :t.shape[2]] = t
    return z


qp, kp, vp = up(pad(q_h, NP)), up(pad(k_h, NP)), up(pad(v_h, NP))
mask_h = torch.full((1, H, NP, NP), -1e4); mask_h[:, :, :NT, :NT] = bias_h / s
mask = up(mask_h)
try:
    us, clk = timed("pad_q", lambda: ttnn.pad(q, [(0, 0), (0, 0), (0, NP - NT), (0, 0)], 0.0))
    log(op="pad_one_operand", us=us, aiclk=clk)
except Exception as e:
    log(op="pad_one_operand", error=str(e)[:300])

for qc, kc in [(128, 256), (256, 256), (64, 256), (128, 128), (256, 128), (128, 384), (64, 768), (128, 768)]:
    if NP % qc or NP % kc:
        continue
    for stats in (ttnn.float32, None):
        name = f"sdpa32_q{qc}_k{kc}_stats{'32' if stats else '16'}"
        kw = dict(im_dtype=ttnn.float32, out_im_dtype=ttnn.float32, stats_dtype=stats)
        p = SG.plan(qp, kp, vp, mask, qp, qc, kc, GRID, (ttnn.MathFidelity.HiFi4, False, True, False), s)
        if not SG.cb_fits_l1(p, q_dtype=ttnn.float32, k_dtype=ttnn.float32, v_dtype=ttnn.float32,
                             mask_dtype=ttnn.float32, out_dtype=ttnn.float32, **kw):
            log(op=name, error="cb over L1 (host model)", cb_bytes=SG.cb_bytes(
                p, q_dtype=ttnn.float32, k_dtype=ttnn.float32, v_dtype=ttnn.float32, mask_dtype=ttnn.float32,
                out_dtype=ttnn.float32, **kw))
            continue

        def run(qc=qc, kc=kc, kw=kw):
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([M, H, NP, D]), ttnn.float32, ttnn.TILE_LAYOUT, dev,
                                                 ttnn.DRAM_MEMORY_CONFIG)
            SG.sdpa(dev, qp, kp, vp, mask, out, qc, kc, GRID, (ttnn.MathFidelity.HiFi4, False, True, False), s,
                    **kw)
            return out
        try:
            a = run(); ttnn.synchronize_device(dev); ah = host(a); ttnn.deallocate(a)
            b = run(); ttnn.synchronize_device(dev); bh = host(b); ttnn.deallocate(b)
            us, clk = timed(name, run)
        except Exception as e:
            log(op=name, error=str(e)[:600]); continue
        err = float((ah[..., :NT, :D] - o64).abs().max())
        log(op=name, us=us, aiclk=clk, max_abs_vs_f64=err, err_ratio_vs_shipped=round(err / ref_err, 3),
            finite=bool(torch.isfinite(ah).all()), run_to_run_equal=bool(torch.equal(ah, bh)))
log(op="end")
