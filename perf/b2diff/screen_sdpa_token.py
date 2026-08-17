"""S6 screen: the boltz-2 token DiT attention core, unfused chain vs fused SDPA, off-fold.

S1 (state/boltz2-512aa-deep-perf.md 11.4) rejected the fused route on an arm that rerouted the
4800 token calls AND the 264 trunk calls at once. The trunk hands its pair bias over in L1
(_PAIR_BIAS_L1_NORM) and ttnn SDPA TT_FATALs on an L1 mask, so that arm forced an 8 MB per-call
DRAM spill and moved block:PairformerLayer by +771.8 ms, MORE than AttentionPairBias itself moved.
The token DiT's bias is bias_token, uploaded once per fold to DRAM, so it never had that problem.
This screen prices the token half alone.

Shapes are the production ones at 512 aa: q/k/v [1,16,512,64] (head_dim 48 padded to 64),
bias [1,16,512,512], every tensor interleaved DRAM.
"""
import json, os, time
import torch, ttnn
import tt_bio.tenstorrent as T

R = int(os.environ.get("SCREEN_REPS", "50"))
WARM = 5
OUT = os.environ.get("SCREEN_OUT", "perf/b2diff/screen_sdpa_token_512.json")

dev = ttnn.open_device(device_id=0)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       fp32_dest_acc_en=False, packer_l1_acc=False)
H, S, D = 16, 512, 64
def tt(shape):
    return ttnn.from_torch(torch.randn(*shape) * 0.1, layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.bfloat16)
q, k, v, z = tt((1, H, S, D)), tt((1, H, S, D)), tt((1, H, S, D)), tt((1, H, S, S))
scale = 48 ** -0.5

def unfused(_):
    kt = ttnn.transpose(k, -2, -1)
    logits = T.batched_matmul(q, kt, compute_kernel_config=ckc)
    ttnn.deallocate(kt)
    logits = ttnn.add_(logits, z)
    logits = ttnn.multiply_(logits, scale)
    probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=ckc)
    ttnn.deallocate(logits)
    o = T.batched_matmul(probs, v, compute_kernel_config=ckc)
    ttnn.deallocate(probs)
    return o

def unfused_l2(_):     # with L2 (_APB_SCALE_FOLD) already applied: the multiply_ is gone
    kt = ttnn.transpose(k, -2, -1)
    logits = T.batched_matmul(q, kt, compute_kernel_config=ckc)
    ttnn.deallocate(kt)
    logits = ttnn.add_(logits, z)
    probs = ttnn.softmax(logits, dim=-1, compute_kernel_config=ckc)
    ttnn.deallocate(logits)
    o = T.batched_matmul(probs, v, compute_kernel_config=ckc)
    ttnn.deallocate(probs)
    return o

def fused(_):
    return ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=z, is_causal=False, scale=scale,
        program_config=T._sdpa_program_config_for_lengths(S, S))

def bench(name, fn):
    for _ in range(WARM):
        ttnn.deallocate(fn(0))
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = []
    for i in range(R):
        outs.append(fn(i))
        if len(outs) > 4:
            ttnn.deallocate(outs.pop(0))
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / R
    for o in outs:
        ttnn.deallocate(o)
    print(f"{name:22s} {dt*1e6:9.2f} us/call", flush=True)
    return dt * 1e6

res = {"host": os.uname().nodename, "reps": R,
       "note": "pc card 0; timings only. Card 0 miscomputes some matmuls -- no parity claim here."}
res["unfused"] = bench("unfused chain", unfused)
res["unfused_l2"] = bench("unfused, L2 applied", unfused_l2)
try:
    res["fused_sdpa"] = bench("fused ttnn SDPA", fused)
except Exception as e:
    res["fused_sdpa"] = None
    res["fused_error"] = repr(e)[:400]
    print("fused SDPA failed:", repr(e)[:400])
# alternating repeat, so a drift cannot masquerade as the lever
res["unfused_2"] = bench("unfused chain (2nd)", unfused)
if res.get("fused_sdpa"):
    res["fused_sdpa_2"] = bench("fused ttnn SDPA (2nd)", fused)

json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
ttnn.close_device(dev)
