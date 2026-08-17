"""Off-fold screen for the boltz-2 diffusion rollout-invariant hoists (L6, L7).

Times, at the production 512 aa shapes, exactly the ops the two hoists delete, batched
back-to-back the way the fold issues them (one sync per burst, never per op --
[[tt-bio-isolated-op-timing-oversync-inflates-cost]]). Also times a clone at the logits
shape so every byte model here has a measured copy roof to be scored against.

Run:
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-diffusion-perf \
  PYTHONPATH=<worktree> /home/moritz/tt-bio/env/bin/python3 perf/b2diff/screen_hoist.py
"""
import json, os, sys, time
import torch, ttnn

R = int(os.environ.get("SCREEN_REPS", "100"))
WARM = 5
OUT = os.environ.get("SCREEN_OUT", "perf/b2diff/screen_hoist_512.json")

dev = ttnn.open_device(device_id=0)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       fp32_dest_acc_en=False, packer_l1_acc=False)

def tt(shape, dtype=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(torch.randn(*shape) * 0.1, layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=dtype, memory_config=mc)

def bench(name, fn, bytes_per_iter):
    for _ in range(WARM):
        o = fn(0); ttnn.deallocate(o) if o is not None else None
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = []
    for i in range(R):
        o = fn(i)
        outs.append(o)
        if len(outs) > 8:
            ttnn.deallocate(outs.pop(0))
    ttnn.synchronize_device(dev)
    dt = time.perf_counter() - t0
    for o in outs:
        ttnn.deallocate(o)
    us = dt / R * 1e6
    gbs = bytes_per_iter / (dt / R) / 1e9
    print(f"{name:28s} {us:9.2f} us/op   {bytes_per_iter/1e6:8.2f} MB   {gbs:7.1f} GB/s", flush=True)
    return {"us_per_op": us, "mb": bytes_per_iter / 1e6, "gb_s": gbs}

res = {"ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else "0.68.0",
       "host": os.uname().nodename, "reps": R, "note": "pc card 0; timings only, no parity claim"}

# --- copy roof at the logits shape, the denominator every byte model below is scored against
logits = tt((1, 16, 512, 512))
res["clone_logits"] = bench("clone[1,16,512,512]", lambda i: ttnn.clone(logits), 2 * 8.389e6)

# --- L7 token: the per-layer bias slice, 4800/fold, MEASURED in opcensus_512.json root_by_op
zt = tt((1, 384, 512, 512))
res["slice_token_bias"] = bench("slice[1,384,512,512]->16", lambda i: zt[:, (i % 24) * 16:(i % 24) * 16 + 16, :, :],
                                2 * 8.389e6)
ttnn.deallocate(zt)

# --- L7 atom: 1200/fold
za = tt((224, 12, 32, 128))
res["slice_atom_bias"] = bench("slice[224,12,32,128]->4", lambda i: za[:, (i % 3) * 4:(i % 3) * 4 + 4, :, :],
                               (224 * 4 * 32 * 128 * 2) * 2)
ttnn.deallocate(za)

# --- L6: the atom AdaLN conditioning half, 4 ops x 2400 calls/fold
s_atom = tt((1, 224, 32, 128))
w = ttnn.from_torch(torch.ones(128), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
W1 = tt((128, 128)); B1 = ttnn.from_torch(torch.zeros(128), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
W2 = tt((128, 128))
NB = 224 * 32 * 128 * 2
res["adaln_atom_to_memcfg"] = bench("to_memory_config(s)->L1",
    lambda i: ttnn.to_memory_config(s_atom, memory_config=ttnn.L1_MEMORY_CONFIG), 2 * NB)
sn = ttnn.layer_norm(s_atom, weight=w, epsilon=1e-5, compute_kernel_config=ckc)
res["adaln_atom_layernorm"] = bench("layer_norm(s,w)[atom]",
    lambda i: ttnn.layer_norm(s_atom, weight=w, epsilon=1e-5, compute_kernel_config=ckc), 2 * NB)
res["adaln_atom_linear"] = bench("linear(s,[128,128])[atom]",
    lambda i: ttnn.linear(sn, W1, bias=B1, compute_kernel_config=ckc,
                          memory_config=ttnn.L1_MEMORY_CONFIG), 2 * NB + 128 * 128 * 2)
ttnn.deallocate(sn)

# --- L1's op, for the token side, on the same instrument
s_tok = tt((1, 512, 768))
NT = 512 * 768 * 2
wt = ttnn.from_torch(torch.ones(768), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
res["adaln_token_layernorm"] = bench("layer_norm(s,w)[token]",
    lambda i: ttnn.layer_norm(s_tok, weight=wt, epsilon=1e-5, compute_kernel_config=ckc), 2 * NT)
res["adaln_token_layernorm_nw"] = bench("layer_norm(a)[token]",
    lambda i: ttnn.layer_norm(s_tok, epsilon=1e-5, compute_kernel_config=ckc), 2 * NT)
res["eltwise_token_add"] = bench("add[1,512,768]",
    lambda i: ttnn.add(s_tok, s_tok), 3 * NT)

json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
ttnn.close_device(dev)
