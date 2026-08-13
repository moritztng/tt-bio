"""Screen for the persistent-mask q-split: the actual op, at the shapes the fold actually runs.

Stock arm is what the fold does today at these sizes -- `triatt_sdpa.sdpa` declines on
`fill_preconditions` and the call falls through to `ttnn.transformer.scaled_dot_product_attention`
at the same program config. The qsplit arm is the same call with `q_pf = q_num_chunks`.
"""
import json, os, statistics as st, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio import triatt_sdpa as PM
from tt_bio.tenstorrent import _sdpa_program_config

REPS, SIZES = 7, [int(x) for x in sys.argv[1].split(",")]
OUT = sys.argv[2]
dev = ttnn.open_device(device_id=0)
res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"), "loadavg": open("/proc/loadavg").read().split()[:3], "cases": []}


def timed(f):
    for _ in range(2):
        o = f(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev); t0 = time.perf_counter(); o = f()
        ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0)
        if _ < REPS - 1:
            ttnn.deallocate(o)
    return st.median(ts) * 1e3, o


for S in SIZES:
    H, D, q_chunk, k_chunk = 12, 32, S // 2, 256
    g = torch.Generator().manual_seed(S)
    mk = lambda sh: ttnn.from_torch(torch.randn(*sh, generator=g).to(torch.bfloat16),
                                    layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q, k, v, bias = mk((S, H, S, D)), mk((S, H, S, D)), mk((S, H, S, D)), mk((1, H, S, S))
    scale = D ** -0.5
    case = {"S": S, "H": H, "q_chunk": q_chunk, "k_chunk": k_chunk}

    PM._Q_SPLIT = False
    case["stock_serves"] = PM.sdpa(q, k, v, bias, scale, q_chunk, k_chunk) is not None
    ms_stock, o_stock = timed(lambda: ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=scale,
        program_config=_sdpa_program_config(q_chunk, k_chunk)))

    PM._Q_SPLIT = True
    probe = PM.sdpa(q, k, v, bias, scale, q_chunk, k_chunk)
    case["qsplit_serves"] = probe is not None
    if probe is None:
        case["reject"] = {str(kk): vv for kk, vv in PM.REJECTS.items()}
        ms_q = None
    else:
        ttnn.deallocate(probe)
        ms_q, o_q = timed(lambda: PM.sdpa(q, k, v, bias, scale, q_chunk, k_chunk))
        case["torch_equal"] = bool(torch.equal(ttnn.to_torch(o_stock), ttnn.to_torch(o_q)))
        case["max_abs_diff"] = float((ttnn.to_torch(o_stock).float()
                                      - ttnn.to_torch(o_q).float()).abs().max())
        ttnn.deallocate(o_q)
    ttnn.deallocate(o_stock)
    case["ms_stock"], case["ms_qsplit"] = round(ms_stock, 3), ms_q and round(ms_q, 3)
    case["speedup"] = ms_q and round(ms_stock / ms_q, 4)
    res["cases"].append(case)
    print(json.dumps(case), flush=True)
    for t in (q, k, v, bias):
        ttnn.deallocate(t)
    json.dump(res, open(OUT, "w"), indent=1)
ttnn.close_device(dev)
