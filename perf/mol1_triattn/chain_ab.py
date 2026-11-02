# triangle attention block, add_to_input as the Pairformer calls it: the fused SDPA with its K/V forwarding chain
# along each head's q cores for every batch
# (triatt_sdpa.KV_CHAIN, arm 1) vs every core reading K/V from DRAM (arm 0), on the served
# (head-pitched) path. Output must be torch.equal. chain_ab.py OUT.jsonl ARMS S:c:H:ending ...  (ARMS e.g. 01)
import json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(W), str(W / "perf"), str(W / "perf/mol1_triattn")]
out, ARMS, cells = Path(sys.argv[1]), sys.argv[2], [tuple(map(int, c.split(":"))) for c in sys.argv[3:]]
sys.argv = [sys.argv[0]]
import torch, ttnn
import bench as B
import clocksample
import tt_bio.sdpa_generic as SG, tt_bio.tenstorrent as T, tt_bio.triatt_ln as TL, tt_bio.triatt_qkv as TQ, tt_bio.triatt_sdpa as TS
dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
T._MM_DEFAULT_WIDE = B.ARMS["served"][1]
for S, c, H, end in cells:
    sd = B.weights(c, H, 7)
    torch.manual_seed(S + c + end)
    x = torch.randn(1, S, S, c).bfloat16()
    rec = {"S": S, "c": c, "H": H, "ending": end}
    with T.levers(B.ARMS["served"][0]):
        mod = T.TriangleAttention(32, H, bool(end), sd, ckc, softmax_site="pairformer")
        got = {}
        for arm in ARMS + ARMS:
            TS.KV_CHAIN = dict.fromkeys(TS.KV_CHAIN, arm == "1")
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            st0 = (TL.STATS[3], T.UNPITCH_STATS[0], TS.STATS[0], TQ.GOP_STATS[0], SG.CHAIN_BUILDS[0])
            y = ttnn.to_torch(mod(xt, add_to_input=True))
            rec[f"stats{arm}"] = [a - b for a, b in zip((TL.STATS[3], T.UNPITCH_STATS[0], TS.STATS[0], TQ.GOP_STATS[0], SG.CHAIN_BUILDS[0]), st0)]
            if arm in got:
                rec[f"repeat_equal{arm}"] = bool(torch.equal(got[arm], y))
            got[arm] = y
            ts = []
            with clocksample.during(period=0.5) as clk:
                for _ in range(7):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    mod(xt, add_to_input=True)
                    ttnn.synchronize_device(dev)
                    ts.append(time.perf_counter() - t0)
            ttnn.deallocate(xt)
            rec.setdefault(f"ms{arm}", []).append(round(sorted(ts)[3] * 1e3, 3))
            rec.setdefault(f"aiclk{arm}", []).append(clk.summary())
        rec["equal"] = all(torch.equal(got[ARMS[0]], got[a]) for a in ARMS)
        rec["max_abs"] = max(float((got[ARMS[0]].float() - got[a].float()).abs().max()) for a in ARMS)
        del mod
    print(json.dumps(rec), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
T.cleanup()
