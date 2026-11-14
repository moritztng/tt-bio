# triangle attention block, add_to_input as the Pairformer calls it: triatt_ln's head-pitched q, k, v, g
# (triatt_ln.HEAD_PITCH, read in place by the fused SDPA and the one-program tail) vs the plain MT pitch.
# Output must be torch.equal. hp_ab.py OUT.jsonl S:c:H:ending ...  (on a p300 through p300run.py)
import json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(W), str(W / "perf"), str(W / "perf/mol1_triattn")]
out, cells = Path(sys.argv[1]), [tuple(map(int, c.split(":"))) for c in sys.argv[2:]]
sys.argv = [sys.argv[0]]
import torch, ttnn
import bench as B
import clocksample
import tt_bio.tenstorrent as T, tt_bio.triatt_ln as TL, tt_bio.triatt_qkv as TQ, tt_bio.triatt_sdpa as TS
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
        for arm in ("0", "1", "0", "1"):
            TL.HEAD_PITCH = {"wormhole": arm == "1", "blackhole": arm == "1"}
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            st0 = (TL.STATS[3], T.UNPITCH_STATS[0], TS.STATS[0], TQ.GOP_STATS[0])
            y = ttnn.to_torch(mod(xt, add_to_input=True))
            rec[f"stats{arm}"] = [a - b for a, b in zip((TL.STATS[3], T.UNPITCH_STATS[0], TS.STATS[0], TQ.GOP_STATS[0]), st0)]
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
        rec["equal"] = bool(torch.equal(got["0"], got["1"]))
        rec["max_abs"] = float((got["0"].float() - got["1"].float()).abs().max())
        del mod
    print(json.dumps(rec), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
T.cleanup()
