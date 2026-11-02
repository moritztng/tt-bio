# ending triangle attention, add_to_input as the Pairformer calls it: the fused LN reading x^T with up to G column
# groups (triatt_ln.TR_GROUPS) vs the transpose-first path (G 1 at Mol-1 widths). Output must be torch.equal.
# tr_ab.py OUT.jsonl G S:c:H ...  (on a p300 open through tt_bio.main.ensure_p300_mesh_descriptor first)
import json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(W), str(W / "perf"), str(W / "perf/mol1_triattn")]
out, G, cells = Path(sys.argv[1]), sys.argv[2], [tuple(map(int, c.split(":"))) for c in sys.argv[3:]]
sys.argv = [sys.argv[0]]
import torch, ttnn
import bench as B
import clocksample
import tt_bio.tenstorrent as T, tt_bio.triatt_ln as TL
dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
T._MM_DEFAULT_WIDE = B.ARMS["served"][1]
for S, c, H in cells:
    sd = B.weights(c, H, 7)
    torch.manual_seed(S + c)
    x = torch.randn(1, S, S, c).bfloat16()
    rec = {"S": S, "c": c, "H": H, "G": int(G)}
    with T.levers(B.ARMS["served"][0]):
        mod = T.TriangleAttention(32, H, True, sd, ckc, softmax_site="pairformer")
        got = {}
        for arm in ("1", G, "1", G):
            TL.TR_GROUPS = {"wormhole": int(arm), "blackhole": int(arm)}
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            TL.REJECTS.clear()
            y = ttnn.to_torch(mod(xt, add_to_input=True))
            rec[f"rejects{arm}"] = sorted({k[0] for k in TL.REJECTS})
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
        rec["equal"] = bool(torch.equal(got["1"], got[G]))
        rec["max_abs"] = float((got["1"].float() - got[G].float()).abs().max())
        del mod
    print(json.dumps(rec), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
T.cleanup()
