# ending triangle attention, add_to_input as the Pairformer calls it: TRANS tail (R-row units) vs the
# two-program path (tail + pair_transpose_add). Output must be torch.equal; time per call, AICLK during.
# tail_t_ab.py OUT.jsonl S:c:H ...  (on a p300 open through tt_bio.main.ensure_p300_mesh_descriptor first)
import json, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(W), str(W / "perf"), str(W / "perf/mol1_triattn")]
out, cells = Path(sys.argv[1]), [tuple(map(int, c.split(":"))) for c in sys.argv[2:]]
sys.argv = [sys.argv[0]]
import torch, ttnn
import bench as B
import clocksample
import tt_bio.tenstorrent as T, tt_bio.triatt_qkv as Q
dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
T._MM_DEFAULT_WIDE = B.ARMS["served"][1]
for S, c, H in cells:
    sd = B.weights(c, H, 7)
    torch.manual_seed(S + c)
    x = torch.randn(1, S, S, c).bfloat16()
    rec = {"S": S, "c": c, "H": H}
    with T.levers(B.ARMS["served"][0]):
        mod = T.TriangleAttention(32, H, True, sd, ckc, softmax_site="pairformer")
        got = {}
        for arm in ("0", "1", "0", "1"):
            Q.TAIL_T = arm
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            g0 = Q.GOP_STATS[0]
            y = ttnn.to_torch(mod(xt, add_to_input=True))
            rec[f"engaged{arm}"] = Q.GOP_STATS[0] - g0
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
        rec["rows"] = Q._TAIL_T_LAST if hasattr(Q, "_TAIL_T_LAST") else None
        rec["equal"] = bool(torch.equal(got["0"], got["1"]))
        d = (got["0"].float() - got["1"].float()).abs()
        rec["max_abs"] = float(d.max())
        rec["finite"] = bool(torch.isfinite(got["1"]).all())
        del mod
    print(json.dumps(rec), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
T.cleanup()
