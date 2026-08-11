"""C2 (bfloat8_b triangle bias) x C1 (wide q_chunk): does the stack exist, and what does it cost
in accuracy? Op-level, on device, at the three production shapes.

PREDICTIONS, written before the run (perf/triatt_root/c2_stack_qb2c1.json records the outcome):
  P1  The b8 path runs at 320, 512 and 576. ttnn.typecast to bfloat8_b needs TILE layout, which the
      bias already has, and the SDPA "mask must be in DRAM" validation is about buffer type, not
      dtype -- so nothing in state doc section 8.4's refusal applies here.
  P2  b8 is NOT torch.equal to bf16 (it is a lossy cast) but is close on the op: rmsd/std about
      0.0025 and PCC >= 0.9999 at 512. Section 6 measured 0.002547 / 1.000000 for exactly this.
      wideq is torch.equal to narrowq at the same bias dtype -- C1's own bit-exactness must survive
      the b8 cast, since the cast happens before the chunking.
  P3  Timing at 512, against the shipped (narrow, bf16) = 7.1298 ms of section 6: wide/bf16 ~1.08x,
      wide/b8 ~1.66x (section 6's L2 row, 4.2947 ms). narrow/b8 is UNMEASURED anywhere; the two
      levers attack the same bias re-read, so they must stack sub-multiplicatively -- if narrow/b8
      lands near 1.53x (= 1.66/1.084) the stack is multiplicative and section 6's L2 row is
      consistent; if it lands near 1.66x on its own then C1 buys nothing once C2 is in and the
      proposal changes shape. This is the number this run exists to produce.
"""
import json, sys, time
import torch, ttnn
import tt_bio.tenstorrent as T

OUT = sys.argv[1]
SEQS = (320, 512, 576)
ARMS = (("narrowq", False, False), ("wideq", True, False),
        ("narrowq_b8", False, True), ("wideq_b8", True, True))
REPEAT = 5
res = {"ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
       "host": "qb2", "card": 1, "predictions": __doc__, "rows": []}

dev = ttnn.open_device(device_id=0)
T.COMPUTE_GRID_MAIN = dev.compute_with_storage_grid_size()
print("grid %dx%d  ttnn %s" % (T.COMPUTE_GRID_MAIN.x, T.COMPUTE_GRID_MAIN.y, res["ttnn"]), flush=True)


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


for s in SEQS:
    torch.manual_seed(0)
    tq, tk, tv = (torch.randn(s, 8, s, 32) * 0.1 for _ in range(3))
    tb = torch.randn(1, 8, s, s) * 0.1
    q, k, v = (ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
               for x in (tq, tk, tv))
    bias = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    scale = 32 ** -0.5
    ref = None
    for name, wide, b8 in ARMS:
        T._SDPA_WIDE_Q, T._TRIATT_BIAS_B8 = wide, b8
        T._tri_att_q_chunks.cache_clear()
        cand = T._tri_att_q_chunks(s, s)
        try:
            out = T._tri_att_sdpa(q, k, v, bias, scale)          # warm: pays JIT + any L1 throw
            ttnn.synchronize_device(dev)
            got = ttnn.to_torch(out)
            ttnn.deallocate(out)
            ts = []
            for _ in range(REPEAT):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                o = T._tri_att_sdpa(q, k, v, bias, scale)
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(o)
        except Exception as e:                                    # noqa: BLE001
            res["rows"].append({"seq": s, "arm": name, "error": f"{type(e).__name__}: {e}"[:400]})
            print("  seq %4d %-11s ERROR %s" % (s, name, str(e)[:200]), flush=True)
            continue
        if ref is None:
            ref = got
        d = (got.float() - ref.float())
        used = [c for c in cand if (s, s, c) not in T._SDPA_Q_CHUNK_OVER_L1][0]
        row = {"seq": s, "arm": name, "wide": wide, "b8": b8, "q_chunk": used,
               "k_chunk": T._sdpa_chunks_shipped(s, s)[1], "candidates": list(cand),
               "ms_min": round(min(ts), 4), "ms_median": round(sorted(ts)[len(ts) // 2], 4),
               "ms_all": [round(x, 4) for x in ts],
               "torch_equal_vs_shipped": bool(torch.equal(got, ref)),
               "rmsd_over_std": round(float(d.pow(2).mean().sqrt() / ref.float().std()), 6),
               "pcc_vs_shipped": round(pcc(got, ref), 6),
               "over_l1": sorted(list(x) for x in T._SDPA_Q_CHUNK_OVER_L1 if x[0] == s)}
        res["rows"].append(row)
        print("  seq %4d %-11s q%-4d min %8.4f ms  eq %-5s  rmsd/std %.6f  pcc %.6f" %
              (s, name, used, row["ms_min"], row["torch_equal_vs_shipped"],
               row["rmsd_over_std"], row["pcc_vs_shipped"]), flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    for t in (q, k, v, bias):
        ttnn.deallocate(t)

base = {r["seq"]: r["ms_min"] for r in res["rows"] if r.get("arm") == "narrowq"}
res["speedup_vs_shipped"] = {
    f'{r["seq"]}|{r["arm"]}': round(base[r["seq"]] / r["ms_min"], 4)
    for r in res["rows"] if "ms_min" in r and r["seq"] in base}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["speedup_vs_shipped"], indent=1), flush=True)
ttnn.close_device(dev)
print("wrote", OUT, flush=True)
