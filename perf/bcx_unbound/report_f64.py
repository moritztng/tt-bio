import json
d = json.load(open("perf/bcx_unbound/primal_f64.json"))
for k in ("f64_blocks_graded", "f64_block_s", "f64_s", "bf16_s", "device_s", "prefix_only"):
    print(f"{k} = {d.get(k)}")
print("aiclk_during =", json.dumps(d.get("aiclk_during")))
for arm in ("device_vs_f64", "host_bf16_vs_f64"):
    for t in ("msa", "pair"):
        v = d[arm][t]
        print(f"{arm:18s} {t:5s} rel_l2 {v['rel_l2']:.6f}  cos {v['cos']:.6f}")
s = d.get("stamp", {})
print("stamp:", s.get("host"), "card", s.get("card"), "commit", str(s.get("commit"))[:9],
      "dirty", s.get("tt_bio_dirty"), "threads", s.get("torch_threads"),
      "loadavg", s.get("loadavg"), s.get("utc"))
