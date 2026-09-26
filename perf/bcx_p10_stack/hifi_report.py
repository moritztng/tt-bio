import glob, statistics as st, sys, json
sys.path.insert(0, 'perf/bcx_p10_stack')
import compare as C

ARMS = [("h_off", "(0,0,off)"), ("h_agtri", "(1,1,agtri)"), ("h_hifi", "(1,1,hifi)")]
res = {}
for tag, name in ARMS:
    rows, reach, clk = [], [], []
    for f in sorted(glob.glob('perf/bcx_p10_stack/out/c*_%s/round_events.json' % tag)):
        r, stamp = C.rounds_of(f)
        rows += r
        fused = stamp.get("fused_hifi_stats") or {}
        ent = (stamp.get("kernel_entry_stats") or {}).get("tri_att_sdpa_hifi")
        reach.append({"msa": sum(stamp.get("extra_msa_swapped") or []),
                      "tmpl": (stamp.get("template_calls") or {}).get("taped", 0),
                      "agtri_served": (stamp.get("triatt_sdpa_stats") or {}).get("served", 0),
                      "hifi_served": fused.get("served", 0), "hifi_taped": fused.get("taped", 0),
                      "hifi_declined": fused.get("declined", 0), "entry": ent,
                      "fp32_softmax": stamp.get("fp32_softmax_calls"),
                      "rounds": len(r)})
    res[name] = {"n": len(rows),
                 "wall": st.median([r["wall"] for r in rows]),
                 "host": st.median([r["host"] for r in rows]),
                 "dev": st.median([r["dev"] for r in rows]),
                 "lo": min(r["wall"] for r in rows), "hi": max(r["wall"] for r in rows),
                 "aiclk": sorted(r["aiclk"] for r in rows)[len(rows)//2],
                 "aiclk_min": min(r["aiclk"] for r in rows),
                 "load1": st.median([r["load1"] for r in rows]), "reach": reach}

base = res["(0,0,off)"]["wall"]
ag = res["(1,1,agtri)"]
print("One sitting, qb1 card 0, arms alternating at the process boundary, 2 cycles of 6.\n")
print(f"{'arm':<14}{'n':>3}{'round s':>10}{'min':>8}{'max':>8}{'host':>8}{'dev':>8}"
      f"{'x vs off':>10}{'x vs agtri':>12}{'AICLK':>7}{'load1':>7}")
for _, name in ARMS:
    r = res[name]
    print(f"{name:<14}{r['n']:>3}{r['wall']:>10.3f}{r['lo']:>8.3f}{r['hi']:>8.3f}"
          f"{r['host']:>8.3f}{r['dev']:>8.3f}{base/r['wall']:>10.3f}"
          f"{ag['wall']/r['wall']:>12.3f}{r['aiclk']:>7}{r['load1']:>7.1f}")
print("\ndevice column: off %.3f  agtri %.3f  hifi %.3f   agtri->hifi %.3fx"
      % (res["(0,0,off)"]["dev"], ag["dev"], res["(1,1,hifi)"]["dev"],
         ag["dev"] / res["(1,1,hifi)"]["dev"]))
print("AICLK min across every round, every arm: %d"
      % min(res[n]["aiclk_min"] for _, n in ARMS))
print("\nreach, per arm process:")
for _, name in ARMS:
    for r in res[name]["reach"]:
        print(" ", name, r)
json.dump(res, open('perf/bcx_p10_stack/out/hifi_report.json', 'w'), indent=1)
