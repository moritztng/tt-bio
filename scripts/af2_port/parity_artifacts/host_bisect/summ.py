import json, sys
for f in sys.argv[1:]:
    try:
        t = open(f).read(); d = json.loads(t[t.index("{"):])
    except Exception as e:
        print(f"{f}: UNREADABLE {e}"); continue
    print("%-34s taps_failed %3d  scalars %2d  pcc_min %.10f  env_ratio %12.6f  tmpl_host %s  served %s" % (
        f.rsplit("/",1)[-1].replace(".json",""), d["taps_failed"], d["scalars_failed"], d["pcc_min"],
        d["envelope_ratio_max"], d.get("template_host"),
        (d.get("triatt_fused_stats") or {}).get("stats", {}).get("served", "-")))
