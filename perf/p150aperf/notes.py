import json
d=json.load(open("/home/ttuser/.coworker/wt/perf-excellence-p150a/docs/perf_baselines.json"))
mach=d["cards"]["p150a"]["machines"]["tt-quietbox"]
print("BLOCK date",mach.get("date"),"ver",mach.get("tt_bio_version"))
print("BLOCK note:",mach.get("note","")[:600])
for m in ("protenix-v2","boltz2","boltz2-affinity","openfold3","esmc-600m","opendde"):
    e=mach["models"][m]
    print("\n==",m,e["value"],e["unit"],"lat",e.get("latency_ms"),e.get("date"),e.get("tt_bio_version"))
    print("  note:",e.get("note","")[:700])
