import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "perf/tokenbucket/od512_paired.json"
d = json.load(open(path))
a, w = d["ab"], d["warm_folds"]
med, mean = a["paired_delta_median_s"], a["paired_delta_mean_s"]
loads = [float(f["loadavg"][0]) for f in w]
spans = {k: round(max(v["warm_times_s"]) - min(v["warm_times_s"]), 3)
         for k, v in a["arms"].items()}
ok = max(loads) <= 7.0 and (med >= 0) == (mean >= 0) and max(spans.values()) < 0.6
print(f"median {med:+.3f} s  mean {mean:+.4f} s  deltas {a['paired_delta_s']}")
print(f"loadavg {min(loads)}-{max(loads)}  per-arm span {spans}")
print("ACCEPTED" if ok else "REJECTED: noise-dominated, do not decide on this run")
if ok:
    verb = "KEEPS" if med <= 0.167 else "LOSES"
    print(f"margin 0.167 s -> OpenDDE {verb} the beat-DGX-H200 bar")
