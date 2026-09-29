import json, glob, statistics, os
os.chdir("/home/ttuser/.coworker/wt/train-w-fourchip")
out = {"session": "pass1 2026-09-19", "host": "qb1", "chips_order": [1, 2, 3, 0],
       "recipe": {"micro": 4, "global_batch": 64, "tokens": 256, "blocks": 8, "seed": 0,
                  "data": "sabdab/train", "steps": 4, "torch_threads": "unset (torch default 16 per rank)"},
       "arms": {}}
for d in sorted(glob.glob("runs/w-ladder/w*-r*")):
    tag = os.path.basename(d)
    a = {"rows": {}, "prov": {}}
    for f in sorted(glob.glob(d + "/history-rank*.jsonl")):
        r = int(f.split("rank")[-1].split(".")[0])
        a["rows"][r] = [json.loads(l) for l in open(f) if l.strip()]
    for f in sorted(glob.glob(d + "/provenance-rank*.json")):
        r = int(f.split("rank")[-1].split(".")[0])
        p = json.loads(open(f).read())
        a["prov"][r] = {"nodes": p.get("device_nodes"), "aiclk": p.get("aiclk"),
                        "cotenancy": p.get("config", {}).get("cotenancy", {})}
    out["arms"][tag] = a
json.dump(out, open("perf/train_w_fourchip/ladder_pass1_raw.json", "w"), indent=1)

# the settled summary
print(f"{'arm':>7} {'micros':>7} {'wall':>9} {'data':>7} {'cadence':>9} {'fwd':>7} {'devbwd':>7} "
      f"{'losses':>9} {'hostbwd':>9} {'opt':>7} {'reduce':>7} {'MB':>8} {'nodes':>8} {'AICLK':>6}")
summ = {}
for tag, a in out["arms"].items():
    for r, rows in a["rows"].items():
        s = rows[1:] or rows          # settled; w4 has only its first step
        med = lambda k: round(statistics.median([x.get(k) or 0.0 for x in s]), 3)
        stg = lambda k: round(statistics.median([x["stages"].get(k, 0.0) for x in s]), 3)
        line = dict(arm=f"{tag}/r{r}", micros=s[0]["stages"]["micro_batches"], wall=med("wall"),
                    data=med("data"), cadence=round(med("wall") + med("data") + med("outer"), 3),
                    fwd=stg("forward"), devbwd=stg("device_backward"), losses=stg("losses"),
                    hostbwd=stg("host_backward"), opt=stg("optimizer"),
                    reduce=med("reduce_s"), mb=med("reduce_mb"),
                    nodes=a["prov"].get(r, {}).get("nodes"),
                    aiclk=(a["prov"].get(r, {}).get("aiclk") or {}).get("median"),
                    settled=len(rows) > 1)
        summ[line["arm"]] = line
        print(f"{line['arm']:>7} {line['micros']:>7} {line['wall']:>9.3f} {line['data']:>7.3f} "
              f"{line['cadence']:>9.3f} {line['fwd']:>7.3f} {line['devbwd']:>7.3f} "
              f"{line['losses']:>9.3f} {line['hostbwd']:>9.3f} {line['opt']:>7.3f} "
              f"{line['reduce']:>7.3f} {line['mb']:>8.2f} {str(line['nodes']):>8} "
              f"{str(line['aiclk']):>6}{'' if line['settled'] else '   FIRST STEP ONLY'}")
json.dump(summ, open("perf/train_w_fourchip/ladder_pass1_summary.json", "w"), indent=1)
