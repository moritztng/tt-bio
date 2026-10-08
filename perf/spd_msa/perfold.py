"""Per-fold top-pose DockQ/LRMSD for every rep under the given run dirs (uses grade.py's scorer + cache)."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spd"))
import grade
data = Path("~/spd-data").expanduser()
cf = data / "grade_cache.json"
cache = json.loads(cf.read_text())
rows = []
for d in map(Path, sys.argv[1:]):
    for f in d.rglob("bench.jsonl"):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if r.get("ev") != "rep" or r["err"] or not r.get("samples_conf"):
                continue
            sd = Path(r["struct_dir"]); sd = sd if sd.exists() else f.parent / sd.name
            cif = sd / f"{r['input']}.cif"
            if str(cif) not in cache:
                cache[str(cif)] = grade.score_one((str(cif), str(data / "ref" / f"{r['input']}.cif"), r["input"]))[1]
            s = cache[str(cif)]
            rows.append((r["input"], r["arm"].split(":")[0], r["seed"], s["dockq"], s["lrmsd"], s["irmsd"],
                         r["samples_conf"][0].get("iptm"), str(f.parent)))
cf.write_text(json.dumps(cache))
for x in sorted(rows):
    print(f"{x[0]} {x[1]:6s} s{x[2]} dockq {x[3]:.3f} lrmsd {x[4]:7.2f} irmsd {x[5]:6.2f} iptm {x[6]:.3f}  {x[7]}")
