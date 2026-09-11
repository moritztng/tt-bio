"""Score the fold-level A/B/A2 arms straight off the artifacts on disk.

Independent of fold_arm.py's report: sha256 the CIF the fold actually wrote, read the model's
own accuracy number out of its results.json, and take the E6 counter from the per-pid dumps the
spawn children left. A2 is the A/A floor, same card, same session.
"""
import glob, hashlib, json, os, sys

WT = "/home/ttuser/.coworker/wt/b2x-trimul-e6-crossmodel-parity"
OUT = f"{WT}/perf/b2x_crossmodel/out_qb2"


def plddt(res):
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if "plddt" in k.lower() and isinstance(v, (int, float)):
                    yield k, v
                else:
                    yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
    return dict(walk(res))


rows = []
for model in sorted({os.path.basename(d).rsplit("_", 1)[0]
                     for d in glob.glob(f"{WT}/foldout/*_A") + glob.glob(f"{WT}/foldout/*_B")}):
    for arm in ("A", "B", "A2"):
        d = f"{WT}/foldout/{model}_{arm}"
        cifs = sorted(glob.glob(f"{d}/**/*.cif", recursive=True))
        if not cifs:
            continue
        rj = sorted(glob.glob(f"{d}/**/results.json", recursive=True))
        res = json.load(open(rj[0])) if rj else {}
        e6 = 0
        for f in glob.glob(f"{OUT}/stats_{model}_{arm}/*.json"):
            e6 += json.load(open(f)).get("gated_moves", 0)
        rows.append({
            "model": model, "arm": arm, "flag": arm == "B",
            "e6_moves": e6,
            "cif_sha256": {os.path.relpath(c, d): hashlib.sha256(open(c, "rb").read()).hexdigest()
                           for c in cifs},
            "accuracy": plddt(res),
        })
        r = rows[-1]
        print(f"{model:12s} {arm:2s} flag={int(r['flag'])} e6={e6:4d} "
              f"sha={list(r['cif_sha256'].values())[0][:16]} acc={r['accuracy']}")

verdict = {}
for model in sorted({r["model"] for r in rows}):
    per = {r["arm"]: r for r in rows if r["model"] == model}
    if not {"A", "B"} <= set(per):
        verdict[model] = "INCOMPLETE"
        continue
    sh = lambda a: tuple(sorted(per[a]["cif_sha256"].items()))
    floor = sh("A") == sh("A2") if "A2" in per else None
    verdict[model] = {
        "A_eq_B": sh("A") == sh("B"),
        "A_A2_floor_bit_exact": floor,
        "e6_A": per["A"]["e6_moves"], "e6_B": per["B"]["e6_moves"],
        "non_vacuous": per["B"]["e6_moves"] > 0 and per["A"]["e6_moves"] == 0,
    }
print()
print(json.dumps(verdict, indent=2))
json.dump({"rows": rows, "verdict": verdict}, open(f"{OUT}/fold_scored.json", "w"), indent=2)
print(f"wrote {OUT}/fold_scored.json")
