"""Grade MOL1 predictions with FoldBench's own evaluator and write state/mol1/QUALITY.json.

    <ost env>/bin/python grade.py --inputs INPUTS --cases CASES.json --work WORK --out QUALITY.json \
        --arm gpu=DIR --arm gpu_self=DIR --arm tt_bh=DIR [--arm tt_wh=DIR] [--headline DIR]

INPUTS is the frozen tree quality_inputs.py wrote (truth/ holds <pdb>-assembly1.cif). Every arm DIR has upstream's
layout, <case>/seed_<s>/predictions/<case>_sample_<k>.cif + <case>_summary_confidence_sample_<k>.json, case names
as quality_inputs.safe() spells them. Per arm, each case's CIFs go through FoldBench's Protenix postprocess, then
foldbench/evaluation (OpenStructure compare-structures / compare-ligand-structures, DockQ v2 for nucleic-acid
interfaces), exactly as FoldBench's evaluate.py runs it. Cases outside FoldBench (pfm 11-set, upstream examples)
are scored as whole complexes: global lDDT and the mean DockQ over every interface OST finds.
Rows (top-ranked sample per case and seed; confidences averaged over its samples) go to stats.py.
"""
import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "foldbench")]
import stats  # noqa: E402
from quality_inputs import safe  # noqa: E402

FB_TARGETS = HERE / "foldbench_targets"
NA_TASKS = ("interface_protein_dna", "interface_protein_rna")


def postprocess(src, dst):
    """FoldBench algorithms/Protenix/postprocess.py process_file: occupancy 1, B 0, entity type from group_PDB;
    entity_poly_seq rebuilt from atom_site when the writer left it out (metadata only, coordinates untouched)."""
    import biotite.structure.io.pdbx as pdbx
    f = pdbx.CIFFile.read(str(src)); b = f.block; a = b.get("atom_site")
    n = len(a["group_PDB"])
    a["occupancy"] = pdbx.CIFColumn(pdbx.CIFData(["1"] * n))
    a["B_iso_or_equiv"] = pdbx.CIFColumn(pdbx.CIFData(["0"] * n))
    ent = {}
    for e, g in zip(a["label_entity_id"].as_array().tolist(), a["group_PDB"].as_array().tolist()):
        if e not in ent:
            ent[e] = ("polymer" if g == "ATOM" else "non-polymer", g)
        elif ent[e][1] != g:
            ent[e] = ("polymer", g)
    b["entity"] = pdbx.CIFCategory({"id": list(ent), "type": [v[0] for v in ent.values()]})
    if "entity_poly_seq" not in b:  # tt-bio writes atom_site only; DockQv2 finds polymer chains through this category
        seq = {}
        for g, e, n, m in zip(*(a[k].as_array().tolist() for k in ("group_PDB", "label_entity_id", "label_seq_id",
                                                                   "label_comp_id"))):
            if g == "ATOM":
                seq.setdefault((e, n), m)
        b["entity_poly_seq"] = pdbx.CIFCategory({"entity_id": [k[0] for k in seq], "hetero": ["n"] * len(seq),
                                                 "mon_id": list(seq.values()), "num": [k[1] for k in seq]})
    for k in list(b):  # biotite 0.40 writes a still-lazy category without its newline ("#loop_"); OST then rejects it
        b[k] = b[k]
    f.write(str(dst))


def predictions(arm_dir, case, work):
    """[(seed, sample, ranking_score, postprocessed cif, summary dict)] for one case of one arm."""
    name = safe(case); out = []
    # upstream writes <dump>/<case>/seed_*; tt-bio predict writes <out>/mol1_results_<case>/<case>/seed_*
    hits = [*(arm_dir / name).glob("seed_*/predictions/*_sample_*.cif"),
            *(arm_dir / f"mol1_results_{name}" / name).glob("seed_*/predictions/*_sample_*.cif")]
    for cif in sorted(hits):
        seed = int(cif.parts[-3].split("_")[1]); k = int(re.search(r"_sample_(\d+)\.cif$", cif.name).group(1))
        summ = json.loads(cif.with_name(f"{name}_summary_confidence_sample_{k}.json").read_text())
        dst = work / "pp" / name / f"{name}_{seed}_{k}.cif"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            postprocess(cif, dst)
        out.append((seed, k, float(summ.get("ranking_score", 0)), dst, summ))
    return out


def targets(case):
    """(FoldBench task, [target rows]) for a case; non-FoldBench cases get one whole-complex row."""
    base = case.get("base", case["id"])
    if case["source"].startswith("FoldBench/"):
        task = case["source"].split("/", 1)[1]
        rows = [r for r in csv.DictReader(open(FB_TARGETS / f"{task}.csv")) if r["pdb_id"] == base]
        return task, rows
    if not case.get("truth"):
        return None, []
    return "whole", [{"pdb_id": case["truth"].split(":", 1)[1].split("-")[0] + "-assembly1", "chain_id": "A"}]


def evaluate(arm, arm_dir, cases, inputs, work):
    """Run FoldBench's evaluator for one arm; returns stats rows."""
    from evaluation.eval_by_ost import eval_by_ost
    from evaluation.eval_by_dockqv2 import eval_by_dockqv2
    rows = []
    by_task = defaultdict(list)  # (task, variant) -> [(case, target row, preds)]
    for c in cases:
        task, trows = targets(c)
        preds = predictions(arm_dir, c["id"], work / arm)
        if not preds:
            rows.append({"arm": arm, "case": c["id"], "set": c["set"], "missing": True})
            continue
        variant = c["id"].split(":", 1)[1] if ":" in c["id"] and not c["id"].startswith(("pfm:", "upstream:")) else ""
        for t in trows:
            by_task[(task, variant)].append((c, t, preds))
    for (task, variant), items in by_task.items():
        ev = work / arm / f"eval{('.' + variant) if variant else ''}" / task
        (ev / "raw").mkdir(parents=True, exist_ok=True)
        recs = [dict(t, seed=s, sample=k, ranking_score=r, prediction_path=str(p))
                for c, t, preds in items for s, k, r, p, _ in preds]
        df = pd.DataFrame(recs)
        ost_task = "monomer_protein" if task == "whole" else task
        for _ in range(2):  # FoldBench waits 20 s per future; the second call collects the slow ones (outputs cached)
            eval_by_ost(df, ost_task, str(ev), str(inputs / "truth"))
            if ost_task != "interface_protein_ligand":
                rigid_rescue(df, ev, inputs / "truth")
        scored = pd.read_csv(ev / "raw" / f"{ost_task}_ost.csv")
        if task in NA_TASKS:
            eval_by_dockqv2(df, task, str(ev), str(inputs / "truth"))
            dq = pd.read_csv(ev / "raw" / f"{task}_dockqv2.csv")[["pdb_id", "interface_chain_id_1",
                                                                     "interface_chain_id_2", "seed", "sample",
                                                                     "dockq_score"]]
            scored = scored.drop(columns=["dockq_score"]).merge(
                dq, on=["pdb_id", "interface_chain_id_1", "interface_chain_id_2", "seed", "sample"], how="left")
        for c, t, preds in items:
            key = c["id"] + "|" + "|".join(t.get(k, "") for k in ("interface_chain_id_1", "interface_chain_id_2",
                                                                 "native_chain_id_1", "native_chain_id_2"))
            sc = scored[scored.pdb_id == t["pdb_id"]]
            for k in ("interface_chain_id_1", "native_chain_id_1"):
                if k in t:
                    sc = sc[(sc[k] == t[k]) & (sc[k.replace("_1", "_2")] == t[k.replace("_1", "_2")])]
            for seed in sorted({p[0] for p in preds}):
                ps = [p for p in preds if p[0] == seed]
                top = max(ps, key=lambda p: p[2])
                hit = sc[(sc.seed == seed) & (sc["sample"] == top[1])]
                row = {"arm": arm, "case": key, "set": c["set"], "seed": seed, "ranking_score": top[2],
                       "plddt": float(np.mean([p[4]["plddt"] for p in ps])) / 100.0,
                       "ptm": float(np.mean([p[4]["ptm"] for p in ps])),
                       "iptm": float(np.mean([p[4].get("iptm", np.nan) for p in ps]))}
                h = hit.iloc[0].to_dict() if len(hit) else {}
                if task == "interface_protein_ligand":
                    row.update(lig_rmsd=h.get("rmsd"), lddt_pli=h.get("lddt-pli"))
                elif task == "whole":
                    row.update(lddt=h.get("lddt"), dockq=whole_dockq(ev, t["pdb_id"], seed, top[1]))
                elif task.startswith("monomer"):
                    row.update(lddt=h.get("lddt"))
                else:
                    row.update(dockq=h.get("dockq_score"), lddt=h.get("lddt"))
                rows.append({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()})
    return rows


def rigid_rescue(df, ev, truth):
    """OST 2.8 --rigid-scores raises 'at least 3 points are required for superposition' on some targets (8c5h
    against itself) and the whole call fails, DockQ and lDDT included. Rerun exactly those without it; RMSD/GDT are
    not graded and the other scores do not depend on the superposition."""
    import subprocess
    from evaluation.eval_by_ost import OST_COMPARE_STRUCTURE
    for r in df.itertuples():
        out = ev / "detail" / f"{r.pdb_id}_{r.seed}_{r.sample}_structure_ost.json"
        try:
            if "required for superposition" not in json.loads(out.read_text()).get("traceback", ""):
                continue
        except (OSError, ValueError):
            continue
        subprocess.run(OST_COMPARE_STRUCTURE.replace("--rigid-scores ", "").format(
            model_file=r.prediction_path, reference_file=truth / f"{r.pdb_id}.cif", output_path=out),
            shell=True, executable="/bin/bash", capture_output=True, check=False)


def whole_dockq(ev, pdb_id, seed, sample):
    p = ev / "detail" / f"{pdb_id}_{seed}_{sample}_structure_ost.json"
    try:
        d = json.loads(p.read_text())
        return float(np.mean(d["dockq"])) if d.get("dockq") else None
    except (OSError, ValueError, KeyError):
        return None


def headline(arm_dir, spec, inputs, work):
    """FoldBench-AB top-ranked success over the five-seed budget, with a bootstrap interval over interfaces."""
    cases = [dict(id=h["id"], set="headline", source="FoldBench/interface_antibody_antigen") for h in spec]
    rows = evaluate("headline", arm_dir, cases, inputs, work)
    s = stats.success(rows, "headline")
    if not s:
        return None
    m, lo, hi = stats.boot_mean(list(s.values()), np.random.default_rng(stats.SEED))
    return {"success_rate": m, "ci95": [lo, hi], "n_interfaces": len(s), "molence_claim": 0.750,
            "reproduced": lo <= 0.750 <= hi, "missing": sorted({r["case"] for r in rows if r.get("missing")})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arm", action="append", default=[], help="name=DIR (gpu, gpu_self, tt_bh, tt_wh)")
    ap.add_argument("--headline", type=Path)
    a = ap.parse_args()
    spec = json.loads(a.cases.read_text())
    arms = dict(x.split("=", 1) for x in a.arm)
    rows = []
    for arm, d in arms.items():
        rows += evaluate(arm, Path(d), spec["paired"], a.inputs, a.work)
    (a.work / "rows.json").write_text(json.dumps(rows))
    ok = [r for r in rows if not r.get("missing")]
    sets = sorted({c["set"] for c in spec["paired"]})
    per_arch, per_set = {}, {}
    for test, arch in (("tt_bh", "bh"), ("tt_wh", "wh")):
        if test not in arms:
            continue
        missing = sorted({r["case"] for r in rows if r["arm"] == test and r.get("missing")})
        v = stats.verdict(ok, test)
        per_set[arch] = {s: stats.verdict(ok, test, cases={r["case"] for r in ok if r["set"] == s}) for s in sets}
        failed_sets = [s for s, x in per_set[arch].items() if x["verdict"] == "FAIL"]
        verdict = "FAIL" if missing or failed_sets or v["verdict"] == "FAIL" else v["verdict"]
        per_arch[arch] = {"verdict": verdict, "pooled": v, "failed_sets": failed_sets, "missing": missing}
    floor = stats.self_test(ok) if "gpu_self" in arms else None
    arch = {k: v["verdict"] for k, v in per_arch.items()}
    q = {"verdict": "PASS" if arch.get("wh") == "PASS" and arch.get("bh") == "PASS" else
         ("FAIL" if "FAIL" in arch.values() else "INCOMPLETE"),
         "arch": arch, "per_arch": per_arch, "per_feature": per_set,
         "metrics": {k: v["pooled"]["metrics"] for k, v in per_arch.items()},
         "n": {"cases": len(spec["paired"]), "rows": len(ok),
               "per_arm": {arm: len({r["case"] for r in ok if r["arm"] == arm}) for arm in arms}},
         "seed_floor": floor and floor["metrics"], "self_test": floor and floor["verdict"],
         "headline": headline(a.headline, spec["headline"]["cases"], a.inputs, a.work) if a.headline else None,
         "paths": {"cases": str(a.cases), "inputs": str(a.inputs), "arms": arms, "rows": str(a.work / "rows.json")}}
    a.out.write_text(json.dumps(q, indent=1, default=float))
    print(json.dumps({"verdict": q["verdict"], "arch": arch, "self_test": q["self_test"]}))


if __name__ == "__main__":
    main()
