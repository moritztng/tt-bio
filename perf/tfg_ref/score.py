"""Score OpenDDE predictions of the TFG panel: interface DockQ, ranking score and constraint satisfaction per sample.

Every sample is checked against BOTH constraint sets of its target (contact and pocket), so the re-rank baseline can be
computed from the unguided candidates. DockQ is the mean over the native antibody-antigen interfaces (report section
2.1, "interface-mean DockQ"); the antibody-antibody interface is not scored. Constraint checks follow upstream's
examples/tfg/check_constraints.py (contact: CA-CA inside the window; epitope residue reached at <= 5 A).

usage: score.py PANEL_DIR RUN_DIR OUT_JSONL [--targets a,b,...]
RUN_DIR holds .../<job name>/seed_<s>/predictions/ as written by box_run.sh.
"""
import argparse
import json
import math
import re
from pathlib import Path

import gemmi
import numpy as np
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

GATE = 5.0


def chain_of(job, entity, copy=1):
    item = next(iter(job["sequences"][int(entity) - 1].values()))
    return item["id"][int(copy) - 1]


def satisfaction(job, st):
    """(contacts met, n contacts) and (epitope residues reached, n, required) of job's constraint on structure st."""
    model = st[0]

    def res(chain, pos):
        for r in model[chain]:
            if r.seqid.num == int(pos):
                return r
        return None

    c = job["constraint"]
    if "contact" in c:
        met = 0
        for p in c["contact"]:
            ra, rb = res(chain_of(job, p["entity1"], p["copy1"]), p["position1"]), res(
                chain_of(job, p["entity2"], p["copy2"]), p["position2"])
            if ra is None or rb is None:
                continue
            d = ra[p["atom1"]][0].pos.dist(rb[p["atom2"]][0].pos)
            met += p["min_distance"] <= d <= p["max_distance"]
        return met, len(c["contact"]), len(c["contact"])
    e = c["epitope"]
    mov = np.array([a.pos.tolist() for ch in c["movable_chains"] for r in model[ch] for a in r
                    if a.element.name != "H"])
    reached = 0
    for r in e["residues"]:
        rr = res(chain_of(job, r["entity"], r["copy"]), r["position"])
        if rr is None:
            continue
        xyz = np.array([a.pos.tolist() for a in rr if a.element.name != "H"])
        reached += float(np.linalg.norm(xyz[:, None] - mov[None], axis=-1).min()) <= GATE
    return reached, len(e["residues"]), max(1, math.ceil(e.get("min_fraction", 0.3) * len(e["residues"])))


def dockq(pred_cif, native, meta):
    model = load_PDB(str(pred_cif))
    mapping = {c: c for c in meta["pred_chains"]}
    res, _ = run_on_all_native_interfaces(model, native, chain_map=mapping)
    ag = set(meta["antigen_chains"])
    per = {k: v["DockQ"] for k, v in res.items() if (k[0] in ag) != (k[1] in ag)}
    return (float(np.mean(list(per.values()))) if per else 0.0), {"".join(k): round(v, 4) for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--targets", default="")
    a = ap.parse_args()
    panel, run = Path(a.panel), Path(a.run)
    want = set(a.targets.split(",")) - {""}
    done = set()
    out = Path(a.out)
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["target"], r["cond"], r["seed"], r["sample"]))
    natives, metas, jobs = {}, {}, {}
    with out.open("a") as f:
        for cif in sorted(run.rglob("predictions/*_sample_*.cif")):
            m = re.match(r"(.+)_(unconstrained|contact|pocket)_sample_(\d+)\.cif", cif.name)
            tid, cond, sample = m.group(1), m.group(2), int(m.group(3))
            seed = int(cif.parent.parent.name.split("_")[1])
            if (want and tid not in want) or (tid, cond, seed, sample) in done:
                continue
            if tid not in metas:
                metas[tid] = json.loads((panel / tid / "meta.json").read_text())
                natives[tid] = load_PDB(str(panel / tid / "native.cif"))
                jobs[tid] = {c: json.loads((panel / tid / f"{tid}_{c}.json").read_text())[0] for c in ("contact", "pocket")}
            conf = json.loads((cif.parent / cif.name.replace("_sample_", "_summary_confidence_sample_").replace(".cif", ".json")).read_text())
            st = gemmi.read_structure(str(cif))
            dq, per = dockq(cif, natives[tid], metas[tid])
            row = {"target": tid, "cond": cond, "seed": seed, "sample": sample, "dockq": round(dq, 4), "per_interface": per,
                   "ranking_score": conf.get("final_score", conf.get("ranking_score")),
                   "contact": satisfaction(jobs[tid]["contact"], st), "pocket": satisfaction(jobs[tid]["pocket"], st)}
            f.write(json.dumps(row) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
