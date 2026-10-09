"""Score TT (or GPU) predictions of a TFG set with tfg-upstream's scorer, for any condition name.

perf/tfg_ref/score.py is the one scorer (interface-mean DockQ, upstream's constraint checks); it only knows the
conditions unconstrained/contact/pocket. This walks the same layout for any condition (the robustness set's
c4/wrong1/wrong2/shift too) and adds `req`: satisfaction of the constraint that condition actually requested.
`contact` and `pocket` stay the TRUE requests of the target, as in the reference scorer.

usage: score.py SET_DIR RUN_DIR OUT_JSONL
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tfg_ref"))
import gemmi  # noqa: E402
from DockQ.DockQ import load_PDB  # noqa: E402
from score import dockq, satisfaction  # noqa: E402

NAME = re.compile(r"([0-9a-z]{4})_(\w+?)_sample_(\d+)\.cif")


def main():
    panel, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    done = set()
    if out.exists():
        done = {(r["target"], r["cond"], r["seed"], r["sample"]) for r in map(json.loads, out.read_text().splitlines())}
    natives, metas, jobs = {}, {}, {}
    with out.open("a") as f:
        for cif in sorted(run.rglob("predictions/*_sample_*.cif")):
            m = NAME.fullmatch(cif.name)
            cj = cif.parent / cif.name.replace("_sample_", "_summary_confidence_sample_").replace(".cif", ".json")
            if not m or not cj.exists():
                continue
            tid, cond, sample = m.group(1), m.group(2), int(m.group(3))
            seed = int(cif.parent.parent.name.split("_")[1])
            if (tid, cond, seed, sample) in done or not (panel / tid / "meta.json").exists():
                continue
            if tid not in metas:
                metas[tid] = json.loads((panel / tid / "meta.json").read_text())
                natives[tid] = load_PDB(str(panel / tid / "native.cif"))
                jobs[tid] = {p.stem.split("_", 1)[1]: json.loads(p.read_text())[0] for p in (panel / tid).glob(f"{tid}_*.json")}
            conf = json.loads(cj.read_text())
            st = gemmi.read_structure(str(cif))
            dq, per = dockq(cif, natives[tid], metas[tid])
            row = {"target": tid, "cond": cond, "seed": seed, "sample": sample, "dockq": round(dq, 4),
                   "per_interface": per, "ranking_score": conf.get("ranking_score")}
            for c in ("contact", "pocket"):
                if c in jobs[tid]:
                    row[c] = satisfaction(jobs[tid][c], st)
            if jobs[tid].get(cond, {}).get("constraint"):
                row["req"] = satisfaction(jobs[tid][cond], st)
            f.write(json.dumps(row) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
