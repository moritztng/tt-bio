"""native.cif + meta.json for upstream's TFG examples, so score.py scores them like a panel target.

Each protein chain of the example JSON is matched to the deposited entry's chain with the identical full entity
sequence (first unused chain wins, so two copies map to two chains). The antibody is the constraint's
movable_chains, the antigen the protein chains its contact request names; other chains (9lh2's fixed second VHH)
and ligands are not scored.

    python perf/tfg_acc/natives.py EXAMPLES_DIR [ID...]
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tfg_ref"))
from build_inputs import chain_record, fetch, write_native  # noqa: E402


def _dist(st, recs, m, job, p):
    ids = [next(iter(e.values())).get("id", []) for e in job["sequences"]]
    ends = []
    for k in ("1", "2"):
        c = ids[int(p[f"entity{k}"]) - 1][int(p.get(f"copy{k}", 1)) - 1]
        r = recs[m[c]][1].get(int(p[f"position{k}"]))
        a = r.find_atom(p.get(f"atom{k}", "CA"), "*") if r else None
        if a is None:
            return float("inf")
        ends.append(a.pos)
    return ends[0].dist(ends[1])


def main():
    root = Path(sys.argv[1])
    for tid in sys.argv[2:] or sorted(p.name for p in root.iterdir() if p.is_dir()):
        d = root / tid
        job = json.loads((d / f"{tid}_contact.json").read_text())[0]
        st = fetch(tid, d / f"{tid}.deposited.cif")
        st.setup_entities()
        recs = {ch.name: chain_record(st, ch.name) for ch in st[0]}
        groups = []  # per protein entity: its pred chain ids and the deposited chains with its sequence
        for ent in job["sequences"]:
            (kind, item), = ent.items()
            if kind == "proteinChain":
                hits = [n for n, r in recs.items() if r and r[0] == item["sequence"]]
                if len(hits) < len(item["id"]):
                    sys.exit(f"{tid}: {len(hits)} deposited chains with chain {item['id'][0]}'s sequence")
                groups.append((item["id"], hits))
        # copies of one entity are interchangeable by sequence: take the assignment under which the deposited
        # structure meets the example's own contact request (9lh2: two VHH copies both touch the antigen)
        best = None
        for choice in itertools.product(*(itertools.permutations(h, len(i)) for i, h in groups)):
            m = {c: n for (i, _), ns in zip(groups, choice) for c, n in zip(i, ns)}
            met = sum(_dist(st, recs, m, job, p) <= p.get("max_distance", 8.0) for p in job["constraint"]["contact"])
            if best is None or met > best[0]:
                best = (met, m)
        pred = [c for i, _ in groups for c in i]
        native = [best[1][c] for c in pred]
        ab = list(job["constraint"]["movable_chains"])
        ids = [next(iter(e.values())).get("id", []) for e in job["sequences"]]
        named = {ids[int(p["entity1"]) - 1][int(p.get("copy1", 1)) - 1] for p in job["constraint"]["contact"]}
        ag = [c for c in pred if c in named and c not in ab]
        keep = [i for i, c in enumerate(pred) if c in ab or c in ag]  # e.g. 9lh2's fixed second VHH is not scored
        native, pred = [native[i] for i in keep], [pred[i] for i in keep]
        write_native(st, native, pred, d / "native.cif")
        meta = {"pdb": tid, "native_chains": native, "pred_chains": pred, "antibody_chains": ab, "antigen_chains": ag}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(tid, f"native meets {best[0]}/{len(job['constraint']['contact'])} contacts", meta)


if __name__ == "__main__":
    main()
