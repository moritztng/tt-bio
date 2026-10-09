"""Wrong and shifted contact constraints for the robustness check (tech report section 2.3, Table 4).

From a panel target's 5-contact request this writes, next to it, four contact inputs with n = 4:
    <tid>_c4.json      the first four requested pairs, unperturbed (the control)
    <tid>_wrong1.json  one of the four antigen residues replaced by one >= 12 A from the true epitope
    <tid>_wrong2.json  two of the four replaced the same way
    <tid>_shift.json   all four antigen residues replaced by a neighbour whose CA is 4-6 A from the original CA
The target's own inputs are copied beside them, so the scorer still checks the true contact and pocket requests.
The true epitope is every antigen residue with a heavy atom within 4.5 A of an antibody heavy atom in the native
(the report's pocket rule). "12 A from the epitope" is CA to the nearest epitope CA. The antibody side of each pair
is kept. Choices are seeded by the target id, so a rerun writes the same inputs. A target without enough
candidate residues is skipped and says so.

    python perf/tfg_acc/perturb.py PANEL_DIR OUT_SET_DIR TID...
"""
import copy
import json
import random
import shutil
import sys
from pathlib import Path

import gemmi
import numpy as np

CONDS = ("c4", "wrong1", "wrong2", "shift")


def chain_of(job, entity, copy_=1):
    item = next(iter(job["sequences"][int(entity) - 1].values()))
    return item["id"][int(copy_) - 1]


def main():
    panel, out = Path(sys.argv[1]), Path(sys.argv[2])
    for tid in sys.argv[3:]:
        meta = json.loads((panel / tid / "meta.json").read_text())
        job = json.loads((panel / tid / f"{tid}_contact.json").read_text())[0]
        model = gemmi.read_structure(str(panel / tid / "native.cif"))[0]
        ag_chains, ab_chains = set(meta["antigen_chains"]), set(meta["antibody_chains"])
        ab_xyz = np.array([a.pos.tolist() for ch in model if ch.name in ab_chains for r in ch for a in r
                           if a.element.name != "H"])
        ag = {}  # (chain, pos) -> (CA xyz, min heavy-atom distance to the antibody)
        for ch in model:
            if ch.name not in ag_chains:
                continue
            for r in ch:
                if r.find_atom("CA", "*") is None:
                    continue
                xyz = np.array([a.pos.tolist() for a in r if a.element.name != "H"])
                d = np.linalg.norm(xyz[:, None] - ab_xyz[None], axis=-1).min()
                ag[(ch.name, r.seqid.num)] = (np.array(r.find_atom("CA", "*").pos.tolist()), d)
        epi = np.array([ca for ca, d in ag.values() if d <= 4.5])
        far = sorted(k for k, (ca, _) in ag.items() if np.linalg.norm(epi - ca, axis=-1).min() >= 12.0)
        pairs = job["constraint"]["contact"][:4]
        # the antigen side of each pair: entity1 is the antigen in build_inputs.py's requests
        side = [(chain_of(job, p["entity1"], p["copy1"]), int(p["position1"])) for p in pairs]
        assert all(c in ag_chains for c, _ in side), f"{tid}: entity1 of a request is not the antigen"
        rng = random.Random(tid)
        chain_entity = {chain_of(job, e + 1, c + 1): (e + 1, c + 1) for e, s in enumerate(job["sequences"])
                        for c in range(len(next(iter(s.values()))["id"]))}

        def put(p, key):
            p = dict(p)
            p["entity1"], p["copy1"] = str(chain_entity[key[0]][0]), chain_entity[key[0]][1]
            p["position1"] = str(key[1])
            return p

        if len(far) < 2:
            print(tid, "skipped: fewer than 2 antigen residues >= 12 A from the epitope")
            continue
        variants = {"c4": pairs}
        for n in (1, 2):
            idx = rng.sample(range(4), n)
            picks = rng.sample(far, n)
            variants[f"wrong{n}"] = [put(p, picks[idx.index(i)]) if i in idx else p for i, p in enumerate(pairs)]
        shifted = []
        for p, key in zip(pairs, side):
            ca = ag[key][0]
            near = sorted(k for k, (x, _) in ag.items() if k != key and 4.0 <= np.linalg.norm(x - ca) <= 6.0)
            if not near:
                break
            shifted.append(put(p, rng.choice(near)))
        if len(shifted) == 4:
            variants["shift"] = shifted
        d = out / tid
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "msa").exists():
            shutil.copytree(panel / tid / "msa", d / "msa", ignore=shutil.ignore_patterns("*.tar.gz", "*.sh", "*.json"))
        for f in ("meta.json", "native.cif", *(f"{tid}_{c}.json" for c in ("unconstrained", "contact", "pocket"))):
            shutil.copyfile(panel / tid / f, d / f)
        for cond, ps in variants.items():
            j = copy.deepcopy(job)
            j["name"] = f"{tid}_{cond}"
            j["constraint"]["contact"] = ps
            (d / f"{tid}_{cond}.json").write_text(json.dumps([j], indent=1))
        print(tid, "wrote", ",".join(variants), "far", len(far))


if __name__ == "__main__":
    main()
