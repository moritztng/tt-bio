"""Freeze PVO timing inputs (state/pvo/INPUTS.md) into a box-ready tree with relative MSA paths and hashes.

    python3 make_inputs.py --src ~/pfm-accuracy-data/box --out DIR 9TH6 9LV4 9W3L

Reads PFM's acc.json (the Protenix job format both OpenDDE and Mol-1 accept) and copies each complex's a3m files
byte for byte, so GPU, TT and pvo-accuracy all see the same MSAs. Writes DIR/<PDB>.json (MSA paths relative to DIR)
and DIR/INDEX.json: sha256 of every file plus the MSA depth per chain (sequences in the a3m minus the query),
which is what MATCHED quotes.
"""
import argparse, hashlib, json, shutil
from pathlib import Path


def depth(a3m):
    return max(sum(1 for l in a3m.open() if l.startswith(">")) - 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("pdbs", nargs="+")
    a = ap.parse_args()
    jobs = {j["name"].upper(): j for j in json.loads((a.src / "acc.json").read_text())}
    a.out.mkdir(parents=True, exist_ok=True)
    index = {"sha256": {}, "depth": {}, "tokens": {}}
    for pdb in a.pdbs:
        job = json.loads(json.dumps(jobs[pdb.upper()]))
        job.pop("modelSeeds", None)  # seeds come from the driver, never from the input
        for e in job["sequences"]:
            pc = e.get("proteinChain")
            if not pc:
                continue
            for k in ("pairedMsaPath", "unpairedMsaPath"):
                if pc.get(k):
                    src = a.src / "msa" / Path(pc[k]).relative_to("/root/pfm/acc/msa")
                    rel = Path("msa") / src.relative_to(a.src / "msa")
                    (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, a.out / rel)
                    pc[k] = str(rel)
                    index["depth"][str(rel)] = depth(src)
        index["tokens"][pdb] = sum(len(e["proteinChain"]["sequence"]) * e["proteinChain"].get("count", 1)
                                   for e in job["sequences"] if "proteinChain" in e)
        (a.out / f"{pdb}.json").write_text(json.dumps([job], indent=1))
    for p in sorted(a.out.rglob("*")):
        if p.is_file() and p.name != "INDEX.json":
            index["sha256"][str(p.relative_to(a.out))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (a.out / "INDEX.json").write_text(json.dumps(index, indent=1))
    print(json.dumps({"tokens": index["tokens"], "depth": index["depth"]}, indent=1))


if __name__ == "__main__":
    main()
