"""Freeze the PVO accuracy set (state/pvo/INPUTS.md, 11 complexes) into one box-ready tree.

    python3 make_acc.py --out ~/pvo-accuracy-data/box --prefix /root/kc/acc

Source is PFM's acc.json plus its a3m files, which pvo-recon froze as the PVO inputs: the same MSA bytes feed
GPU, TT and the scorer, and no row rebuilds one (an unmatched MSA flattered PFM's 10-03 offer by 14 %). MSA paths
are rewritten absolute under --prefix, the path the tree will sit at on the box, so nothing depends on a cwd.
INDEX.json carries the sha256 of every file and the MSA depth per chain, which is what the box checks on arrival.
"""
import argparse, hashlib, json, shutil
from pathlib import Path

SRC = Path.home() / "pfm-accuracy-data" / "box"


def depth(a3m):
    return max(sum(1 for l in a3m.open() if l.startswith(">")) - 1, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prefix", default="/root/kc/acc", help="where the tree lives on the box")
    ap.add_argument("--src", default=SRC, type=Path)
    a = ap.parse_args()
    jobs = json.loads((a.src / "acc.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    index = {"prefix": a.prefix, "sha256": {}, "depth": {}, "tokens": {}}
    for job in jobs:
        job.pop("modelSeeds", None)  # seeds come from the arm, never from the input
        for e in job["sequences"]:
            pc = e.get("proteinChain")
            if not pc:
                continue
            for k in ("pairedMsaPath", "unpairedMsaPath"):
                if not pc.get(k):
                    continue
                rel = Path("msa") / Path(pc[k]).parent.name / Path(pc[k]).name
                src = a.src / rel
                (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, a.out / rel)
                pc[k] = f"{a.prefix}/{rel}"
                index["depth"][str(rel)] = depth(src)
        index["tokens"][job["name"]] = sum(len(e["proteinChain"]["sequence"]) * e["proteinChain"].get("count", 1)
                                           for e in job["sequences"] if "proteinChain" in e)
    (a.out / "acc.json").write_text(json.dumps(jobs, indent=1))
    for p in sorted(a.out.rglob("*")):
        if p.is_file() and p.name != "INDEX.json":
            index["sha256"][str(p.relative_to(a.out))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (a.out / "INDEX.json").write_text(json.dumps(index, indent=1))
    print(json.dumps({"complexes": len(jobs), "tokens": index["tokens"], "files": len(index["sha256"])}, indent=1))


if __name__ == "__main__":
    main()
