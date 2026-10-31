"""Freeze the three MOL1 gated inputs (state/mol1/PROTOCOL.md section 2), with their MSAs, once.

    python mol1_inputs.py --upstream /root/mol1 --out /root/kc/inputs

Writes mol1-{256,512,1024}.json (one job each), the a3m files under <size>/msa/, and INDEX.json with the sha256 of
every file. MSA paths inside the JSONs are relative to --out, so both sides run with cwd = the inputs directory and
read the same bytes. 256 reuses the MSAs upstream ships for 7r6r; 512 and 1024 are searched once through upstream's
own `update_infer_json` (the ColabFold MMseqs2 path `mol pred` uses).
"""
import argparse, hashlib, json, shutil
from pathlib import Path

CASES = {256: ("examples/example.json", "7r6r"),
         512: ("examples/example_without_msa.json", "8cnh-assembly1"),
         1024: ("examples/example_without_msa.json", "8j2w-assembly1")}
MSA_KEYS = ("pairedMsaPath", "unpairedMsaPath")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from mol.runner.msa_search import update_infer_json
    for size, (src, name) in CASES.items():
        job = next(j for j in json.loads((a.upstream / src).read_text()) if j["name"] == name)
        d = a.out / f"mol1-{size}"
        tmp = d / "search.json"
        d.mkdir(exist_ok=True)
        # upstream's own relative paths are relative to its checkout
        for s in job["sequences"]:
            for k in MSA_KEYS:
                if k in s.get("proteinChain", {}):
                    s["proteinChain"][k] = str(a.upstream / s["proteinChain"][k])
        tmp.write_text(json.dumps([job], indent=2))
        searched, _ = update_infer_json(str(tmp), str(d / "search"))
        job = json.loads(Path(searched).read_text())[0]
        for i, s in enumerate(job["sequences"]):
            pc = s.get("proteinChain")
            for k in MSA_KEYS if pc else ():
                if k in pc:
                    dst = d / "msa" / str(i) / Path(pc[k]).name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(pc[k], dst)
                    pc[k] = str(dst.relative_to(a.out))
        (a.out / f"mol1-{size}.json").write_text(json.dumps([job], indent=2))
        shutil.rmtree(d / "search", ignore_errors=True)
        for p in d.glob("search*.json"):
            p.unlink()
    files = sorted(p for p in a.out.rglob("*") if p.is_file() and p.name != "INDEX.json")
    idx = {"cases": {s: {"source": f"{src}:{n}"} for s, (src, n) in CASES.items()},
           "sha256": {str(p.relative_to(a.out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
           "note": "run with cwd = this directory; MSA paths in the JSONs are relative to it"}
    (a.out / "INDEX.json").write_text(json.dumps(idx, indent=1))
    print(json.dumps(idx["sha256"], indent=1))


if __name__ == "__main__":
    main()
