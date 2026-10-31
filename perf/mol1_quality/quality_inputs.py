"""Freeze the MOL1 quality inputs (state/mol1/quality/CASES.json) with their MSAs, once, in the upstream env.

    python -I quality_inputs.py --cases CASES.json --upstream /root/mol1 --out /root/kc/quality [--jobs 4]

Every FoldBench / pfm case: the RCSB entry mmCIF -> upstream's own `cif_to_input_json(assembly_id="1")` (what
`mol json --assembly_id 1` runs) -> upstream's `update_infer_json` (the ColabFold MMseqs2 search `mol pred` uses).
Recipe variants reuse the base case's MSAs: `ligand_as_smiles` swaps each CCD ligand for its CCD SMILES,
`ligand_as_sdf_file` for a FILE_ SDF of the CCD ideal conformer, `use_msa_false` drops nothing from the JSON (the run
flag does it). Upstream example cases are copied with the MSA/template files they ship.
Writes <out>/<case>.json (one job each, MSA paths relative to <out>, run with cwd = <out>), the ground-truth
assembly CIFs under <out>/truth/ (RCSB <pdb>-assembly1.cif, FoldBench's reference format), and INDEX.json with the
sha256 of every file. A case that fails is recorded under "failed" with its error, never dropped silently.
"""
import argparse, gzip, hashlib, json, shutil, traceback, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MSA_KEYS = ("pairedMsaPath", "unpairedMsaPath", "templatesPath")
RCSB = "https://files.rcsb.org/download/"


def fetch(url, dst):
    if not dst.exists():
        data = urllib.request.urlopen(url, timeout=300).read()
        dst.write_bytes(gzip.decompress(data) if url.endswith(".gz") else data)
    return dst


def safe(case_id):
    return case_id.replace(":", "__").replace("/", "_").replace("#", "_")


def localise(job, d, out, src_root):
    """Copy every MSA/template file a job names under d/msa/<i>/ and make the path relative to out."""
    for i, s in enumerate(job["sequences"]):
        body = next(iter(s.values()))
        for k in MSA_KEYS:
            if isinstance(body, dict) and body.get(k):
                p = Path(body[k]) if Path(body[k]).is_absolute() else src_root / body[k]
                dst = d / "msa" / str(i) / p.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(p, dst, dirs_exist_ok=True) if p.is_dir() else shutil.copyfile(p, dst)
                body[k] = str(dst.relative_to(out))
    return job


def base_case(c, a):
    from mol.data.inference.json_maker import cif_to_input_json
    from mol.runner.msa_search import update_infer_json
    name = safe(c["id"]); d = a.out / name; d.mkdir(parents=True, exist_ok=True)
    pdb = c["truth"].split(":")[1].split("-")[0]
    entry = fetch(f"{RCSB}{pdb}.cif.gz", a.out / "entry" / f"{pdb}.cif")
    fetch(f"{RCSB}{pdb}-assembly1.cif.gz", a.out / "truth" / f"{pdb}-assembly1.cif")
    raw = d / "raw.json"
    cif_to_input_json(str(entry), assembly_id="1", altloc="first", output_json=str(raw), sample_name=name)
    job = json.loads(raw.read_text()); job = job[0] if isinstance(job, list) else job
    job["name"] = name
    raw.write_text(json.dumps([job], indent=1))
    searched, _ = update_infer_json(str(raw), str(d / "search"), use_msa=True)
    job = localise(json.loads(Path(searched).read_text())[0], d, a.out, d)
    shutil.rmtree(d / "search", ignore_errors=True); raw.unlink()
    return job


def variant(c, a, jobs):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from mol.data.core.ccd import get_component_rdkit_mol
    job = json.loads(json.dumps(jobs[c["base"]])); job["name"] = safe(c["id"])
    if c["recipe"] in ("ligand_as_smiles", "ligand_as_sdf_file"):
        n = 0
        for s in job["sequences"]:
            lig = s.get("ligand")
            if lig and lig["ligand"].startswith("CCD_") and "_" not in lig["ligand"][4:]:
                m = get_component_rdkit_mol(lig["ligand"][4:])
                if m is None or m.GetNumHeavyAtoms() < 6:  # leave ions/tiny additives as CCD
                    continue
                if c["recipe"] == "ligand_as_smiles":
                    lig["ligand"] = Chem.MolToSmiles(Chem.RemoveHs(m))
                else:
                    f = a.out / safe(c["id"]) / f"{lig['ligand'][4:]}.sdf"; f.parent.mkdir(parents=True, exist_ok=True)
                    mh = Chem.AddHs(Chem.RemoveHs(m))
                    if AllChem.EmbedMolecule(mh, randomSeed=0) != 0:
                        raise RuntimeError(f"no conformer for {lig['ligand']}")
                    Chem.MolToMolFile(Chem.RemoveHs(mh), str(f))
                    lig["ligand"] = "FILE_" + str(f.relative_to(a.out))
                n += 1
        if not n:
            raise RuntimeError("no CCD ligand to re-express")
    return job


def upstream_case(c, a):
    src = c["source"].split("examples/", 1)[1]
    path, _, name = src.partition("#")
    jobs = json.loads((a.upstream / "examples" / path).read_text())
    job = next(j for j in jobs if not name or j["name"] == name)
    job["name"] = safe(c["id"])
    d = a.out / job["name"]; d.mkdir(parents=True, exist_ok=True)
    return localise(job, d, a.out, a.upstream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--headline", action="store_true", help="also freeze the GPU-only FoldBench-AB headline set")
    a = ap.parse_args()
    for sub in ("entry", "truth"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    spec = json.loads(a.cases.read_text())
    cases = list(spec["paired"])
    if a.headline:
        have = {c["id"] for c in cases}
        cases += [dict(id=h["id"], set="headline", truth="rcsb:" + h["id"], recipe="default")
                  for h in spec["headline"]["cases"] if h["id"] not in have]
    jobs, failed = {}, {}

    def run(c):
        try:
            if c["id"].startswith("upstream:"):
                return c["id"], upstream_case(c, a)
            return c["id"], base_case(c, a)
        except Exception:
            return c["id"], traceback.format_exc(limit=3)

    plain = [c for c in cases if "base" not in c]
    with ThreadPoolExecutor(a.jobs) as ex:
        for cid, r in ex.map(run, plain):
            (jobs if isinstance(r, dict) else failed)[cid] = r
            print(cid, "ok" if isinstance(r, dict) else "FAILED", flush=True)
    for c in (c for c in cases if "base" in c):
        try:
            jobs[c["id"]] = variant(c, a, jobs)
        except Exception:
            failed[c["id"]] = traceback.format_exc(limit=3)
    for cid, job in jobs.items():
        (a.out / f"{safe(cid)}.json").write_text(json.dumps([job], indent=1))
    files = sorted(p for p in a.out.rglob("*") if p.is_file() and p.name != "INDEX.json" and "entry" not in p.parts)
    rec = {c["id"]: c.get("recipe", "default") for c in cases}
    idx = {"cases": {cid: f"{safe(cid)}.json" for cid in jobs}, "recipes": {cid: rec[cid] for cid in jobs},
           "headline_only": sorted(c["id"] for c in cases if c["set"] == "headline" and c["id"] in jobs),
           "failed": failed,
           "sha256": {str(p.relative_to(a.out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
           "note": "run with cwd = this directory; MSA/template/FILE_ paths in the JSONs are relative to it"}
    (a.out / "INDEX.json").write_text(json.dumps(idx, indent=1))
    print(len(jobs), "frozen,", len(failed), "failed")


if __name__ == "__main__":
    main()
