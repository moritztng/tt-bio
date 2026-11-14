"""Freeze the PNR GPU inputs (state/pnr/PROTOCOL.md) into one directory the box reads, with their sha256.

    python3 pnr_inputs.py OUT

mol1-512.json: the page's 512 fixture (perf/size512/fixtures/cdk2x2_512.yaml) as one Mol-1 job, its fixed 35-sequence
a3m as chain A's unpaired MSA (one chain, so nothing to pair) and no MSA search. pdl1-224.yaml + its target structure:
the RFD4 binder input, byte copies of state/rfd4/inputs, checked against the protocol's sha256. BindCraft 2 needs
nothing: its PD-L1 target ships in its own examples/.
"""
import hashlib, json, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "perf/size512/fixtures"
RFD4_IN = Path("/home/moritz/.coworker/state/rfd4/inputs")
PDL1_224_SHA = "a1c2a73f5db76a19977ce5e7b9bb67e666b17abdacfe06473512c3ad2d925829"  # PROTOCOL.md, RFD4 section


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1])
    (out / "mol1").mkdir(parents=True, exist_ok=True)
    seq = re.search(r"sequence:\s*([A-Z]+)", (FIX / "cdk2x2_512.yaml").read_text()).group(1)
    assert len(seq) == 512, len(seq)
    shutil.copyfile(FIX / "cdk2x2_512.a3m", out / "mol1/cdk2x2_512.a3m")
    job = [{"name": "cdk2x2_512", "sequences": [{"proteinChain": {
        "sequence": seq, "count": 1, "id": ["A"], "unpairedMsaPath": "mol1/cdk2x2_512.a3m"}}]}]
    (out / "mol1-512.json").write_text(json.dumps(job, indent=1) + "\n")
    assert sha(RFD4_IN / "pdl1-224.yaml") == PDL1_224_SHA, "pdl1-224.yaml is not the frozen protocol input"
    for f in ("pdl1-224.yaml", "5o45_cropped.cif.zst"):
        shutil.copyfile(RFD4_IN / f, out / f)
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "INDEX.json")
    idx = {str(p.relative_to(out)): sha(p) for p in files}
    (out / "INDEX.json").write_text(json.dumps({"sha256": idx, "fixture_yaml_sha256": sha(FIX / "cdk2x2_512.yaml"),
                                                "note": "MSA paths in mol1-512.json are relative to this directory"},
                                               indent=1) + "\n")
    print(json.dumps(idx, indent=1))


if __name__ == "__main__":
    main()
