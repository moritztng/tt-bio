"""Size and capability sweep for `--model openbind`: does it fold, at what size, and where.

Not an accuracy check — the accuracy legs live in scripts/full_parity_gate.py. This looks
for the two failure modes a new port hits at size: an L1-residency gate that goes dark above
some length (tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa) and an allocation-count OOM
at 1024 (of3-1024aa-oom-allocation-count-not-size). Cheap sampler settings, one sample, one
seed: a crash or an OOM shows up regardless.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:<slug> \\
        <devpy> scripts/ob0_size_sweep.py --out /tmp/ob0_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LADDER = REPO / "scripts" / "rf3_port" / "size_ladder"

# capability class -> (yaml, extra tt-bio args). Every class the model claims to accept.
CAPABILITY = [
    # prot_custom_msa carries its own `msa:` a3m, so the MSA case needs no server and no
    # staging. examples/ubq.yaml has neither an `msa:` key nor a cached alignment, so
    # without --single_sequence it reaches for colabfold_search and dies on the host, not
    # on the card.
    ("protein, user-supplied MSA", "examples/prot_custom_msa.yaml", []),
    ("protein, single-sequence", "examples/8hel_nomsa.yaml", ["--single_sequence"]),
    ("protein + templates", "examples/7xi5_tmpl.yaml", []),
    ("multi-chain (heterodimer)", "examples/9bk6.yaml", []),
    ("protein + ligand (CCD)", "examples/fkg_ligand.yaml", ["--single_sequence"]),
    ("protein + ligand (SMILES)", "examples/ligand.yaml", ["--single_sequence"]),
    ("RNA + DNA + protein", "<generated>", ["--single_sequence"]),
]

# A nucleic-acid case does not exist in examples/, so the sweep writes one. Short chains: the
# question is whether the featurizer and the model accept the molecule types at all.
RNA_DNA_YAML = """version: 1
sequences:
  - protein:
      id: A
      sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
  - rna:
      id: B
      sequence: GGCUAGCUAGCUAGCC
  - dna:
      id: C
      sequence: ATGCATGCATGCATGC
"""


def ladder_yaml(rung: Path, dest: Path) -> tuple[Path, int]:
    """One RF3 size-ladder rung as a tt-bio yaml. Protein chains only, in file order."""
    spec = json.loads((rung / "input.json").read_text())[0]
    chains, total = [], 0
    for i, comp in enumerate(spec["components"]):
        seq = comp.get("seq")
        if not seq:
            continue
        chains.append(f"  - protein:\n      id: {chr(ord('A') + i)}\n      sequence: {seq}")
        total += len(seq)
    out = dest / f"{rung.name}.yaml"
    out.write_text("version: 1\nsequences:\n" + "\n".join(chains) + "\n")
    return out, total


def fold(label: str, yaml: Path | str, extra: list[str], out_root: Path,
         steps: int, timeout: float) -> dict:
    out_dir = out_root / re.sub(r"[^a-z0-9]+", "_", label.lower())
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_root / f"{out_dir.name}.log"
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(yaml),
           "--model", "openbind", "--out_dir", str(out_dir), "--override",
           "--seed", "0", "--sampling_steps", str(steps), "--diffusion_samples", "1"] + extra
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=REPO, timeout=None if timeout <= 0 else timeout)
    wall = time.time() - t0
    rec = {"label": label, "yaml": str(yaml), "rc": proc.returncode,
           "wall_s": round(wall, 1), "log": str(log)}
    results = list(out_dir.rglob("results.json"))
    if results:
        try:
            rows = json.loads(results[0].read_text())
            row = rows[0] if isinstance(rows, list) else rows
            rec.update(status=row.get("status"), plddt=row.get("plddt"),
                       ptm=row.get("ptm"), n_residues=row.get("n_residues"))
        except Exception as e:  # a results.json that exists but will not parse is a finding
            rec["results_error"] = repr(e)
    if proc.returncode != 0 or rec.get("status") not in (None, "ok"):
        txt = Path(log).read_text()
        oom = [l for l in txt.splitlines()
               if "out of memory" in l.lower() or "Out of Memory" in l or "OOM" in l]
        rec["failure_tail"] = (oom[-1] if oom else txt.strip().splitlines()[-1] if txt.strip() else "")
        rec["oom"] = bool(oom)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/ob0_sweep.json")
    ap.add_argument("--work", default="/home/ttuser/ob0_sweep")
    ap.add_argument("--rungs", default="cdk2_128,cdk2_256,cdk2_512,cdk2_768,cdk2_1024")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--skip-capability", action="store_true")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "yaml").mkdir(exist_ok=True)
    records = []

    for name in [r for r in args.rungs.split(",") if r]:
        rung = LADDER / name
        if not (rung / "input.json").exists():
            records.append({"label": f"ladder {name}", "rc": -1,
                            "failure_tail": f"no ladder input at {rung}"})
            continue
        y, aa = ladder_yaml(rung, work / "yaml")
        rec = fold(f"ladder {name} ({aa} aa)", y, ["--single_sequence"], work,
                   args.steps, args.timeout)
        rec["aa"] = aa
        records.append(rec)
        print(json.dumps(rec), flush=True)

    if not args.skip_capability:
        for label, y, extra in CAPABILITY:
            if y == "<generated>":
                y = work / "yaml" / "rna_dna_protein.yaml"
                y.write_text(RNA_DNA_YAML)
            rec = fold(label, y, extra, work, args.steps, args.timeout)
            records.append(rec)
            print(json.dumps(rec), flush=True)

    Path(args.out).write_text(json.dumps(records, indent=2))
    print("\n| case | aa | rc | wall (s) | pLDDT | note |")
    print("|---|---|---|---|---|---|")
    for r in records:
        print(f"| {r['label']} | {r.get('aa') or r.get('n_residues') or '-'} | {r['rc']} "
              f"| {r.get('wall_s', '-')} "
              f"| {round(r['plddt'], 1) if isinstance(r.get('plddt'), (int, float)) else '-'} "
              f"| {'OOM' if r.get('oom') else (r.get('failure_tail', '') or 'ok')[:70]} |")
    bad = [r for r in records if r["rc"] != 0]
    print(f"\n{len(records) - len(bad)}/{len(records)} folded; {len(bad)} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
