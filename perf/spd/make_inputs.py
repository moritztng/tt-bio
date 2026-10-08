"""Build the SPD speed inputs into one data dir (default ~/spd-data) and print its manifest.

    python perf/spd/make_inputs.py [--data ~/spd-data]

Needs, already in <data>:
  msa/                      tt-bio MSA cache of complex730 (the .107 pfm-ttfast cache: 5,969 paired + 1,911 + 3,978
                            unpaired rows, 9,947 after the engine merges them)
  build/inputs/<pdb>.yaml   the PopVax-shaped complexes 8y9t, 9dis, 9w8a (wk/pfm-gpu perf/pfm_gpu/inputs)
  build/msa/<pdb>/cache/    their caches, from wk/pfm-gpu's make_msa.py (ColabFold API, greedy pairing)

Writes <data>/inputs/<name>.yaml and merges every cache into <data>/msa:
  c730          complex730, A 580 + B 150 = 730 tokens, the campaign's reference fold
  p8y9t p9dis p9w8a   PopVax-shaped target ~600 + binder ~125 (717 / 754 / 778 tokens)
  l256 l512     complex730 cropped to A 206 + B 50 and A 410 + B 102, the MSA cropped by column (paired rows
                stay row-aligned), so these rungs keep the 730's alignment depth
  l1024 l1536   chain A cropped to 512, as a homodimer and a homotrimer. Identical chains do not pair, so these
                rungs carry the 1,911-row unpaired alignment of chain A on every copy
  <PDB> x 11    the accuracy set (perf/pfm_accuracy/set.tsv): post-cutoff two-chain complexes with deposited
                structures, copied from --acc (pfm-accuracy's data dir: inputs/, msa/cache/, ref/) with their
                references into <data>/ref
"""
import argparse, hashlib, re, shutil
from pathlib import Path

import yaml

from tt_bio.cache import paired_msa_dir, seq_hash

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("~/spd-data").expanduser())
ap.add_argument("--acc", type=Path, default=Path("~/pfm-accuracy-data").expanduser())
a = ap.parse_args()
D = a.data
MSA = D / "msa"
(D / "inputs").mkdir(parents=True, exist_ok=True)
C730 = yaml.safe_load((Path(__file__).parent / "complex730.yaml").read_text())
A, B = (s["protein"]["sequence"] for s in C730["sequences"])


def write_yaml(name, chains, note):
    seqs = [{"protein": {"id": cid, "sequence": s}} for cid, s in chains]
    (D / "inputs" / f"{name}.yaml").write_text(f"version: 1\n# {note}\n" + yaml.safe_dump({"sequences": seqs},
                                                                                          sort_keys=False))


def crop_a3m(text, n):
    """Keep the first n match columns of every row (and the insertions before them). Row order and count are
    kept, so a paired file stays row-aligned with its partner."""
    out = []
    for line in text.splitlines():
        if line.startswith(">") or not line:
            out.append(line); continue
        cols, keep = 0, []
        for tok in re.findall(r"[a-z.]*[A-Z\-]", line):
            if cols == n:
                break
            keep.append(tok); cols += 1
        out.append("".join(keep))
    return "\n".join(out) + "\n"


def crop_complex(name, na, nb):
    a, b = A[:na], B[:nb]
    for full, s in ((A, a), (B, b)):
        (MSA / f"{seq_hash(s)}.a3m").write_text(crop_a3m((MSA / f"{seq_hash(full)}.a3m").read_text(), len(s)))
    src, dst = paired_msa_dir(MSA, [A, B]), paired_msa_dir(MSA, [a, b])
    dst.mkdir(parents=True, exist_ok=True)
    for full, s in ((A, a), (B, b)):
        (dst / f"{seq_hash(s)}.a3m").write_text(crop_a3m((src / f"{seq_hash(full)}.a3m").read_text(), len(s)))
    write_yaml(name, [("A", a), ("B", b)], f"complex730 cropped to A 1-{na} + B 1-{nb}, MSA cropped by column")


def homomer(name, n, copies):
    s = A[:n]
    (MSA / f"{seq_hash(s)}.a3m").write_text(crop_a3m((MSA / f"{seq_hash(A)}.a3m").read_text(), n))
    write_yaml(name, [(chr(65 + i), s) for i in range(copies)],
               f"complex730 chain A 1-{n} x {copies}; identical chains do not pair")


write_yaml("c730", [("A", A), ("B", B)], "complex730 (UniRef50_A0A2C9LWN7 crops 1-580 + 581-730), 730 tokens")
crop_complex("l256", 206, 50)
crop_complex("l512", 410, 102)
homomer("l1024", 512, 2)
homomer("l1536", 512, 3)
for pdb in ("8y9t", "9dis", "9w8a"):
    src = D / "build" / "msa" / pdb / "cache"
    if not src.exists():
        print(f"skip p{pdb}: no cache at {src}"); continue
    shutil.copytree(src, MSA, dirs_exist_ok=True)
    shutil.copy(D / "build" / "inputs" / f"{pdb}.yaml", D / "inputs" / f"p{pdb}.yaml")

if (a.acc / "msa" / "cache").exists():
    shutil.copytree(a.acc / "msa" / "cache", MSA, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*_tmp_*"))
    shutil.copytree(a.acc / "ref", D / "ref", dirs_exist_ok=True)
    for pdb in (l.split("\t")[0] for l in (Path(__file__).parents[1] / "pfm_accuracy" / "set.tsv")
                .read_text().splitlines()[1:]):
        shutil.copy(a.acc / "inputs" / f"{pdb}.yaml", D / "inputs" / f"{pdb}.yaml")
else:
    print(f"skip accuracy set: no {a.acc}/msa/cache")

# Manifest: what every box must hold, byte for byte.
for f in sorted((D / "inputs").glob("*.yaml")):
    seqs = [s["protein"]["sequence"] for s in yaml.safe_load(f.read_text())["sequences"]]
    files = [MSA / f"{seq_hash(s)}.a3m" for s in set(seqs)]
    p = paired_msa_dir(MSA, seqs)
    if p is not None:
        files += [p / f"{seq_hash(s)}.a3m" for s in set(seqs)]
    rows = {str(x.relative_to(MSA)): x.read_text().count(">") for x in files if x.exists()}
    missing = [str(x.relative_to(MSA)) for x in files if not x.exists()]
    h = hashlib.sha256(b"".join(x.read_bytes() for x in sorted(files) if x.exists())).hexdigest()[:16]
    print(f"{f.stem}\ttokens={sum(map(len, seqs))}\tchains={len(seqs)}\tmsa_sha={h}\trows={rows}"
          + (f"\tMISSING={missing}" if missing else ""))
