"""Build the SPD speed inputs into one data dir (default ~/spd-data) and print its manifest.

    python perf/spd/make_inputs.py [--data ~/spd-data]
    python perf/spd/make_inputs.py --boltz2-csv     # only the Boltz-2 CSVs, on a box that already has the data

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
  <PDB>_512 x 11  the same complexes at 512 tokens, the GAP cell's size: the whole binder plus the window of the
                target centred on its interface, MSA cropped by column window, reference renumbered to the crop
                (perf/pfm_accuracy/set512.tsv). Routes that only exist at <= 512 tokens get a verdict here; the
                11-set itself runs 546-791 tokens and never reaches them

Boltz-2 reads a complex's paired MSA as one `{seq_hash}.csv` per chain in its paired dir, not the a3m pair the
other models read. For every input that pairs, the CSVs are written from the same paired and unpaired a3m with
the engine's own writer (`main.write_boltz_csvs`, what an online search writes), so all four fold models see one
alignment. Without them a cache-only Boltz-2 run silently folds unpaired. Every input is foldable by Protenix-v2,
OpenDDE, OpenFold3 and Boltz-2 (protein chains only); BindCraft 2 brings its own target.
"""
import argparse, hashlib, re, shutil
from pathlib import Path

import yaml

from tt_bio.cache import paired_msa_dir, seq_hash

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("~/spd-data").expanduser())
ap.add_argument("--acc", type=Path, default=Path("~/pfm-accuracy-data").expanduser())
ap.add_argument("--boltz2-csv", action="store_true", help="only write the Boltz-2 CSVs for the inputs already here")
ap.add_argument("--truth512", action="store_true", help="only (re)write the 512-token accuracy set from the 11-set here")
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


def crop_a3m(text, n, start=0):
    """Keep match columns start..start+n-1 of every row (and the insertions between them). Row order and count
    are kept, so a paired file stays row-aligned with its partner."""
    out = []
    for line in text.splitlines():
        if line.startswith(">") or not line:
            out.append(line); continue
        toks = re.findall(r"[a-z.]*[A-Z\-]", line)[start:start + n]
        if toks and start:
            toks[0] = toks[0][-1]  # an insertion before the window's first column belongs to the cut
        out.append("".join(toks))
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


def truth512(tokens=512, contact=10.0):
    """Each accuracy-set complex at `tokens`: whole binder + the target window centred on the interface (target
    residues with a CA within `contact` A of a binder CA in the deposited structure). Writes the input, the
    windowed MSAs, the reference with the target renumbered 1.. over the window, and set512.tsv."""
    import gemmi
    import numpy as np
    acc = Path(__file__).parents[1] / "pfm_accuracy"
    rows = ["pdb\ttarget\tbinder\twhy"]
    for line in (acc / "set.tsv").read_text().splitlines()[1:]:
        pdb, tch, bch, why = line.split("\t")
        (_, T), (_, Bs) = [(s["protein"]["id"], s["protein"]["sequence"])
                           for s in yaml.safe_load((D / "inputs" / f"{pdb}.yaml").read_text())["sequences"]]
        n = tokens - len(Bs)
        st = gemmi.read_structure(str(D / "ref" / f"{pdb}.cif"))
        st.setup_entities()
        pol = {ch.name: [r for r in ch.get_polymer() if r.label_seq and r.find_atom("CA", "*")] for ch in st[0]}
        bx = np.array([r["CA"][0].pos.tolist() for r in pol[bch]])
        iface = [r.label_seq for r in pol[tch]
                 if np.linalg.norm(bx - np.array(r["CA"][0].pos.tolist()), axis=1).min() < contact]
        mid = int(np.median(iface))
        start = min(max(mid - n // 2, 1), len(T) - n + 1)  # 1-based label_seq of the window's first residue
        t = T[start - 1:start - 1 + n]
        name = f"{pdb}_{tokens}"
        (MSA / f"{seq_hash(t)}.a3m").write_text(crop_a3m((MSA / f"{seq_hash(T)}.a3m").read_text(), n, start - 1))
        src, dst = paired_msa_dir(MSA, [T, Bs]), paired_msa_dir(MSA, [t, Bs])
        dst.mkdir(parents=True, exist_ok=True)
        (dst / f"{seq_hash(t)}.a3m").write_text(crop_a3m((src / f"{seq_hash(T)}.a3m").read_text(), n, start - 1))
        shutil.copy(src / f"{seq_hash(Bs)}.a3m", dst / f"{seq_hash(Bs)}.a3m")
        write_yaml(name, [("A", t), ("B", Bs)], f"{pdb} target {start}-{start + n - 1} (interface median "
                   f"{mid}, {len(iface)} contacts) + binder 1-{len(Bs)}, {tokens} tokens, MSA cropped by column")
        doc = gemmi.cif.read(str(D / "ref" / f"{pdb}.cif"))
        site = doc.sole_block().find("_atom_site.", ["label_seq_id", "auth_asym_id"])
        for i in reversed(range(len(site))):
            r = site[i]
            if r[1] != tch:
                continue
            k = int(r[0]) - start + 1 if r[0] not in (".", "?") else 0
            if 1 <= k <= n:
                r[0] = str(k)
            else:
                site.remove_row(i)
        doc.write_file(str(D / "ref" / f"{name}.cif"))
        rows.append(f"{name}\t{tch}\t{bch}\t{pdb} at {tokens} tokens: target {start}-{start + n - 1} + whole binder")
        print(f"{name}: target {start}-{start + n - 1} of {len(T)}, binder {len(Bs)}, {len(iface)} interface residues")
    (acc / f"set{tokens}.tsv").write_text("\n".join(rows) + "\n")


def homomer(name, n, copies):
    s = A[:n]
    (MSA / f"{seq_hash(s)}.a3m").write_text(crop_a3m((MSA / f"{seq_hash(A)}.a3m").read_text(), n))
    write_yaml(name, [(chr(65 + i), s) for i in range(copies)],
               f"complex730 chain A 1-{n} x {copies}; identical chains do not pair")


def boltz2_csvs():
    from tt_bio.main import write_boltz_csvs
    for f in sorted((D / "inputs").glob("*.yaml")):
        seqs = [s["protein"]["sequence"] for s in yaml.safe_load(f.read_text())["sequences"]]
        p = paired_msa_dir(MSA, seqs)
        if p is None or not p.exists():
            continue
        hs = {seq_hash(s) for s in seqs}
        write_boltz_csvs(p, {h: (p / f"{h}.a3m").read_text() for h in hs if (p / f"{h}.a3m").exists()},
                         {h: (MSA / f"{h}.a3m").read_text() for h in hs})


if a.boltz2_csv:
    boltz2_csvs()
    raise SystemExit(0)
if a.truth512:
    truth512()
    boltz2_csvs()
    raise SystemExit(0)
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
    truth512()
else:
    print(f"skip accuracy set: no {a.acc}/msa/cache")

boltz2_csvs()

# Manifest: what every box must hold, byte for byte (the a3m; the Boltz-2 CSVs derive from them).
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
