#!/usr/bin/env python3
"""Build a real-target RF3 fixture: sequence + ColabFold MSA + crystal CA ground truth.

Why real targets. The campaign's only fixture family is `cdk2x2_*`, CDK2 tandem-repeated to a
length. Above 298 aa that is a chimera whose inter-domain geometry has no ground truth
(`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`), so the 512 aa segments of
state/fused-sdpa-adopt.md are scored on within-segment pairs of a protein folded twice, with one
target and one MSA. Nothing in that design can tell a target-specific result from a general one.

Three rules this encodes, each of which a hand-built fixture gets wrong:

1. LENGTH MUST PAD ONTO THE RUNG. RF3 pads the token dim to `PAIRFORMER_PAD_MULTIPLE = 64`, and
   `_sdpa_chunks_shipped` picks the fused kernel's k_chunk from the PADDED length. Two targets of
   480 and 520 aa run different k_chunks, hence different online-softmax chunk counts, hence
   different arithmetic -- pooling them would confound the target axis with the kernel config.
   Every target for a rung must satisfy `rung - 63 <= L <= rung`.

2. RESIDUE INDEXING IS `label_seq_id`, NOT `auth_seq_id`. The fold is handed the entity's canonical
   one-letter sequence, so fold token i is entity position i+1 is mmCIF `label_seq_id == i+1`,
   exactly and with no alignment step. `auth_seq_id` carries the depositor's numbering and does not
   line up. The identity of every scored residue is asserted against the sequence.

3. MSA DEPTH IS A FIXED PARAMETER, NOT WHATEVER THE SERVER RETURNS. Depth changes trunk quality and
   therefore the instrument's sensitivity, so it is pinned to `--depth` for every target and
   recorded in the meta.

The a3m and the ground truth are written to disk and committed, so a rerun never touches the
network and the alignment cannot drift.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

GQL = "https://data.rcsb.org/graphql"
CIF = "https://files.rcsb.org/download/{}.cif.gz"
CF = "https://api.colabfold.com"
PAD = 64

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # the modified residues a canonical sequence writes as their parent
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C", "CME": "C", "KCX": "K",
    "LLP": "K", "MLY": "K", "HYP": "P", "PCA": "Q", "CGU": "E", "SAC": "S", "OCS": "C",
}


def _get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def entity(pdb):
    q = """query($id:String!){entry(entry_id:$id){
      rcsb_entry_info{resolution_combined}
      polymer_entities{rcsb_id
        entity_poly{pdbx_seq_one_letter_code_can rcsb_sample_sequence_length}
        rcsb_polymer_entity{pdbx_description}
        rcsb_polymer_entity_container_identifiers{asym_ids}}}}"""
    req = urllib.request.Request(
        GQL, data=json.dumps({"query": q, "variables": {"id": pdb}}).encode(),
        headers={"Content-Type": "application/json"})
    e = json.load(urllib.request.urlopen(req, timeout=90))["data"]["entry"]
    pes = e["polymer_entities"]
    assert len(pes) == 1, f"{pdb} has {len(pes)} polymer entities; this builder wants exactly one"
    pe = pes[0]
    seq = re.sub(r"\s", "", pe["entity_poly"]["pdbx_seq_one_letter_code_can"])
    asym = pe["rcsb_polymer_entity_container_identifiers"]["asym_ids"]
    assert len(asym) == 1, f"{pdb} entity spans {asym}; this builder wants a single chain"
    return {"seq": seq, "asym": asym[0],
            "resolution": e["rcsb_entry_info"]["resolution_combined"][0],
            "description": pe["rcsb_polymer_entity"]["pdbx_description"]}


def crystal_ca(pdb, asym, seq):
    """{label_seq_id: [x,y,z]} for CA atoms of one chain, identity-checked against `seq`."""
    raw = _get(CIF.format(pdb))
    sha = hashlib.sha256(raw).hexdigest()
    txt = gzip.decompress(raw).decode("utf-8", "replace")
    cols, ca, in_loop = [], {}, False
    for line in txt.splitlines():
        if line.startswith("_atom_site."):
            cols.append(line.strip().split(".", 1)[1])
            in_loop = True
            continue
        if not in_loop:
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            if line.startswith("#"):
                break
            continue
        f = line.split()
        ix = {k: i for i, k in enumerate(cols)}
        if f[ix["label_asym_id"]] != asym:
            continue
        if f[ix["label_atom_id"]].strip('"') != "CA":
            continue
        alt = f[ix["label_alt_id"]]
        if alt not in (".", "?", "A"):
            continue
        s = f[ix["label_seq_id"]]
        if s in (".", "?"):
            continue
        i = int(s)
        if i in ca:            # a second altloc that slipped the filter
            continue
        one = AA3.get(f[ix["label_comp_id"]].upper())
        assert one is not None, f"{pdb} unmapped residue {f[ix['label_comp_id']]} at {i}"
        assert 1 <= i <= len(seq), f"{pdb} label_seq_id {i} outside 1..{len(seq)}"
        assert seq[i - 1] == one, \
            f"{pdb} residue {i} is {one} in the crystal, {seq[i-1]} in the entity sequence"
        ca[i] = [float(f[ix["Cartn_x"]]), float(f[ix["Cartn_y"]]), float(f[ix["Cartn_z"]])]
    return ca, sha


def colabfold_a3m(seq, depth, poll=10, limit=1800):
    """Query length + depth are the only two things this pins; the server picks the rest."""
    body = urllib.parse.urlencode({"q": f">query\n{seq}\n", "mode": "env"}).encode()
    req = urllib.request.Request(f"{CF}/ticket/msa", data=body)
    tid = json.load(urllib.request.urlopen(req, timeout=120))["id"]
    t0 = time.time()
    while True:
        st = json.load(urllib.request.urlopen(f"{CF}/ticket/{tid}", timeout=60))["status"]
        if st == "COMPLETE":
            break
        assert st in ("PENDING", "RUNNING"), f"colabfold ticket {tid} -> {st}"
        assert time.time() - t0 < limit, f"colabfold ticket {tid} still {st} after {limit}s"
        time.sleep(poll)
    tar = tarfile.open(fileobj=io.BytesIO(_get(f"{CF}/result/download/{tid}", timeout=300)))
    heads, rows = [], []
    for name in ("uniref.a3m", "bfd.mgnify30.metaeuk50.smag30.a3m"):
        m = next((x for x in tar.getmembers() if x.name.endswith(name)), None)
        if m is None:
            continue
        ls = tar.extractfile(m).read().decode().rstrip("\n").split("\n")
        ls = [x for x in ls if not x.startswith("#")]
        for h, r in zip(ls[0::2], ls[1::2]):
            assert h.startswith(">"), f"{name} is not strict header/row pairs"
            heads.append(h)
            rows.append(r)
    # the query is row 0 of the first block; drop the duplicate queries the other blocks repeat
    assert rows and rows[0].replace("-", "").upper() == seq, "colabfold query row != our sequence"
    keep_h, keep_r, seen = [heads[0]], [rows[0]], {rows[0]}
    for h, r in zip(heads[1:], rows[1:]):
        if r in seen:
            continue
        seen.add(r)
        keep_h.append(h)
        keep_r.append(r)
        if len(keep_r) >= depth:
            break
    return "\n".join(x for pair in zip(keep_h, keep_r) for x in pair) + "\n", tid


YAML = """version: 1
# {pdb} {desc}. {n} aa, one chain, {res} A X-ray. Folded protein-only: any cofactor is
# absent from both arms, so it cannot bias the A/B. Length pads to {padded} on
# PAIRFORMER_PAD_MULTIPLE = 64, which is the rung this target belongs to.
sequences:
  - protein:
      id: A
      sequence: {seq}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--name", required=True, help="fixture name, e.g. chox_499")
    ap.add_argument("--rung", type=int, required=True, help="padded token count this must land on")
    ap.add_argument("--depth", type=int, default=256, help="MSA rows kept, query included")
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parent / "targets")
    ap.add_argument("--min-coverage", type=float, default=0.90,
                    help="fraction of the entity sequence that must have a modelled CA")
    a = ap.parse_args()
    pdb = a.pdb.upper()

    e = entity(pdb)
    seq, n = e["seq"], len(e["seq"])
    padded = -(-n // PAD) * PAD
    assert padded == a.rung, \
        f"{pdb} is {n} aa -> pads to {padded}, not the requested rung {a.rung}. " \
        f"A rung-{a.rung} target must be {a.rung - PAD + 1}..{a.rung} aa."

    ca, sha = crystal_ca(pdb, e["asym"], seq)
    cov = len(ca) / n
    assert cov >= a.min_coverage, f"{pdb} models {len(ca)}/{n} CAs ({cov:.1%}) < {a.min_coverage:.0%}"

    a3m, tid = colabfold_a3m(seq, a.depth)
    depth = a3m.count("\n>") + 1

    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / f"{a.name}.a3m").write_text(a3m)
    (a.outdir / f"{a.name}.yaml").write_text(YAML.format(
        pdb=pdb, desc=e["description"], n=n, res=e["resolution"], padded=padded, seq=seq))
    (a.outdir / f"{a.name}.gt.json").write_text(json.dumps(
        {"pdb": pdb, "asym": e["asym"], "seq": seq, "n_res": n, "padded": padded,
         "resolution": e["resolution"], "description": e["description"],
         "ca": {str(k): v for k, v in sorted(ca.items())}}, indent=0) + "\n")
    (a.outdir / f"{a.name}.meta.json").write_text(json.dumps(
        {"pdb": pdb, "name": a.name, "rung": a.rung, "n_res": n, "padded": padded,
         "resolution": e["resolution"], "description": e["description"],
         "ca_modelled": len(ca), "ca_coverage": round(cov, 4),
         "cif_url": CIF.format(pdb), "cif_sha256": sha,
         "msa_depth": depth, "msa_depth_requested": a.depth,
         "colabfold_ticket": tid, "colabfold_mode": "env"}, indent=1) + "\n")
    print(f"{a.name}: {pdb} {n} aa -> padded {padded}, {len(ca)} CA ({cov:.1%}), "
          f"MSA depth {depth}, {e['resolution']} A -- {e['description']}")


if __name__ == "__main__":
    main()
