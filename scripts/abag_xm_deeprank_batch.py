#!/usr/bin/env python3
"""Batched DeepRank-Ab scoring for abag-xm folds.

Builds one ensemble PDB (MODEL/ENDMDL + REMARK lines), runs `deeprank-ab-predict` ONCE, parses the
output CSV, and writes {rank: predicted_dockq} JSON.

Two modes:

    --fold_dir/--target/--gen/--out_json   one fold, 50 models in one CLI call
    --manifest folds.json                  SEVERAL folds in one CLI call

The single-fold mode replaced a per-sample driver path (50 CLI calls x ~30-60 s ESM reload each =
~30-50 min/fold). The manifest mode goes one step further, because measurement showed the
remaining cost is still nearly half fixed: on qb1 under folding contention,
T(n) = 79.3 s + 1.88 s/model (fit over n = 5/25/50, residuals within 3.2 s), so 46% of a 50-model
fold is overhead paid again for every fold. Scoring 5 folds per invocation cuts 173 s/fold to
110 s/fold, which over the 492-fold slab is 23.7 h -> 15.0 h.

Every fold in one manifest must declare the same chains, because the chain IDs are flags on the
invocation rather than per-model; the caller groups by signature and the script refuses a mixed
manifest instead of scoring half the models against the wrong interface.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO


def _cif_to_pdb_lines(cif_path, target):
    """Convert a sample CIF to a list of PDB ATOM/END lines (no MODEL wrapper)."""
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure(target, str(cif_path))
    io = PDBIO()
    io.set_structure(struct)
    buf = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    buf.close()
    io.save(buf.name)
    with open(buf.name) as f:
        lines = f.readlines()
    os.unlink(buf.name)
    # keep only ATOM/HETATM/TER/END; drop HEADER/CRYST1 (ensemble has one set)
    out = [ln for ln in lines if ln[:6] in ("ATOM  ", "HETATM", "TER\n", "TER \n",
                                            "END\n", "END \n")]
    # drop trailing END (we add per-model ENDMDL instead)
    out = [ln for ln in out if ln.strip() != "END"]
    return out


def _declared_chains(target):
    """{"A": antigen, "H": heavy, "L": light} as declared in this target's campaign YAML.

    Returns {} if the YAML cannot be read, so the caller falls back to auto-detection rather
    than failing -- a missing chain hint should degrade, not break.
    """
    y = Path(__file__).resolve().parent.parent / "examples" / "abag_xm" / f"{target}.yaml"
    try:
        import yaml as _yaml
        doc = _yaml.safe_load(y.read_text())
    except Exception:
        return {}
    out = {}
    for s in doc.get("sequences", []) or []:
        for _k, v in (s or {}).items():
            cid = (v or {}).get("id")
            if cid in ("A", "H", "L"):
                out[cid] = cid
    return out


def _sample_cifs(fold_dir, target):
    """[(rank, cif)] for the samples present, rank 0 being the unsuffixed winner."""
    sdir = Path(fold_dir) / "structures"
    cifs = [sdir / f"{target}.cif"] + [sdir / f"{target}_model_{k}.cif"
                                       for k in range(1, 50)]
    present = [(k, c) for k, c in enumerate(cifs) if c.exists()]
    if not present:
        raise RuntimeError(f"no sample CIFs in {sdir}")
    return present


def build_ensemble_pdb(fold_dir, target, out_pdb):
    """Concatenate all 50 sample CIFs into one ensemble PDB with REMARK names."""
    present = _sample_cifs(fold_dir, target)
    with open(out_pdb, "w") as f:
        for k, cif in present:
            name = f"r{k}"
            f.write(f"REMARK  4      MODEL     {k+1} FROM x/{name}.pdb\n")
            f.write(f"MODEL        {k+1}\n")
            for ln in _cif_to_pdb_lines(cif, target):
                f.write(ln if ln.endswith("\n") else ln + "\n")
            f.write("ENDMDL\n")
        f.write("END\n")
    return [k for k, _ in present]


def build_multi_ensemble_pdb(folds, out_pdb):
    """Concatenate SEVERAL folds' samples into one ensemble, named f<fold>r<rank>.

    46% of a 50-model invocation is fixed cost (measured: T(n) = 79.3 s + 1.88 s/model on qb1
    under folding contention), all of which is paid again for every fold when each is scored on
    its own. Putting several folds in one ensemble pays it once. The model identity has to survive
    into the CSV for the scores to be split back apart, which is why the REMARK name carries the
    fold index as well as the rank.
    """
    per_fold = []
    n_model = 0
    with open(out_pdb, "w") as f:
        for i, fd in enumerate(folds):
            present = _sample_cifs(fd["fold_dir"], fd["target"])
            for k, cif in present:
                n_model += 1
                name = f"f{i}r{k}"
                f.write(f"REMARK  4      MODEL     {n_model} FROM x/{name}.pdb\n")
                f.write(f"MODEL        {n_model}\n")
                for ln in _cif_to_pdb_lines(cif, fd["target"]):
                    f.write(ln if ln.endswith("\n") else ln + "\n")
                f.write("ENDMDL\n")
            per_fold.append([k for k, _ in present])
        f.write("END\n")
    return per_fold


def find_predictions_csv(stdout, work_dirs):
    """Locate the predictions CSV from CLI stdout or by scanning work dirs."""
    # stdout often has "Output written → <path>" or "predictions.csv"
    for line in stdout.splitlines():
        m = re.search(r"([^\s]+predictions\.csv)", line)
        if m:
            p = Path(m.group(1))
            if p.exists():
                return p
        m = re.search(r"→\s*([^\s]+\.csv)", line)
        if m:
            p = Path(m.group(1))
            if p.exists():
                return p
    # scan recent work dirs for *predictions.csv
    import glob
    for wd in work_dirs:
        for csv in sorted(glob.glob(str(wd) + "**/*predictions.csv", recursive=True),
                          key=os.path.getmtime, reverse=True):
            return Path(csv)
    return None


def _yaml_sequences(target):
    """{declared_id: sequence} from the campaign YAML, for the ids we care about."""
    y = Path(__file__).resolve().parent.parent / "examples" / "abag_xm" / f"{target}.yaml"
    out = {}
    try:
        import yaml as _yaml
        doc = _yaml.safe_load(y.read_text())
    except Exception:
        return out
    for ent in (doc or {}).get("sequences", []) or []:
        for v in ent.values():
            v = v or {}
            if v.get("id") in ("A", "H", "L") and v.get("sequence"):
                out[v["id"]] = v["sequence"]
    return out


def _cif_chain_sequences(cif):
    """{chain_id: one-letter sequence} for the first model of a CIF."""
    from Bio.PDB.Polypeptide import protein_letters_3to1
    st = MMCIFParser(QUIET=True).get_structure("x", str(cif))
    out = {}
    for ch in next(st.get_models()).get_chains():
        s = []
        for res in ch:
            nm = res.get_resname().strip().upper()
            s.append(protein_letters_3to1.get(nm.capitalize(),
                                              protein_letters_3to1.get(nm, "X")))
        out[ch.id] = "".join(s)
    return out


def _resolve_chains(fold_dir, target):
    """{declared_id: ACTUAL chain id in this fold's CIFs}, or {} if it cannot be resolved.

    The YAML's declared ids are NOT what the generators write. Measured across 76 complete folds:
    boltz2 labels chains A/H/L (23/23 match the declaration) while opendde-abag (0/24) and
    protenix-v2 (0/29) label them A/B/C -- so passing the declared ids straight through made
    deeprank-ab-predict die with "Heavy chain 'H' not found (have ['A','B','C'])" on 53 of 76
    folds, i.e. two of the three generators. The earlier "declared chains" work was verified on a
    boltz2 fold, the one generator where the assumption happens to hold.

    Resolve by SEQUENCE rather than by position: match each declared sequence to the CIF chain
    carrying it. Across all 53 mismatching folds this agrees exactly with the positional guess
    (A->A, H->B, L->C), but matching on sequence is what makes it a fact per fold rather than a
    pattern, and it refuses instead of guessing when two chains are indistinguishable.
    """
    declared = _yaml_sequences(target)
    if not declared:
        return {}
    present = _sample_cifs(fold_dir, target)
    try:
        cif_seqs = _cif_chain_sequences(present[0][1])
    except Exception:
        return {}
    resolved = {}
    for did, dseq in declared.items():
        hits = [cid for cid, cs in cif_seqs.items() if cs == dseq]
        if len(hits) != 1:
            # Ambiguous or absent: a wrong chain pair silently measures a different interface,
            # which is exactly the failure the declared-chain work existed to prevent.
            return {}
        resolved[did] = hits[0]
    return resolved


def _chain_args(target, fold_dir):
    """(cli flags, map) for this fold's antibody/antigen chains, or ([], {}) to let ANARCI guess."""
    # Every other label -- DockQ, epitope Jaccard, interface lDDT, CDR RMSD -- is computed on the
    # DECLARED antibody-antigen interface, so a ranker scoring a different chain pair would not be
    # measuring the same thing. Auto-detection is a known failure mode here too: the DockQ
    # auto-mapper once picked a non-contacting pair and returned a confident zero.
    ids = _resolve_chains(fold_dir, target)
    if not (ids.get("A") and ids.get("H")):
        return [], {}
    flags = ["--antigen_chain_id", ids["A"], "--heavy_chain_id", ids["H"]]
    if ids.get("L"):
        flags += ["--light_chain_id", ids["L"]]
    return flags, ids


def _run_cli(cli, ensemble, chain_flags, work, stdout_tail=1500):
    """Run deeprank-ab-predict once and return the predictions CSV path."""
    # deeprank-ab-predict auto-detects chains via ANARCI, which needs hmmscan/hmmsearch
    # (HMMER3 built from source into ~/.local/bin). Prepend it to PATH so the subprocess
    # works regardless of the parent env -- auto-detection is still the fallback.
    env = {**os.environ,
           "PATH": os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")}
    r = subprocess.run([str(cli), str(ensemble)] + chain_flags, capture_output=True,
                       text=True, cwd=str(Path(work)), env=env)
    print(r.stdout[-stdout_tail:])
    if r.returncode != 0:
        print(f"[deeprank-batch] CLI failed rc={r.returncode}", file=sys.stderr)
        print(r.stderr[-1500:], file=sys.stderr)
        sys.exit(1)

    import glob
    csv_path = find_predictions_csv(r.stdout, [work, "/tmp", "/tmp/tmp*"])
    if csv_path is None:
        cs = sorted(glob.glob("/tmp/**/*predictions.csv", recursive=True),
                    key=os.path.getmtime, reverse=True)
        if cs:
            csv_path = Path(cs[0])
    if csv_path is None:
        print("[deeprank-batch] no predictions CSV found", file=sys.stderr)
        sys.exit(2)
    print(f"[deeprank-batch] predictions CSV: {csv_path}", flush=True)
    return csv_path


def _read_csv_scores(csv_path, pattern):
    """[(groups, score)] for every row whose pdb_id matches `pattern`.

    The CSV is NOT written in rank order (observed: r4, r1, r0, ...), so there is deliberately no
    row-order fallback: if the identifier stops parsing, mapping by position would hand every
    sample a confidently wrong label rather than fail. Callers treat a short read as fatal.
    """
    import csv as csvmod
    out = []
    unparsed = 0
    with open(csv_path) as f:
        for row in csvmod.DictReader(f):
            pid = row.get("pdb_id") or row.get("mol") or ""
            m = re.match(pattern, pid)
            if not m:
                unparsed += 1
                continue
            v = row.get("predicted_dockq") or row.get("dockq")
            if v is None:
                unparsed += 1
                continue
            out.append((m.groups(), float(v)))
    if unparsed:
        print(f"[deeprank-batch] WARNING: {unparsed} CSV rows did not parse against "
              f"{pattern!r}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir")
    ap.add_argument("--target")
    ap.add_argument("--gen")
    ap.add_argument("--out_json")
    # Several folds in one CLI invocation, to pay the fixed cost once. JSON list of
    # {"target", "gen", "fold_dir", "out_json"}. Every fold in one manifest must declare the same
    # chains, because the chain IDs are per-invocation flags, not per-model -- the caller groups.
    ap.add_argument("--manifest",
                    help="JSON list of folds to score in ONE invocation (see --help notes)")
    ap.add_argument("--deeprank_venv",
                    default=os.path.expanduser("~/.deeprank_ab_venv"))
    args = ap.parse_args()

    cli = Path(args.deeprank_venv) / "bin" / "deeprank-ab-predict"

    if args.manifest:
        if any((args.fold_dir, args.target, args.gen, args.out_json)):
            sys.exit("--manifest is exclusive with --fold_dir/--target/--gen/--out_json")
        return _main_multi(cli, json.load(open(args.manifest)))
    missing = [f for f in ("fold_dir", "target", "gen", "out_json") if not getattr(args, f)]
    if missing:
        sys.exit(f"need --{' --'.join(missing)} (or --manifest)")
    return _main_single(cli, args)


def _main_single(cli, args):
    work = tempfile.mkdtemp(prefix=f"deeprank_batch_{args.target}_{args.gen}_")
    ensemble = Path(work) / f"{args.target}_{args.gen}_ensemble.pdb"
    ranks = build_ensemble_pdb(args.fold_dir, args.target, ensemble)
    print(f"[deeprank-batch] {args.target}/{args.gen}: {len(ranks)} models -> "
          f"{ensemble.name}", flush=True)

    flags, ids = _chain_args(args.target, args.fold_dir)
    if ids:
        print(f"[deeprank-batch] declared chains: {ids}", flush=True)
    else:
        print(f"[deeprank-batch] {args.target}: no declared chains found, "
              f"falling back to ANARCI auto-detection", flush=True)

    csv_path = _run_cli(cli, ensemble, flags, work)
    scores = {str(int(g[0])): v for g, v in _read_csv_scores(csv_path, r"r(\d+)$")}
    if len(scores) != len(ranks):
        # Previously this path fell back to mapping rows onto ranks by position. The CSV is not
        # rank-ordered, so that fallback would have handed every sample a confidently wrong
        # label; a short read is a hole to rescore, not something to fill in.
        print(f"[deeprank-batch] {args.target}/{args.gen}: got {len(scores)}/{len(ranks)} "
              f"scores -- NOT written", file=sys.stderr)
        sys.exit(3)
    with open(args.out_json, "w") as f:
        json.dump(scores, f)
    print(f"[deeprank-batch] wrote {len(scores)} scores -> {args.out_json}", flush=True)


def _main_multi(cli, folds):
    if not folds:
        sys.exit("--manifest is empty")
    for fd in folds:
        for k in ("target", "gen", "fold_dir", "out_json"):
            if not fd.get(k):
                sys.exit(f"manifest entry missing {k!r}: {fd}")

    # The chain IDs are flags on the invocation, so a manifest mixing an A+H target with an
    # A+H+L one would score half its models against the wrong interface. Refuse rather than
    # silently produce labels that are not comparable to every other label in the dataset.
    sigs = {}
    for fd in folds:
        flags, ids = _chain_args(fd["target"], fd["fold_dir"])
        sigs.setdefault(tuple(flags), []).append(fd["target"])
    if len(sigs) != 1:
        lines = "\n".join(f"  {list(v)} -> {list(k)}" for k, v in sigs.items())
        sys.exit("manifest mixes RESOLVED chain signatures (generators label chains "
                 "differently -- boltz2 uses A/H/L, protenix/opendde use A/B/C, so a mixed-"
                 "generator manifest lands here); group the folds by signature and run one "
                 f"invocation per group:\n{lines}")
    flags = list(next(iter(sigs)))
    if not flags:
        sys.exit("no declared chains for these targets -- run them singly so ANARCI "
                 "auto-detection stays per-target")

    tag = f"{folds[0]['target']}_x{len(folds)}"
    work = tempfile.mkdtemp(prefix=f"deeprank_multi_{tag}_")
    ensemble = Path(work) / f"{tag}_ensemble.pdb"
    per_fold = build_multi_ensemble_pdb(folds, ensemble)
    total = sum(len(r) for r in per_fold)
    print(f"[deeprank-batch] {len(folds)} folds, {total} models -> {ensemble.name}", flush=True)
    print(f"[deeprank-batch] shared declared chains: {list(flags)}", flush=True)

    csv_path = _run_cli(cli, ensemble, flags, work)
    got = _read_csv_scores(csv_path, r"f(\d+)r(\d+)$")
    split = {}
    for (fi, rk), v in got:
        split.setdefault(int(fi), {})[str(int(rk))] = v

    rc = 0
    for i, fd in enumerate(folds):
        scores = split.get(i, {})
        want = len(per_fold[i])
        if len(scores) != want:
            # A short fold is a hole in the dataset, not something to paper over -- write nothing
            # for it so a later pass rescores it, and make the invocation fail loudly.
            print(f"[deeprank-batch] {fd['target']}/{fd['gen']}: got {len(scores)}/{want} "
                  f"scores -- NOT written", file=sys.stderr)
            rc = 3
            continue
        with open(fd["out_json"], "w") as f:
            json.dump(scores, f)
        print(f"[deeprank-batch] {fd['target']}/{fd['gen']}: wrote {len(scores)} scores -> "
              f"{fd['out_json']}", flush=True)
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
