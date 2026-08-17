"""Acceptance harness for the RFD3 designed-sequence output.

RFD3's checkpoint carries a sequence head and the port runs it every diffusion
step; this verifies the predicted residue identities actually reach the output
CIF, and that threading them through changed nothing else in the compute path:

  1. bit-identity: coordinates of a pre-fix and post-fix run, same seed/spec,
     must be torch.equal (the change is plumbing only, zero new compute).
  2. rfd3-scaffold (A1-10,20,A31-40): zero UNK among the 20 designed positions;
     the 20 motif residues keep their input names.
  3. rfd3-binder (A1-150,60-80): over the designed positions only (never the
     motif-diluted whole chain), no amino acid above 40% and at least 8
     distinct types.
  4. rfd3-na-binder (A1-12,B13-24,60-80): the DNA motif chains keep their
     nucleotide names; the designed positions come back as amino acids.

Specs are the platform's own acceptance payloads (structure + contig) from
perf/wh-correctness/results/payloads/.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_designed_sequence.py run --tag before
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/verify_designed_sequence.py run --tag after
  python3 scripts/rfd3_port/verify_designed_sequence.py check
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PAYLOADS = ROOT / "perf/wh-correctness/results/payloads"
WORK = ROOT / "perf/rfd3-designed-sequence"
ACCEPT = ROOT / "state/tt-bio-rfd3-designed-sequence_accept.json"
CHECKPOINT = Path("~/.boltz/rfd3/weights").expanduser()

SPECS = {
    "binder": "des_rfd3_binder.json",
    "scaffold": "des_rfd3_scaffold.json",
    "na_binder": "des_rfd3_na.json",
}
AA20 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def chain_label(asym: int) -> str:
    # mirror of design._chain_label for the small asym ids these specs use
    if asym < 26:
        return chr(ord("A") + asym)
    return chr(ord("A") + asym // 26 - 1) + chr(ord("A") + asym % 26)


def run(tag: str, timesteps: int, seed: int):
    from tt_bio.rfd3 import design as design_mod

    work = WORK / tag
    work.mkdir(parents=True, exist_ok=True)
    specs, captured = {}, {}
    for sid, payload in SPECS.items():
        d = json.loads((PAYLOADS / payload).read_text())
        pdb = work / f"{sid}.pdb"
        pdb.write_text(d["structure"])
        specs[sid] = {"input": str(pdb), "contig": d["contig"]}

    orig_write_cif = design_mod._write_cif

    def wrapper(coords, f, out_path, **kw):
        captured[Path(out_path).stem] = (coords.detach().clone(), f)
        return orig_write_cif(coords, f, out_path, **kw)

    design_mod._write_cif = wrapper
    design_mod.run_design(specs, work / "cif", checkpoint_dir=str(CHECKPOINT),
                          from_pdb=True, num_timesteps=timesteps, seed=seed,
                          num_designs=1, verbose=True)
    (work / "run_meta.json").write_text(json.dumps({"timesteps": timesteps, "seed": seed}))

    from tt_bio.rfd3.featurize import DESIGNED_RESTYPE_IDX, _RESTYPE_ORDER
    for sid, (X, f) in captured.items():
        rt = f["restype"].argmax(-1) if f["restype"].ndim == 2 else f["restype"]
        rt = rt.tolist()
        tokens = {
            "designed": [int(i) == DESIGNED_RESTYPE_IDX for i in rt],
            "expected": [_RESTYPE_ORDER[int(i)] if 0 <= int(i) < len(_RESTYPE_ORDER) else "?"
                         for i in rt],
            "chain": [chain_label(int(a)) for a in f["asym_id"].tolist()],
            "resid": [int(r) for r in f["residue_index"].tolist()],
            "is_protein": [bool(v) for v in f["is_protein"].tolist()],
        }
        torch.save({"X": X, "tokens": tokens}, work / f"{sid}.pt")
    print(f"[run:{tag}] captured {len(captured)} designs -> {work}", flush=True)


def parse_cif(path: Path):
    """atom_site loop -> (residues {(chain, res_id): comp}, atoms [(chain, res_id, comp, atom)])."""
    lines = path.read_text().splitlines()
    residues, atoms = {}, []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].lstrip().startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            if cols and all(c.startswith("_atom_site.") for c in cols):
                name = {c.split(".", 1)[1]: k for k, c in enumerate(cols)}

                def col(*cands):
                    for c in cands:
                        if c in name:
                            return name[c]
                    raise KeyError(cands)

                ci = col("label_comp_id", "auth_comp_id")
                ca = col("label_asym_id", "auth_asym_id")
                cs = col("label_seq_id", "auth_seq_id")
                ct = col("label_atom_id", "auth_atom_id")
                for row in lines[j:]:
                    if not row.strip() or row.strip() == "loop_" or row.lstrip().startswith("_") or row.startswith("#"):
                        break
                    parts = [p.strip("'\"") for p in row.split()]
                    if len(parts) < len(cols):
                        continue
                    residues[(parts[ca], parts[cs])] = parts[ci]
                    atoms.append((parts[ca], parts[cs], parts[ci], parts[ct]))
                return residues, atoms
            i = j
        else:
            i += 1
    raise ValueError(f"no atom_site loop in {path}")


def check():
    per_spec, bit_identical = {}, True
    for sid in SPECS:
        before = torch.load(WORK / "before" / f"{sid}.pt", weights_only=True)
        after = torch.load(WORK / "after" / f"{sid}.pt", weights_only=True)
        eq = bool(torch.equal(before["X"], after["X"]))
        bit_identical &= eq

        tok = after["tokens"]
        residues, atoms = parse_cif(WORK / "after" / "cif" / f"{sid}.cif")
        actual = [(residues.get((tok["chain"][t], str(tok["resid"][t]))), t)
                  for t in range(len(tok["designed"]))]
        designed = [(name, t) for name, t in actual if tok["designed"][t]]
        motif = [(name, t) for name, t in actual if not tok["designed"][t]]
        n_unk = sum(1 for name, _ in designed if name == "UNK")
        non_aa = [name for name, _ in designed if name not in AA20]
        motif_bad = [(t, tok["expected"][t], name) for name, t in motif
                     if name != tok["expected"][t]]
        counts = Counter(name for name, _ in designed if name in AA20)
        top_share = max(counts.values()) / len(designed) if counts and designed else 1.0
        gly_cb = [t for name, t in designed if name == "GLY"
                  and any(a[0] == tok["chain"][t] and a[1] == str(tok["resid"][t]) and a[3] == "CB"
                          for a in atoms)]
        per_spec[sid] = {
            "bit_equal": eq, "n_atoms": int(after["X"].shape[0]),
            "n_designed": len(designed), "designed_unk": n_unk,
            "designed_non_aa": sorted(set(n for n in non_aa if n)),
            "distinct_aa": len(counts), "top_aa_share": round(top_share, 4),
            "top_aa": counts.most_common(3), "motif_mismatch": motif_bad,
            "gly_with_cb": gly_cb, "aa_counts": dict(counts),
        }
        print(f"[{sid}] bit_equal={eq} designed={len(designed)} unk={n_unk} "
              f"distinct_aa={len(counts)} top={counts.most_common(3)} "
              f"motif_mismatch={len(motif_bad)} gly_with_cb={len(gly_cb)}", flush=True)

    s, b, n = per_spec["scaffold"], per_spec["binder"], per_spec["na_binder"]
    verdict = {
        "bit_identical": bit_identical,
        "scaffold_zero_unk": s["designed_unk"] == 0 and not s["motif_mismatch"],
        "binder_diverse": b["top_aa_share"] <= 0.40 and b["distinct_aa"] >= 8
                          and b["designed_unk"] == 0,
        "na_binder_amino_acids": n["designed_unk"] == 0 and not n["designed_non_aa"]
                                 and not n["motif_mismatch"],
    }
    meta = json.loads((WORK / "after" / "run_meta.json").read_text())
    out = {**meta, **verdict, "evidence": per_spec}
    ACCEPT.parent.mkdir(exist_ok=True)
    ACCEPT.write_text(json.dumps(out, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"[check] wrote {ACCEPT}", flush=True)
    if not all(verdict.values()):
        sys.exit(1)


TIMESTEPS = 4
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "check"])
    ap.add_argument("--tag", choices=["before", "after"], default=None)
    ap.add_argument("--timesteps", type=int, default=TIMESTEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.mode == "run":
        if args.tag is None:
            ap.error("run needs --tag before|after")
        run(args.tag, args.timesteps, args.seed)
    else:
        check()


if __name__ == "__main__":
    main()
