"""A real design population for the filter-tolerance question.

Passes 7 and 8 built two populations and both were flat. The scramble ladder cut one chain out of
a sequence-contiguous crop, so there was no packed interface to grade; the pose ladder moved the
initial guess, which `rm_template_ic=True` strips from the template and four recycles forget.
Neither had mass anywhere near the filter's accept line.

This builds the population from the input PXDesign's AF2-IG stage actually sees: a generated
binder backbone against a real target, sequences assigned by ProteinMPNN. The backbone comes from
a BoltzGen design run on disk (`~/boltzgen_designability_run`, a 119-residue nanobody against a
131-residue target, so a real designed interface), the target is cropped to its epitope so a score
costs what the 208-token parity fixture costs, and MPNN's sampling temperature is the one knob:
low temperature gives sequences that fit the backbone, high temperature gives sequences that do
not, and AF2-IG's confidence follows. That sweeps the accept line with the same kind of input the
production filter grades.

    env/bin/python3 scripts/af2_port/design_population.py --out /tmp/af2ig_pop
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from af2_fixture import AA3TO1, seq_of, write_complex_pdb  # noqa: E402

MPNN = Path.home() / "scratch" / "ProteinMPNN"
DESIGNS = [Path.home() / "boltzgen_designability_run" / "intermediate_designs" / "binder_1.cif",
           Path.home() / "boltzgen_designability_run" / "intermediate_designs" / "binder.cif"]
DESIGN_CHAIN = "A"     # the generated binder in a BoltzGen design cif
TARGET_CHAIN = "B"


def read_cif_chains(path: Path) -> dict[str, list[dict]]:
    """Parse an mmCIF `_atom_site` loop into {chain: ordered residues}, columns by name."""
    cols: list[str] = []
    chains: dict[str, dict[int, dict]] = {}
    order: dict[str, list[int]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            cols.append(line.split(".", 1)[1].strip())
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        rec = dict(zip(cols, line.split()))
        if rec.get("type_symbol") == "H":
            continue
        ch = rec["label_asym_id"]
        key = int(rec["label_seq_id"])
        res = chains.setdefault(ch, {})
        if key not in res:
            res[key] = {"comp": rec["label_comp_id"], "atoms": []}
            order.setdefault(ch, []).append(key)
        res[key]["atoms"].append((rec["label_atom_id"], rec["type_symbol"],
                                  float(rec["Cartn_x"]), float(rec["Cartn_y"]),
                                  float(rec["Cartn_z"])))
    return {ch: [chains[ch][k] for k in order[ch] if chains[ch][k]["comp"] in AA3TO1]
            for ch in chains}


def coords(res_list: list[dict]) -> np.ndarray:
    return np.array([[a[2], a[3], a[4]] for r in res_list for a in r["atoms"]])


def epitope_crop(target: list[dict], binder: list[dict], cut: float, cap: int) -> tuple[list[dict], dict]:
    """The contiguous target window that carries the interface, capped at `cap` residues.

    Cropping the target is what the production targets already are (`laczc_128` is a crop of
    `laczc_256`), and it holds a score at the parity fixture's cost. The window is contiguous so
    the crop is still a folded piece, not a residue selection.
    """
    b = coords(binder)
    contact = [i for i, r in enumerate(target)
               if np.linalg.norm(coords([r])[:, None, :] - b[None, :, :], axis=-1).min() < cut]
    if not contact:
        raise SystemExit("no target residue within %.1f A of the binder" % cut)
    lo, hi = min(contact), max(contact) + 1
    while hi - lo < cap and (lo > 0 or hi < len(target)):
        if hi - lo >= cap:
            break
        if lo > 0:
            lo -= 1
        if hi - lo < cap and hi < len(target):
            hi += 1
    if hi - lo > cap:                      # the contact span itself is wider than the cap
        mid = (min(contact) + max(contact)) // 2
        lo, hi = max(0, mid - cap // 2), min(len(target), max(0, mid - cap // 2) + cap)
    return target[lo:hi], {"crop": [lo, hi], "contacts": len(contact),
                           "contact_span": [min(contact), max(contact)]}


def run_mpnn(pdb: Path, design_chain: str, out: Path, temps: list[float], per_temp: int,
             seed: int, weights: str) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(MPNN / "protein_mpnn_run.py"),
           "--pdb_path", str(pdb), "--pdb_path_chains", design_chain,
           "--out_folder", str(out), "--num_seq_per_target", str(per_temp),
           "--sampling_temp", " ".join("%g" % t for t in temps),
           "--seed", str(seed), "--batch_size", "1",
           "--path_to_model_weights", str(MPNN / weights), "--model_name", "v_48_020"]
    subprocess.run(cmd, check=True, cwd=str(MPNN))
    fa = next((out / "seqs").glob("*.fa"))
    rows, header = [], None
    for line in fa.read_text().splitlines():
        if line.startswith(">"):
            header = line[1:]
            continue
        if header is None:
            continue
        fields = dict(kv.split("=", 1) for kv in header.split(", ") if "=" in kv)
        rows.append({"seq": line.strip(), "native": "T" not in fields,
                     "temp": float(fields.get("T", 0.0)), "sample": int(fields.get("sample", 0)),
                     "mpnn_score": float(fields.get("score", 0.0))})
        header = None
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", default=None, help="comma-separated design cifs")
    ap.add_argument("--epitope-cut", type=float, default=10.0)
    ap.add_argument("--target-cap", type=int, default=88)
    ap.add_argument("--temps", default="0.1,0.2,0.3,0.5")
    ap.add_argument("--per-temp", type=int, default=3)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--weights", default="vanilla_model_weights")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    designs = [Path(p) for p in args.designs.split(",")] if args.designs else DESIGNS
    temps = [float(t) for t in args.temps.split(",")]

    manifest = []
    for design in designs:
        chains = read_cif_chains(design)
        binder, target = chains[DESIGN_CHAIN], chains[TARGET_CHAIN]
        crop, info = epitope_crop(target, binder, args.epitope_cut, args.target_cap)
        pdb = out / (design.stem + "_complex.pdb")
        atoms = write_complex_pdb(str(pdb), crop, binder, shift=0.0)
        native = seq_of(binder)
        rows = run_mpnn(pdb, "B", out / (design.stem + "_mpnn"), temps, args.per_temp,
                        args.seed, args.weights)
        for row in rows:
            seq = row["seq"].split("/")[-1] if "/" in row["seq"] else row["seq"]
            if len(seq) != len(native):
                cand = [s for s in row["seq"].split("/") if len(s) == len(native)]
                if not cand:
                    raise SystemExit("cannot locate the binder segment in %r" % row["seq"])
                seq = cand[-1]
            manifest.append({
                "design": design.stem, "pdb": str(pdb), "tokens": len(crop) + len(binder),
                "target_len": len(crop), "binder_len": len(binder), "atoms": atoms,
                "crop": info, "seq": seq, "native_seq": native,
                "identity": round(sum(a == b for a, b in zip(seq, native)) / len(native), 4),
                "temp": row["temp"], "sample": row["sample"], "mpnn_score": row["mpnn_score"],
                "source": "mpnn_native" if row["native"] else "mpnn",
            })
        manifest.append({
            "design": design.stem, "pdb": str(pdb), "tokens": len(crop) + len(binder),
            "target_len": len(crop), "binder_len": len(binder), "atoms": atoms,
            "crop": info, "seq": native, "native_seq": native, "identity": 1.0,
            "temp": 0.0, "sample": -1, "mpnn_score": 0.0, "source": "boltzgen",
        })

    path = out / "population.jsonl"
    with path.open("w") as fh:
        for i, row in enumerate(manifest):
            row["id"] = "%s_t%g_s%d" % (row["design"], row["temp"], row["sample"])
            fh.write(json.dumps(row) + "\n")
    print(json.dumps({"population": str(path), "rows": len(manifest),
                      "tokens": sorted({r["tokens"] for r in manifest})}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
