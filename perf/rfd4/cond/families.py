"""The eight protein-only RFD4 condition families, as one table and two commands.

    python perf/rfd4/cond/families.py featurize [--family ID]   # host only, no card
    python perf/rfd4/cond/families.py cmd --family ID           # the tt-bio design command

Each family is an input plus the settings upstream runs it with. Six of the eight are annotated
CIFs upstream ships, taken through ``--annotated_cif`` because that is upstream's own
``inputs=<cif>`` path; the binder is a condition spec; partial diffusion starts from a design a
previous run produced, which is why its input is named rather than shipped.

`featurize` is the card-free half: it builds each family's input into features and reports the
token count, the conditions actually set and the emitted set. A family whose conditions come out
empty is a spec that selected nothing, and that is invisible later.
"""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

UPSTREAM = Path(os.environ.get("RFD4_UPSTREAM", "/home/moritz/rfd4-ref-env/src/RFD4-Proteina"))
HERE = Path(__file__).resolve().parent
#: Inputs are copied here by `stage`, so a design host needs the clone once and never again.
INPUTS = HERE / "inputs"

#: id -> what it takes. `input` is relative to the upstream clone; `spec` names a file in specs/.
#: `recipe` is the recycle schedule upstream picks for that task (cookbook / example configs).
FAMILIES = {
    "unconditional": {
        "input": "data/benchmarks/monomer/monomer-short.cif.zst",
        "annotated_cif": True, "recipe": "thinking_medium", "golden": "monomer_short",
        "what": "one expandable 100-residue chain, no conditioning at all",
    },
    "binder": {
        "spec": "binder.yaml", "spec_input": "docs/examples/inputs/5o45_cropped.cif.zst",
        "recipe": "medium/front_loaded", "golden": "pdl1_binder",
        "what": "PD-L1 binder: target held by C_CRDGRP, its sequence by C_SEQ, three hotspots by C_HOT",
    },
    "antibody_cdr": {
        "spec": "antibody.yaml", "recipe": "high/front_loaded", "golden": None,
        "what": "paired VH/VL Fv against a target, CDRs redesigned; upstream ships no paired-Fv "
                "design input, only VHH (its antibody surface is scoring-side)",
    },
    "nanobody_cdr": {
        "input": "examples/antibody_design/structures/vhh/mzb1_7AAH__VHH_caplacizumab_cdr123.cif.zst",
        "annotated_cif": True, "recipe": "high/front_loaded", "golden": None,
        "what": "VHH framework + MZB1 target with CDR1/2/3 conditioning already annotated",
    },
    "enzyme": {
        "input": "data/benchmarks/mcsa/mcsa_37/M0040_13pk.cif.zst",
        "annotated_cif": True, "recipe": "medium/midpeak", "golden": "mcsa_site",
        "what": "M-CSA catalytic site: motif residues around a ligand, scaffold length preset",
    },
    "motif_scaffold": {
        "input": "data/benchmarks/motif_scaffolding/motif_indexed/1QJG_AA.cif.zst",
        "annotated_cif": True, "recipe": "medium/midpeak", "golden": "motif_1qjg",
        "what": "1QJG indexed motif: C_CRD + C_SEQ on the motif residues, scaffold around them",
    },
    "partial_diffusion": {
        "from": "binder", "annotated_cif": True, "recipe": "medium/front_loaded", "golden": None,
        "args": ["--partial", "0.4166666666666667"],
        "what": "re-noise a design from the binder run to 0.42 of the schedule and run the tail "
                "(upstream's antibody refinement setting)",
    },
    "inverse_folding": {
        "input": "data/benchmarks/motif_scaffolding/motif_data_aa/2kl8_aa.cif.zst",
        "annotated_cif": True, "recipe": "medium/flat", "golden": None,
        "args": ["--inverse_folding", "--tsr_k", "2"],
        "what": "hold 2KL8's backbone exactly, design only the sequence, at upstream's example "
                "temperature",
    },
}


def staged(family: str) -> Path | None:
    """Where this family's input CIF lives once staged, or None if it has no shipped input."""
    rel = FAMILIES[family].get("input")
    return INPUTS / Path(rel).name if rel else None


def stage() -> list[str]:
    """Copy every shipped input out of the upstream clone, so a run needs only this directory."""
    import shutil

    INPUTS.mkdir(parents=True, exist_ok=True)
    done = []
    for name, f in FAMILIES.items():
        rel = f.get("input") or (f.get("spec_input") if "spec" in f else None)
        if not rel:
            continue
        src, dst = UPSTREAM / rel, INPUTS / Path(rel).name
        if not src.exists():
            raise SystemExit(f"{name}: {src} is missing; set RFD4_UPSTREAM to the clone")
        if not dst.exists():
            shutil.copy2(src, dst)
        done.append(f"{name}: {dst}")
    return done


def input_path(family: str, parent_design: Path | None = None) -> Path:
    f = FAMILIES[family]
    if "spec" in f:
        return HERE / "specs" / f["spec"]
    if "from" in f:
        if parent_design is None:
            raise SystemExit(f"{family} starts from a {f['from']} design: pass --parent <cif>")
        return parent_design
    path = staged(family)
    return path if path.exists() else UPSTREAM / f["input"]


def command(family: str, out_dir: Path, *, seed: int, num_designs: int,
            parent_design: Path | None = None) -> list[str]:
    f = FAMILIES[family]
    cmd = ["tt-bio", "design", "--model", "rfd4", str(input_path(family, parent_design)),
           "--out_dir", str(out_dir), "--seed", str(seed), "--num_designs", str(num_designs),
           "--recipe", f["recipe"]]
    if f.get("annotated_cif"):
        cmd.append("--annotated_cif")
    return cmd + list(f.get("args", []))


def featurize(family: str, seed: int, parent_design: Path | None = None) -> dict:
    """Build the family's input into features the way the CLI does, and report what it carries."""
    import torch

    from tt_bio import rfd4

    f = FAMILIES[family]
    src = input_path(family, parent_design)
    if "spec" in f:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            prepared = rfd4.prepare_examples(src, Path(tmp) / "inputs")
            rows = [_one(rfd4, d.example_id, d.cif, seed) for d in prepared]
        return {"family": family, "input": str(src), "designs": rows}
    return {"family": family, "input": str(src),
            "designs": [_one(rfd4, src.name.split(".")[0], src, seed)]}


def _one(rfd4, example_id: str, cif: Path, seed: int) -> dict:
    """The conditioning a featurized input actually carries.

    Counting what is SET, not which keys exist: every key exists in every example, because
    `prepare_cif` fills absent conditions with their defaults. A family whose counts are all zero
    when they should not be is a selection that matched nothing, which is the failure this
    harness exists to catch.
    """
    import torch

    from tt_bio.rfd4.ae_fold import C_SEQ, tables

    features, _ = rfd4.featurize(cif, seed=seed)
    group = features["condition_CoordinateGroup_1_atom"]
    return {
        "example_id": example_id,
        "cif": str(cif),
        "tokens": int(features["padding_mask"].shape[-1]),
        "real_tokens": int(features["padding_mask"].sum()),
        "coord_conditioned_tokens": int(features["coord_cond_mask"].sum()),
        "sequence_pinned_residues": int((features[C_SEQ] != tables()["mask_token"]).sum()),
        "hotspot_atoms": int(features["condition_hotspot_1_atom"].sum()),
        "coordinate_group_atoms": int((group > 0).sum()),
        "internal_coordinate_atoms": int(torch.isfinite(
            features["condition_InternalCoordinate_1_atom"]).sum()) // 3,
        "coordinate_atoms": int(torch.isfinite(features["condition_coordinate_1_atom"]).sum()) // 3,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("stage", "featurize", "cmd"))
    ap.add_argument("--family", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_designs", type=int, default=1)
    ap.add_argument("--out_dir", default="./designs")
    ap.add_argument("--parent", default=None, help="the design partial diffusion restarts from")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    families = [a.family] if a.family else list(FAMILIES)
    parent = Path(a.parent) if a.parent else None

    if a.action == "stage":
        print("\n".join(stage()))
        return 0

    if a.action == "cmd":
        for name in families:
            print(shlex.join(command(name, Path(a.out_dir) / name, seed=a.seed,
                                     num_designs=a.num_designs, parent_design=parent)))
        return 0

    rows = []
    for name in families:
        try:
            rows.append(featurize(name, a.seed, parent))
        except (Exception, SystemExit) as e:  # a family that cannot featurize is the answer
            rows.append({"family": name, "error": f"{type(e).__name__}: {e}"})
        print(json.dumps(rows[-1])[:400], flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
