"""Write state/rfd4/rows/rfd4-ref.json from the rebuilt golden index (row rfd4-ref)."""

import json
from pathlib import Path

INDEX = Path("/home/moritz/.coworker/state/rfd4/goldens/INDEX.json")
OUT = Path("/home/moritz/.coworker/state/rfd4/rows/rfd4-ref.json")

GPU_LEGS = [
    "cuEquivariance kernels (CUDA-only extra)",
    "BioIR inference backend (CUDA 13 only)",
    "regional torch.compile warm-up and any timing number",
    "AF2 / AF3-OpenFold3 refolding oracles and reward-guided search metrics (apptainer GPU images)",
]


def main() -> None:
    idx = json.loads(INDEX.read_text())
    recs = idx["records"]
    row = {
        "status": "done",
        "env": {
            "root": "/home/moritz/rfd4-ref-env",
            "build": "/home/moritz/rfd4-ref-env/build_env.sh (= tt-bio perf/rfd4/ref/build_env.sh)",
            "python": "3.12.12",
            "torch": "2.14.1+cpu",
            "upstream_sha": idx["upstream_sha"],
            "weights": ["nvidia/RFD4-Proteina", "nvidia/RFD4-Proteina-prep"],
            "doc": "state/rfd4/REF.md",
        },
        "modules_captured": sorted({r["module"] for r in recs}),
        "module_classes": idx["modules"],
        "features_captured": idx["features"],
        "seeds": idx["seeds"],
        "captures": len(recs),
        "bytes": idx["bytes"],
        "f64_captures": sum(1 for r in recs if r.get("f64") in (True, "True")),
        "goldens_root": "/home/moritz/rfd4-goldens",
        "index": "state/rfd4/goldens/INDEX.json",
        "gpu_legs_owed": GPU_LEGS,
        "notes": [
            "antibody_cdr: upstream ships no end-to-end IgG example (VHH only plus the ig_design "
            "task preset), so the family is covered through the nanobody CDR capture, same modules.",
            "partial diffusion needs a GENERATED annotated complex, not an input CIF; fed an input "
            "it fails in structure export with 'NaN values detected ... coord group 1'.",
            "~512 tokens uses specs/monomer-512.cif.zst (monomer-long's annotated CIF with its "
            "length fields at 512); module I/O only, a full 512-residue CPU design is hours.",
        ],
    }
    OUT.write_text(json.dumps(row, indent=2) + "\n")
    print(f"wrote {OUT}: {len(recs)} captures, {row['bytes']/1e9:.2f} GB, "
          f"{len(row['modules_captured'])} modules, {len(row['features_captured'])} features")


if __name__ == "__main__":
    main()
