"""1HCL CA-lDDT per domain for every diffusion sample of cdk2x2_512 bench runs, one line per seed.

    python perf/spd_wherr/hcl_samples.py RUN_DIR [RUN_DIR ...]

Columns: the ranked output (cdk2x2_512.cif), then samples 1-4. Separates "every sample moved" from "the
ranking picked another sample", which hcl_table.py (ranked output only) cannot.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "b2z2_fusebias"))
from score import GT, native  # noqa: E402
from of3_score_ref import ca_map  # noqa: E402


def main():
    gt = ca_map(GT)
    for run in sys.argv[1:]:
        for d in sorted(Path(run).glob("struct_cdk2x2_512_s*")):
            row = []
            for f in ["cdk2x2_512.cif"] + [f"cdk2x2_512_model_{i}.cif" for i in range(1, 5)]:
                if (d / f).exists():
                    row.append("/".join(f"{v['lddt_ca']:.3f}" for v in native(d / f, 512, gt).values()))
            print(f"{Path(run).name:10s} {d.name.rsplit('_', 1)[1]:5s}", "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
