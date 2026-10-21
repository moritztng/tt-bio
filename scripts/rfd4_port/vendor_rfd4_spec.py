"""Add RFD4's condition-spec loader to the vendored rfd4_prep tree.

``rfd4_proteina_prep`` vendors the featurization datapipe but not the spec loader:
``inference_utils/condition_spec.py`` and ``contig_spec.py``, which turn a design YAML or
JSON into the annotated AtomArray ``prepare_cif`` then reads. tt-bio needs both, because
``tt-bio design --model rfd4`` takes a spec file, not a pre-annotated CIF.

Rather than transcribe them, this runs the prep package's OWN flattening tool
(``tools/vendor_from_main.py`` in the prep clone) with the two loaders added to its roots,
against the RFD4 pin. The result is byte-identical to the shipped ``_cif`` modules for the
39 the prep package already carries, which is checked here, so the six new ones are
produced by the same rule rather than by hand.

    python scripts/rfd4_port/vendor_rfd4_spec.py --repo . \\
        --prep-repo /home/moritz/rfd4-prep-src/repo \\
        --rfd4 /home/moritz/rfd4-upstream/RFD4-Proteina [--check]

The pin is read from state/rfd4/UPSTREAM.json's sha; every RFD4 row shares it.
"""
import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rf3_port"))
import vendor_rf3 as vr  # noqa: E402

PIN_FILE = Path("/home/moritz/.coworker/state/rfd4/UPSTREAM.json")
#: the spec loader, on top of the prep package's own roots
EXTRA_ROOTS = (
    "rfproteina.inference_utils.condition_spec",
    "rfproteina.inference_utils.contig_spec",
)
#: `from rfproteina import x` / `from rfproteina.utils import x` import a MODULE OBJECT, which
#: upstream's rewrite (it matches `from rfproteina... import ` and maps the module path) leaves
#: alone, so the vendored file would import the real rfproteina. Same defect as the
#: rfproteina.metrics leak vendor_rfd4_prep.py patches. Mapped to the flattened siblings.
MODULE_IMPORTS = {
    "from rfproteina import conditions as": "from . import conditions as",
    "from rfproteina.utils import standard_annotations as": "from . import utils_standard_annotations as",
    "from rfproteina.utils import chain  ": "from . import utils_chain  ",
}


def load_tool(prep_repo: Path):
    path = prep_repo / "tools" / "vendor_from_main.py"
    spec = importlib.util.spec_from_file_location("rfd4_prep_vendor_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate(tool, rfd4: Path, pin: str) -> dict[str, str]:
    """flat module name -> source, for the prep roots plus the spec loader."""
    tool.ROOTS = tuple(tool.ROOTS) + EXTRA_ROOTS
    sources = tool.closure(rfd4, pin)
    known = set(sources)
    out = {}
    for module, text in sources.items():
        name = tool.flat_name(module)
        if name in tool.SHIM_ONLY:
            continue
        text = tool.rewrite(text, known, pin)
        for old, new in MODULE_IMPORTS.items():
            text = text.replace(old, new)
        out[name] = text
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--prep-repo", required=True)
    ap.add_argument("--rfd4", required=True)
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    a = ap.parse_args()

    pin = json.loads(PIN_FILE.read_text())["sha"]
    tool = load_tool(Path(a.prep_repo))
    generated = generate(tool, Path(a.rfd4), pin)
    dst = Path(a.repo) / "tt_bio" / "_vendor" / "rfd4_prep" / "_cif"

    # The shipped modules must come out of this run unchanged, or the rule that produced the
    # six new ones is not the rule that produced the other 39.
    shipped = Path(a.prep_repo) / "src" / "rfd4_proteina_prep" / "_cif"
    drift = [n for n, text in generated.items()
             if (shipped / f"{n}.py").exists()
             and strip_header(text) != strip_header((shipped / f"{n}.py").read_text())]
    if drift:
        raise SystemExit(f"regenerating the SHIPPED modules at {pin[:12]} changed {drift}; "
                         "the prep pin and the RFD4 pin have diverged, resolve before vendoring")
    new = sorted(n for n in generated if not (shipped / f"{n}.py").exists())
    print(f"{len(generated)} modules at {pin[:12]}, {len(drift)} drifted, new: {new}")

    if a.check:
        missing = [n for n in new if not (dst / f"{n}.py").exists()]
        print(f"missing from the vendored tree: {missing}" if missing else "vendored tree has all of them")
        return 1 if missing else 0

    for n in new:
        text = generated[n]
        # Only the modules written here are checked: the shipped ones carry their own
        # rfproteina leak (metrics_sse_metrics), which vendor_rfd4_prep.py stubs out.
        leaked = [l for l in text.splitlines() if re.match(r"\s*(from|import) rfproteina\b", l)]
        if leaked:
            raise SystemExit(f"{n}: unflattened rfproteina import: {leaked}")
        text = vr.rewrite_imports(text, pkgs=("atomworks",))
        text = text.replace("from enum import StrEnum\n",
                            f"from {vr.VENDOR_NS}.atomworks._compat import StrEnum\n")
        (dst / f"{n}.py").write_text(text)
    print(f"wrote {len(new)} module(s) to {dst}")
    return vr.check(dst.parent, pkgs=("rfd4_proteina_prep", "atomworks", "rfproteina"))


def strip_header(text: str) -> str:
    return re.sub(r"^# Adapted from [^\n]*\n", "", text, flags=re.M)


if __name__ == "__main__":
    sys.exit(main())
