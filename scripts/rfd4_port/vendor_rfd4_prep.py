"""Vendor RFD4's host package, ``rfd4_proteina_prep``, under tt_bio/_vendor/rfd4_prep.

``rfd4-proteina-prep`` (https://huggingface.co/nvidia/RFD4-Proteina-prep, BSD-3-Clause) is
upstream's own extraction of everything RFD4 does on the host: annotated-CIF parsing, the
condition spec, atom37 encoding, featurization and the annotated-CIF writer. Its public
surface is ``prepare_cif`` and ``write_cif``. Vendoring it means tt-bio featurizes with the
code NVIDIA's own pipeline runs, instead of a transcription of it.

It sits on atomworks 3.0.4, vendored by ``scripts/rf3_port/vendor_rf3.py``. This script reuses
that module's import rewrite, shims and leak check, so both trees are made the same way.

    python scripts/rfd4_port/vendor_rfd4_prep.py --repo . --src <clone>/src/rfd4_proteina_prep
    python scripts/rfd4_port/vendor_rfd4_prep.py --repo . --check

Pinned: c99a272df90566bbcabf4af0e06bb26e58d704b9 (2026-10-20). Bump PIN with the source.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rf3_port"))
import vendor_rf3 as vr  # noqa: E402

PIN = "c99a272df90566bbcabf4af0e06bb26e58d704b9"
NAME = "rfd4_prep"
UPSTREAM = "rfd4_proteina_prep"
#: what a vendored rfd4_prep file may import by bare name: itself and atomworks
PKGS = (UPSTREAM, "atomworks")

#: The one upstream line that cannot stay. ``metrics_sse_metrics`` subclasses rfproteina's
#: ``Metric`` for two evaluation metrics, so importing the prep package needs the whole of
#: rfproteina installed, although prep itself only uses the module's
#: ``annotate_secondary_structure``. The two metric classes are kept, on a base that says
#: where they come from if anything tries to build one.
METRIC_STUB = '''

class Metric:  # tt-bio: rfproteina.metrics.Metric is not vendored
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} is an rfproteina evaluation metric; "
                                  "tt-bio vendors only the RFD4 prep path")
'''
FIXES = (
    ("_cif/metrics_sse_metrics.py", "from rfproteina.metrics import Metric\n", METRIC_STUB.lstrip("\n")),
    # importlib.resources looks the vocabulary up by package name, which the import rewrite
    # does not see because it is a string.
    ("_cif/encoding.py", 'files("rfd4_proteina_prep")', f'files("{vr.VENDOR_NS}.{NAME}")'),
)

#: Training-time design-task transforms. Nothing on the prep path imports them, and the second
#: needs ``assertpy`` at import, which tt-bio does not depend on, so they are not vendored.
EXCLUDE = ("transforms_design_task.py", "transforms_utils_ppi_utils.py")


def rewrite(text: str) -> str:
    text = vr.rewrite_imports(text, pkgs=("atomworks",))
    # rewrite_imports maps `pkg` to `tt_bio._vendor.pkg`; this package is renamed as well
    for kw in ("from", "import"):
        text = text.replace(f"{kw} {UPSTREAM}", f"{kw} {vr.VENDOR_NS}.{NAME}")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--src", help="src/rfd4_proteina_prep of a clone at PIN")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    dst = Path(a.repo) / "tt_bio" / "_vendor" / NAME
    if a.check:
        return vr.check(dst, pkgs=PKGS + ("rfproteina",))
    if not a.src:
        ap.error("--src is required unless --check")

    src = Path(a.src)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", *EXCLUDE))
    for lic in ("LICENSE", "THIRD_PARTY_LICENSES.txt"):
        shutil.copy2(src.parents[1] / lic, dst / lic)

    n = 0
    for path in dst.rglob("*.py"):
        text = path.read_text()
        new = rewrite(text)
        if new != text:
            path.write_text(new)
            n += 1
    print(f"rewrote imports in {n} files")

    for rel, old, new in FIXES:
        path = dst / rel
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"FIXES: {rel} has {text.count(old)} copies of {old!r}, expected 1")
        path.write_text(text.replace(old, new))
    print(f"fixed {len(FIXES)} upstream lines")

    # Python 3.10 has no enum.StrEnum; atomworks' vendored tree already carries the shim
    for path in dst.rglob("*.py"):
        text = path.read_text()
        new = text.replace("from enum import StrEnum\n",
                           f"from {vr.VENDOR_NS}.atomworks._compat import StrEnum\n")
        if new != text:
            path.write_text(new)
    return vr.check(dst, pkgs=PKGS + ("rfproteina",))


if __name__ == "__main__":
    sys.exit(main())
