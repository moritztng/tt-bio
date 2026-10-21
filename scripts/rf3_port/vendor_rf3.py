#!/usr/bin/env python3
"""Vendor the RF3 host-featurization stack into ``tt_bio/_vendor/``.

Three upstream pieces, all BSD:

- ``atomworks`` (PyPI 3.0.4; 2.2.1 until RFD4) -- the AF3 transform pipeline, shared
  by RF3 and RFD4. Vendored rather than pip-depended on because it declares
  ``requires-python >=3.11`` and pins ``biotite==1.6.0``, while tt-bio also runs 3.10,
  where the newest biotite is 1.2.0. With the shims below and the biotite 1.2 backport
  in ``atomworks_local.patch`` it runs on 3.10, and RF3's featurizer stays bit-exact
  on its ten committed fixtures (``tests/test_rf3_featurizer.py``). RFD4's prep
  package is vendored on top of it by ``scripts/rfd4_port/vendor_rfd4_prep.py``,
  which reuses this module.
- ``rf3`` (RosettaCommons/foundry, models/rf3) -- the RF3-specific host transforms
  and input-spec plumbing, plus the torch reference model the ttnn port is scored
  against. Inference only: no losses, no trainers, no Lightning.
- three helpers out of ``foundry.utils``. ``foundry/__init__.py`` itself is an
  env/typecheck/cuEquivariance bootstrap pulling environs, beartype.claw and
  jaxtyping import hooks, so it is replaced by a minimal stub rather than vendored.

Import rewriting is textual and checked afterwards: no bare ``atomworks``/``rf3``/
``foundry`` reference may survive.

Re-run with --check to verify an existing vendor tree without rewriting. Without
--foundry-src only atomworks is re-vendored and rf3/foundry are left as they are.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

VENDOR_NS = "tt_bio._vendor"
PKGS = ("atomworks", "rf3", "foundry")

RF3_KEEP = (
    # host featurization
    "data/__init__.py",
    "data/cyclic_transform.py",
    "data/extra_xforms.py",
    "data/ground_truth_template.py",
    "data/pipeline_utils.py",
    "data/pipelines.py",
    "utils/__init__.py",
    "utils/inference.py",
    "utils/io.py",
    # torch reference model (inference only -- the ttnn port is scored against this)
    "model/__init__.py",
    "model/RF3.py",
    "model/RF3_blocks.py",
    "model/RF3_structure.py",
    "model/layers/__init__.py",
    "model/layers/af3_auxiliary_heads.py",
    "model/layers/af3_diffusion_transformer.py",
    "model/layers/attention.py",
    "model/layers/layer_utils.py",
    "model/layers/mlff.py",
    "model/layers/outer_product.py",
    "model/layers/pairformer_layers.py",
    "model/layers/structure_bias.py",
    "diffusion_samplers/__init__.py",
    "diffusion_samplers/inference_sampler.py",
    "util_module.py",
    # `loss/loss.py` is named for training but `calc_chiral_grads_flat_impl` is on
    # the inference path: the diffusion transformer's chiral conditioning calls it.
    "loss/__init__.py",
    "loss/loss.py",
)
FOUNDRY_KEEP = (
    "common.py",
    "utils/alignment.py",
    "utils/torch.py",
    "model/__init__.py",
    "model/layers/__init__.py",
    "model/layers/blocks.py",
    "training/__init__.py",
    "training/checkpoint.py",
    "utils/rigid.py",
    "utils/rotation_augmentation.py",
)

STRENUM_SHIM = '''"""``enum.StrEnum`` backport.

Added in Python 3.11; tt-bio supports 3.10, which is the deployed runtime. The
upstream atomworks package uses it for three enums.
"""

from enum import Enum


class StrEnum(str, Enum):
    """Minimal stand-in for :class:`enum.StrEnum`."""

    def __str__(self) -> str:
        return str(self.value)


__all__ = ["StrEnum"]
'''

FOUNDRY_INIT = '''"""Minimal stand-in for ``foundry/__init__.py``.

Upstream this module reads a ``.env`` file, optionally installs beartype and
jaxtyping import hooks, and probes for cuEquivariance on CUDA. None of that
applies to tt-bio: there is no CUDA path, and the import hooks would pull two
dependencies in for nothing. Only the two module-level flags are actually read by
the vendored code, so those are all that is kept.
"""

import logging

logger = logging.getLogger("tt_bio._vendor.foundry")

SHOULD_USE_CUEQUIVARIANCE = False
DISABLE_CHECKPOINTING = False

# Upstream reads these from a .env file. NAN_CHECK defaults to True there, and the
# vendored foundry.utils.torch swaps assert_no_nans for a no-op when it is False.
should_check_nans = True
should_debug = False
should_typecheck = False

__all__ = [
    "SHOULD_USE_CUEQUIVARIANCE",
    "DISABLE_CHECKPOINTING",
    "logger",
    "should_check_nans",
    "should_debug",
    "should_typecheck",
]
'''

FOUNDRY_DDP = '''"""Minimal stand-in for ``foundry.utils.ddp``.

Only :class:`RankedLogger` is reachable from the featurization path, and upstream
pulls lightning, lightning_utilities and omegaconf in for it plus a set of
training-time accelerator helpers. tt-bio runs the host featurizer in a single
process, so rank is always zero and the adapter is a passthrough.
"""

import logging
from typing import Any


def get_current_rank() -> int:
    """Rank of the current process. The vendored featurizer is single-process."""
    return 0


def is_rank_zero() -> bool:
    return True


class RankedLogger(logging.LoggerAdapter):
    """Passthrough logger adapter matching the upstream constructor signature."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(logging.getLogger(name), extra or {})
        self.rank_zero_only = rank_zero_only

    def process(self, msg, kwargs):
        return msg, kwargs


__all__ = ["RankedLogger", "get_current_rank", "is_rank_zero"]
'''


def rewrite_imports(text: str, pkgs=PKGS) -> str:
    """Point absolute imports of the vendored packages at the vendor namespace."""
    for pkg in pkgs:
        # from <pkg>[.sub] import ...
        text = re.sub(
            rf"^(\s*)from {pkg}(\.|\s)",
            rf"\1from {VENDOR_NS}.{pkg}\2",
            text,
            flags=re.MULTILINE,
        )
        # import <pkg>[.sub] [as ...]
        text = re.sub(
            rf"^(\s*)import {pkg}(\.[\w.]+)?(\s+as\s+\w+)?(\s*#.*)?$",
            lambda m, p=pkg: (
                f"{m.group(1)}import {VENDOR_NS}.{p}{m.group(2) or ''}"
                + (m.group(3) or f" as {p}")
                + (m.group(4) or "")
            ),
            text,
            flags=re.MULTILINE,
        )
    return text


#: Python 3.12-only syntax, fixed by exact text substitution with identical behaviour.
#: (path under the vendor root, upstream text, 3.10-valid text). A substitution whose
#: upstream text is missing fails the vendor run, so an upstream edit cannot leave the
#: tree silently unpatched.
PY310_FIXES = (
    # PEP 701: a backslash inside an f-string expression. U+2500 written literally.
    ("atomworks/ml/datasets/metadata.py",
     "{'\\u2500' * inner_width}", "{'\u2500' * inner_width}"),
)


def apply_shims(root: Path, pkgs=("atomworks",)) -> list[str]:
    """Make the vendored packages run on Python 3.10.

    The two 3.11-only imports are plain ``from <mod> import a, b, c`` lines, so drop
    the 3.11-only name from the list and add a line importing it from the shim. The
    3.12-only syntax in PY310_FIXES is substituted verbatim.
    """
    touched = []
    (root / "atomworks" / "_compat.py").write_text(STRENUM_SHIM)
    for rel, old, new in PY310_FIXES:
        path = root / rel
        if not path.exists() or rel.split("/")[0] not in pkgs:
            continue
        text = path.read_text()
        if old not in text:
            raise SystemExit(f"PY310_FIXES: {rel} no longer contains {old!r}")
        path.write_text(text.replace(old, new))
        touched.append(rel)
    swaps = (
        ("enum", "StrEnum", f"from {VENDOR_NS}.atomworks._compat import StrEnum"),
        ("typing", "Never", "from typing_extensions import Never"),
    )
    for path in (p for pkg in pkgs for p in (root / pkg).rglob("*.py")):
        original = path.read_text()
        out = []
        for line in original.splitlines():
            for module, name, replacement in swaps:
                m = re.match(rf"^(\s*)from {module} import (.+)$", line)
                if not m:
                    continue
                names = [n.strip() for n in m.group(2).split(",")]
                if name not in names:
                    continue
                indent = m.group(1)
                names.remove(name)
                if names:
                    out.append(f"{indent}from {module} import " + ", ".join(names))
                out.append(indent + replacement)
                break
            else:
                out.append(line)
        text = "\n".join(out) + ("\n" if original.endswith("\n") else "")
        if text != original:
            path.write_text(text)
            touched.append(str(path.relative_to(root)))
    return touched


def check(root: Path, pkgs=PKGS) -> int:
    """Fail if any un-rewritten absolute import of a vendored package survives."""
    bad = []
    for path in root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for pkg in pkgs:
                if re.match(rf"\s*(from {pkg}[.\s]|import {pkg}([.\s]|$))", line):
                    bad.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    for entry in bad[:40]:
        print("  LEAK", entry)
    print(f"{'FAIL' if bad else 'OK'}: {len(bad)} un-rewritten imports")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--atomworks-src", help="site-packages/atomworks of a 3.0.4 install")
    ap.add_argument("--foundry-src", help="RosettaCommons/foundry checkout root")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    vendor = Path(args.repo) / "tt_bio" / "_vendor"
    if args.check:
        return check(vendor)

    if not args.atomworks_src:
        ap.error("--atomworks-src is required unless --check")
    aw_src = Path(args.atomworks_src)

    # atomworks: whole package
    dst = vendor / "atomworks"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(aw_src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The licence lives in the wheel's dist-info, not the package directory, so a plain
    # copy of site-packages/atomworks ships the code without it.
    licences = sorted(aw_src.parent.glob("atomworks-*.dist-info/licenses/LICENSE*"))
    if not licences:
        raise SystemExit(f"no atomworks LICENSE next to {aw_src}; refusing to vendor without it")
    shutil.copy2(licences[-1], dst / "LICENSE")
    if args.foundry_src:
        vendor_rf3_and_foundry(vendor, Path(args.foundry_src))

    # rewrite imports everywhere
    n = 0
    for pkg in PKGS:
        for path in (vendor / pkg).rglob("*.py"):
            text = path.read_text()
            new = rewrite_imports(text)
            if new != text:
                path.write_text(new)
                n += 1
    print(f"rewrote imports in {n} files")

    touched = apply_shims(vendor)
    print(f"shimmed {len(touched)} files: {touched}")
    apply_local_patch(vendor / "atomworks")

    return check(vendor)


#: tt-bio's own changes to atomworks, kept as a patch against the shimmed tree so a
#: re-vendor carries them instead of erasing them: reference conformers generated in a
#: small helper process, concurrently per CCD code (``ml/transforms/conformer_helper.py``
#: and the hooks into ``rdkit_utils``, ``af3_reference_molecule`` and ``utils/timer``).
LOCAL_PATCH = Path(__file__).with_name("atomworks_local.patch")


def apply_local_patch(dst: Path) -> None:
    import subprocess

    r = subprocess.run(["patch", "-p0", "--forward", "--no-backup-if-mismatch", "-d", str(dst),
                        "-i", str(LOCAL_PATCH)], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        raise SystemExit(f"{LOCAL_PATCH.name} did not apply cleanly:\n{r.stdout}{r.stderr}")


def vendor_rf3_and_foundry(vendor: Path, fo_src: Path) -> None:
    # rf3: featurization subset only
    dst = vendor / "rf3"
    shutil.rmtree(dst, ignore_errors=True)
    src = fo_src / "models" / "rf3" / "src" / "rf3"
    (dst).mkdir(parents=True)
    shutil.copy2(src / "__init__.py", dst / "__init__.py")
    for rel in RF3_KEEP:
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        source = src / rel
        if source.exists():
            shutil.copy2(source, target)
        else:
            target.write_text('"""Namespace package for the vendored RF3 subset."""\n')

    # foundry: three helpers plus a stub __init__
    dst = vendor / "foundry"
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "utils").mkdir(parents=True)
    (dst / "__init__.py").write_text(FOUNDRY_INIT)
    (dst / "utils" / "__init__.py").write_text(
        '"""Namespace package for the vendored foundry helpers."""\n'
    )
    (dst / "utils" / "ddp.py").write_text(FOUNDRY_DDP)
    for rel in FOUNDRY_KEEP:
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        source = fo_src / "src" / "foundry" / rel
        if source.exists():
            shutil.copy2(source, target)
        elif not target.exists():
            target.write_text(
                '"""Namespace package for the vendored foundry subset."""\n'
            )


if __name__ == "__main__":
    sys.exit(main())
