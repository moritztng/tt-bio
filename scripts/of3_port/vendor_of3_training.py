#!/usr/bin/env python3
"""Vendor OpenFold3's training data pipeline into ``tt_bio/_vendor/openfold3/``.

The original vendoring kept only the query-to-features inference path and dropped
"the Lightning Dataset/DataModule framework, the LMDB training dataset caches, and
the S3 template-cache build pipeline" (NOTICE:48-51). Training needs all three.
This script puts them back from the same upstream release the rest of the tree came
from, so the batch we emit is upstream's batch rather than a re-implementation.

Two kinds of file are written:

- ADD -- modules absent from the vendor tree (the framework itself, the LMDB cache
  format and its builders, cropping, permutation alignment, the S3 helpers).
- RESTORE -- ``pipelines/preprocessing/template.py`` and ``primitives/caches/format.py``,
  which were vendored as stubs (189 of 2708 and 39 of 1021 lines). Both are strict
  subsets of upstream plus a tt-bio docstring note, so restoring them loses nothing.

Every file is byte-identical to upstream apart from the textual import rewrite
``openfold3.`` -> ``tt_bio._vendor.openfold3.``, with one exception: four modules get
their heavy optional imports (``lmdb``, ``boto3``, ``kalign``) deferred, because
``primitives/caches/format.py`` and ``pipelines/preprocessing/template.py`` sit on the
*inference* import path and a plain restore would make every ``tt-bio predict`` user
install training-only dependencies. The deferrals are listed in ADAPTATIONS below,
are pure import plumbing, and touch nothing that reaches a tensor. Upstream already
uses the same idiom (``format.py`` defers ``convert_datacache_to_lmdb``), as does the
existing vendor tree (``logging_utils.py`` defers ``memory_profiler``).

Usage:
    python scripts/of3_port/vendor_of3_training.py --src <unpacked-openfold3-0.4.3>
    python scripts/of3_port/vendor_of3_training.py --src <...> --check
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

VENDOR_NS = "tt_bio._vendor"
UPSTREAM_VERSION = "0.4.3"

# Modules the vendor tree does not have at all. This is the transitive import closure
# of `core.data.framework.*` over upstream, minus what is already vendored.
ADD = (
    "core/data/framework/__init__.py",
    "core/data/framework/data_module.py",
    "core/data/framework/lightning_utils.py",
    "core/data/framework/stochastic_sampler_dataset.py",
    "core/data/framework/single_datasets/__init__.py",
    "core/data/framework/single_datasets/abstract_single.py",
    "core/data/framework/single_datasets/base_of3.py",
    "core/data/framework/single_datasets/dataset_utils.py",
    "core/data/framework/single_datasets/inference.py",
    "core/data/framework/single_datasets/monomer.py",
    "core/data/framework/single_datasets/pdb.py",
    "core/data/framework/single_datasets/validation.py",
    "core/data/io/dataset_cache.py",
    "core/data/io/s3.py",
    "core/data/io/sequence/template.py",
    "core/data/io/structure/pdb.py",
    "core/data/pipelines/featurization/loss_weights.py",
    "core/data/pipelines/sample_processing/structure.py",
    "core/data/primitives/caches/lmdb.py",
    "core/data/primitives/permutation/__init__.py",
    "core/data/primitives/permutation/mol_labels.py",
    "core/data/primitives/sequence/hash.py",
    "core/data/primitives/sequence/template.py",
    "core/data/primitives/structure/cropping.py",
    "core/data/tools/colabfold_msa_server.py",
    "core/data/tools/kalign.py",
    "core/data/tools/rscb.py",
    "core/utils/geometry/kabsch_alignment.py",
    "core/utils/logging_utils.py",
    "core/utils/permutation_alignment.py",
    # The pydantic models a runner yaml's dataset_configs section validates into.
    "projects/of3_all_atom/config/dataset_configs.py",
)

# Vendored as stubs by the inference-only vendoring; restored in full here.
RESTORE = (
    "core/data/pipelines/preprocessing/template.py",
    "core/data/primitives/caches/format.py",
)

# (file, anchor, replacement). Each anchor must appear exactly once or the script
# fails: an upstream bump that moves one of these should be a loud error, not a
# silently skipped patch.
ADAPTATIONS: tuple[tuple[str, str, str], ...] = (
    # -- lmdb ---------------------------------------------------------------
    # `caches/lmdb.py` annotates with `lmdb.Environment` at def time, so it needs
    # `from __future__ import annotations` before the import can move.
    (
        "core/data/primitives/caches/lmdb.py",
        "import json\nimport pickle as pkl\n",
        "from __future__ import annotations\n\nimport json\nimport pickle as pkl\n",
    ),
    (
        "core/data/primitives/caches/lmdb.py",
        "import lmdb\nfrom tqdm import tqdm\n\nif TYPE_CHECKING:\n",
        "from tqdm import tqdm\n\nif TYPE_CHECKING:\n    import lmdb\n",
    ),
    (
        "core/data/primitives/caches/lmdb.py",
        "    def get(self) -> lmdb.Environment:\n        if self._env is None:\n",
        "    def get(self) -> lmdb.Environment:\n"
        "        import lmdb  # training-only dependency; see vendor_of3_training.py\n\n"
        "        if self._env is None:\n",
    ),
    (
        "core/data/primitives/caches/lmdb.py",
        "    from openfold3.core.data.io.dataset_cache import (\n"
        "        convert_dataclass_to_dict,\n"
        "        read_datacache,\n"
        "    )\n",
        "    import lmdb  # training-only dependency; see vendor_of3_training.py\n\n"
        "    from openfold3.core.data.io.dataset_cache import (\n"
        "        convert_dataclass_to_dict,\n"
        "        read_datacache,\n"
        "    )\n",
    ),
    # `caches/format.py` uses lmdb only in annotations and already has
    # `from __future__ import annotations`, so TYPE_CHECKING is enough.
    (
        "core/data/primitives/caches/format.py",
        "import lmdb\n\nfrom openfold3.core.data.primitives.caches.lmdb import",
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    import lmdb\n\n"
        "from openfold3.core.data.primitives.caches.lmdb import",
    ),
    (
        "core/data/io/dataset_cache.py",
        "import lmdb\n\nfrom openfold3.core.data.primitives.caches.format import",
        "from openfold3.core.data.primitives.caches.format import",
    ),
    (
        "core/data/io/dataset_cache.py",
        "        # Assumed to be an lmdb dir\n        with (\n",
        "        # Assumed to be an lmdb dir\n"
        "        import lmdb  # training-only dependency; see vendor_of3_training.py\n\n"
        "        with (\n",
    ),
    # -- vendoring depth ----------------------------------------------------
    # This one is not an optional-dependency deferral. `framework/__init__.py`
    # imports its single_datasets submodules by rebuilding the module name out of
    # FILESYSTEM PATH COMPONENTS (`path.parts[-6:-1]`), which hardcodes upstream's
    # package depth and yields a bare "openfold3.core.data.framework..." under
    # tt_bio/_vendor/. A textual import rewrite cannot see it because there is no
    # import statement to rewrite. Deriving the prefix from __package__ imports
    # exactly the same modules and is correct at either depth.
    (
        "core/data/framework/__init__.py",
        '        __import__(".".join(list(path.parts[-6:-1]) + [path.parts[-1].split(".")[0]]))\n',
        '        __import__(f"{__package__}.{directory.name}.{path.stem}")\n',
    ),
    # -- kalign -------------------------------------------------------------
    # Only reached when the template pipeline has to realign an hmmsearch hit.
    (
        "core/data/tools/kalign.py",
        "from functools import lru_cache\n\nimport kalign\n",
        "from functools import lru_cache\n",
    ),
    (
        "core/data/tools/kalign.py",
        '    """Wrapper around kalign.align with caching."""\n    return kalign.align(list(sequences))\n',
        '    """Wrapper around kalign.align with caching."""\n'
        "    import kalign  # training-only dependency; see vendor_of3_training.py\n\n"
        "    return kalign.align(list(sequences))\n",
    ),
    # -- boto3 --------------------------------------------------------------
    (
        "core/data/io/s3.py",
        "import io\nimport json\n",
        "from __future__ import annotations\n\nimport io\nimport json\n",
    ),
    (
        "core/data/io/s3.py",
        "import boto3\nimport botocore\nimport botocore.paginate\n"
        "from botocore.config import Config\n",
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    import boto3\n"
        "    import botocore\n"
        "    import botocore.paginate\n",
    ),
    (
        "core/data/io/s3.py",
        "    session = boto3.Session(profile_name=profile)\n    return session.client(\n",
        "    import boto3  # training-only dependency; see vendor_of3_training.py\n"
        "    from botocore.config import Config\n\n"
        "    session = boto3.Session(profile_name=profile)\n    return session.client(\n",
    ),
    (
        "core/data/io/s3.py",
        "    # TODO: rework with existing primitives\n    if session is None:\n",
        "    import boto3  # training-only dependency; see vendor_of3_training.py\n\n"
        "    # TODO: rework with existing primitives\n    if session is None:\n",
    ),
)


def rewrite_imports(text: str) -> str:
    """Point every absolute ``openfold3`` import at the vendored copy."""
    return re.sub(r"\bopenfold3\.", f"{VENDOR_NS}.openfold3.", text)


def render(src_root: Path, rel: str) -> str:
    text = (src_root / rel).read_text()
    n_adapted = 0
    for target, anchor, replacement in ADAPTATIONS:
        if target != rel:
            continue
        if text.count(anchor) != 1:
            raise SystemExit(
                f"{rel}: adaptation anchor found {text.count(anchor)} times, expected 1.\n"
                f"  anchor: {anchor!r}\n"
                f"  Upstream moved; re-derive the patch instead of skipping it."
            )
        text = text.replace(anchor, replacement)
        n_adapted += 1
    if n_adapted:
        print(f"    ({n_adapted} import adaptation(s))")
    return rewrite_imports(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help=f"Unpacked openfold3 {UPSTREAM_VERSION} (the dir containing openfold3/).",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tt_bio/_vendor/openfold3",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Verify the vendored tree matches what this script would write.",
    )
    args = ap.parse_args()

    src_root = args.src / "openfold3"
    if not src_root.is_dir():
        raise SystemExit(f"no openfold3/ under {args.src}")

    stale = 0
    for rel in ADD + RESTORE:
        out = args.dest / rel
        if rel.endswith("__init__.py") and not (src_root / rel).exists():
            # Upstream ships namespace dirs without __init__ in a couple of places.
            want = ""
        else:
            want = render(src_root, rel)

        if args.check:
            have = out.read_text() if out.exists() else None
            if have != want:
                print(f"STALE {rel}")
                stale += 1
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(want)
        print(f"{'RESTORE' if rel in RESTORE else 'ADD':>7} {rel}")

    if args.check:
        # No bare upstream import may survive the rewrite.
        for rel in ADD + RESTORE:
            out = args.dest / rel
            if not out.exists():
                continue
            for i, line in enumerate(out.read_text().splitlines(), 1):
                if re.search(r"(?<!tt_bio\._vendor\.)\bopenfold3\.", line) and (
                    line.lstrip().startswith(("import ", "from "))
                ):
                    print(f"BARE-IMPORT {rel}:{i}: {line.strip()}")
                    stale += 1
        print("OK" if not stale else f"{stale} problem(s)")
        return 1 if stale else 0

    print(f"\nvendored {len(ADD)} new + {len(RESTORE)} restored from openfold3 {UPSTREAM_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
