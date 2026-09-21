#!/usr/bin/env python3
"""Run upstream OpenFold3 0.5.0 `generate_subset_cache.py` against upstream`s own
published training cache, which it cannot currently parse.

`scripts/datasets/pdb_subset_helpers.py` streams the cache with ijson. The cache
published at s3://openfold3-data/pdb_training_set/dataset_caches/ contains bare
`NaN` literals (`"resolution": NaN`), which is not JSON, and no ijson backend
accepts it -- yajl2_c, yajl2_cffi, yajl2 and the pure-python backend all reject
it. Python`s own `json` module does accept `NaN`, so this swaps the parser and
leaves every line of upstream`s selection logic alone: the ID enumeration order,
`random.Random(seed).sample`, the nested-prefix subsets and `write_subset` are
upstream`s, unmodified. The subset this produces is therefore the subset
upstream`s script would produce if its parser could read its own input.

Cost of the swap: the cache is read whole (~1.7 GB of JSON) instead of streamed.

Usage: python subset_nan_shim.py <scripts/datasets dir> --target-dir <dir> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_CACHE: dict[Path, dict] = {}


def _load(path) -> dict:
    p = Path(path)
    if p not in _CACHE:
        with open(p) as f:
            _CACHE[p] = json.load(f)   # stdlib json parses bare NaN; ijson does not
    return _CACHE[p]


def _parse(f):
    """Minimal `ijson.parse` stand-in: only the events the helpers actually read.

    `enumerate_structure_ids` keeps `(prefix == "structure_data", event ==
    "map_key")` and nothing else, so emitting just those is sufficient and the
    ordering is the file`s, which is what the sampling seed rides on.
    """
    for key in _load(f.name)["structure_data"]:
        yield "structure_data", "map_key", key


def _kvitems(f, prefix):
    doc = _load(f.name)
    if prefix == "":
        for k, v in doc.items():
            if k != "structure_data":
                yield k, v
    elif prefix == "structure_data":
        yield from doc["structure_data"].items()
    else:
        raise AssertionError(f"unexpected ijson prefix {prefix!r}")


def main() -> int:
    scripts_dir = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(scripts_dir))
    import pdb_subset_helpers as H

    H.ijson.parse = _parse          # noqa: SLF001 -- deliberate, documented above
    H.ijson.kvitems = _kvitems

    sys.argv = [str(scripts_dir / "generate_subset_cache.py"), *sys.argv[2:]]
    import runpy
    runpy.run_path(str(scripts_dir / "generate_subset_cache.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
