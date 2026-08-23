"""Dump the resolved AF2 config ColabDesign actually runs, as a committed JSON spec.

The config is `alphafold/model/config.py`'s `model_1_ptm` entry after ColabDesign's own
mutations (`colabdesign/af/model.py:95-130`), which is not the same thing as either file alone.
Runs in the external JAX env; nothing in tt_bio imports it.

    ~/pxd_af2_cpu/bin/python scripts/af2_port/dump_model_config.py \
        --out scripts/af2_port/af2ig_model_config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def to_plain(value):
    if hasattr(value, "items"):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from colabdesign.af.alphafold.model import config as af_config

    out = {}
    for name in ("model_1_ptm", "model_3_ptm"):
        cfg = af_config.model_config(name)
        out[name] = to_plain(cfg)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
