"""Rebuild Mol-1's structure -> input JSON converter (`mol json`) under tt_bio/_vendor/mol1/ from a pinned upstream clone.

    python3 scripts/mol1_io/vendor_json_maker.py /home/moritz/mol1-upstream/Mol-1

Copies json_maker.py and every module it imports whole, without editing any logic; only `from mol...` import lines
are rewritten to the flat vendored names below. constants_full.py is shared with vendor_templates.py.
"""
import re
import subprocess
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[2] / "tt_bio" / "_vendor" / "mol1"
sha = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
HEAD = f'''# Vendored from Mol-1 (https://github.com/molence/Mol-1) @ {sha}, Apache-2.0.
# Copyright 2024 ByteDance and/or its affiliates; modifications Copyright 2026 Molence Bio.
# Regenerate with scripts/mol1_io/vendor_json_maker.py; do not edit by hand.
'''
FILES = {  # upstream module -> vendored file stem
    "mol.data.inference.json_maker": "json_maker",
    "mol.data.core.parser": "parser",
    "mol.data.core.filter": "filter",
    "mol.data.core.ccd": "ccd",
    "mol.data.core.substructure_perms": "substructure_perms",
    "mol.data.utils": "data_utils",
    "mol.data.tools.rewrite_biotite": "rewrite_biotite",
    "mol.data.tools.logger": "tools_logger",
    "mol.utils.file_io": "file_io",
    "mol.utils.logger": "logger",
    "mol.utils.torch_utils": "torch_utils",
    "mol.config.data": "config_data",
    "mol.config.types": "config_types",
}
MODS = {**{k: f".{v}" for k, v in FILES.items()}, "mol.data.constants": ".constants_full"}


def rewrite(text):
    def sub_from(m):
        mod, names = m.group(2), m.group(3)
        if mod in MODS:
            return f"{m.group(1)}from {MODS[mod]} import{names}"
        sub = [f"{mod}.{n.strip()}" for n in names.strip().strip("()").split(",") if n.strip()]
        if sub and all(s in MODS for s in sub):   # `from mol.data.core import ccd`
            return f"{m.group(1)}from . import " + ", ".join(MODS[s][1:] + (f" as {s.rsplit('.', 1)[1]}" if MODS[s][1:] != s.rsplit('.', 1)[1] else "") for s in sub)
        raise SystemExit(f"unmapped import {mod} ({names.strip()})")
    out = re.sub(r"^(\s*)from (mol(?:\.\w+)*) import([^\n]*)", sub_from, text, flags=re.M)
    left = re.findall(r"^\s*(?:from|import) mol\b[^\n]*", out, flags=re.M)
    assert not left, left
    return HEAD + out


for mod, stem in FILES.items():
    (OUT / f"{stem}.py").write_text(rewrite((SRC / (mod.replace(".", "/") + ".py")).read_text()))
print("vendored json_maker", sha, "->", OUT)
