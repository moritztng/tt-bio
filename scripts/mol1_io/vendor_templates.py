"""Rebuild Mol-1's template pipeline under tt_bio/_vendor/mol1/ from a pinned upstream clone.

    python3 scripts/mol1_io/vendor_templates.py /home/moritz/mol1-upstream/Mol-1

Copies whole, without editing any logic: mol/data/constants.py (as constants_full.py), mol/data/tools/common.py
(tools_common.py), mol/data/tools/kalign.py, and mol/data/template/{template_parser,template_utils,
template_featurizer}.py, plus the template and RNA MSA searches (mol/data/tools/search.py,
mol/data/sources.py, mol/runner/{template_search,rna_msa_search}.py). Only `from mol...` import lines are rewritten. Three small helpers upstream keeps in
modules with heavy dependencies go to helpers.py: pad_to (verbatim), a JSON-only load_json_cached (upstream's JSON
branch: load once per process and cache), and get_logger = logging.getLogger.
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
# Regenerate with scripts/mol1_io/vendor_templates.py; do not edit by hand.
'''
MODS = {
    "mol.data.constants": ".constants_full",
    "mol.data.msa.msa_utils": ".msa_utils",
    "mol.data.template.template_parser": ".template_parser",
    "mol.data.template.template_utils": ".template_utils",
    "mol.data.tools.kalign": ".kalign",
    "mol.data.tools.common": ".tools_common",
    "mol.data.tools.search": ".search",
    "mol.data.sources": ".sources",
    "mol.data.utils": ".helpers",
    "mol.utils.file_io": ".helpers",
    "mol.utils.logger": ".helpers",
}


def rewrite(text):
    def sub(m):
        mod = m.group(2)
        if mod not in MODS:
            raise SystemExit(f"unmapped import {mod}")
        return f"{m.group(1)}from {MODS[mod]} import"
    out = re.sub(r"^(\s*)from (mol(?:\.\w+)+) import", sub, text, flags=re.M)
    assert not re.search(r"^\s*(from|import) mol\b", out, flags=re.M)
    return HEAD + out


def block(text, start, end=None):
    lines = text.splitlines(keepends=True)
    i = next(k for k, l in enumerate(lines) if re.match(start, l))
    j = next((k for k in range(i + 1, len(lines)) if end and re.match(end, lines[k])), len(lines))
    return "".join(lines[i:j]).rstrip() + "\n\n"


files = {"constants_full.py": "mol/data/constants.py", "tools_common.py": "mol/data/tools/common.py",
         "kalign.py": "mol/data/tools/kalign.py", "template_parser.py": "mol/data/template/template_parser.py",
         "template_utils.py": "mol/data/template/template_utils.py",
         "template_featurizer.py": "mol/data/template/template_featurizer.py",
         "search.py": "mol/data/tools/search.py", "sources.py": "mol/data/sources.py",
         "template_search.py": "mol/runner/template_search.py", "rna_msa_search.py": "mol/runner/rna_msa_search.py"}
for dst, src in files.items():
    (OUT / dst).write_text(rewrite((SRC / src).read_text()))

utils = (SRC / "mol/data/utils.py").read_text()
h = HEAD + ('import json\nimport logging\nfrom pathlib import Path\nfrom typing import Any, Union\n\n'
            'import numpy as np\n\nget_logger = logging.getLogger\n_JSON_FILE_CACHE = {}\n\n\n')
h += block(utils, r"^def pad_to", r"^def |^class |^@")
h += ('def load_json_cached(path: Union[str, Path]) -> Any:\n'
      '    """Upstream mol.utils.file_io.load_json_cached, JSON branch only: load once per process."""\n'
      '    path_str = str(Path(path))\n'
      '    if path_str == ".":\n'
      '        return {}\n'
      '    if not path_str.endswith(".json"):\n'
      '        raise ValueError(f"Unsupported file format: {path_str}")\n'
      '    if path_str not in _JSON_FILE_CACHE:\n'
      '        with open(path_str, "r") as f:\n'
      '            _JSON_FILE_CACHE[path_str] = json.load(f)\n'
      '    return _JSON_FILE_CACHE[path_str]\n')
(OUT / "helpers.py").write_text(h)
print("vendored templates", sha, "->", OUT)
