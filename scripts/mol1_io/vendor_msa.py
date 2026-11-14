"""Rebuild tt_bio/_vendor/mol1/ (Mol-1's numpy MSA assembly line) from a pinned upstream clone.

    python3 scripts/mol1_io/vendor_msa.py /home/moritz/mol1-upstream/Mol-1

Copies, without editing any logic: the residue/MSA id tables and chain-type names from mol/data/constants.py,
mol/data/tools/common.py::parse_fasta, all of mol/data/msa/msa_utils.py, and FeatureAssemblyLine + MSAFeat +
ensure_ends_with_newline from mol/data/msa/msa_featurizer.py. Only the import lines are rewritten.
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
# Regenerate with scripts/mol1_io/vendor_msa.py; do not edit by hand.
'''


def block(text, start, end=None):
    """Source from the line matching `start` up to (not including) the line matching `end`."""
    lines = text.splitlines(keepends=True)
    i = next(k for k, l in enumerate(lines) if re.match(start, l))
    j = next((k for k in range(i + 1, len(lines)) if end and re.match(end, lines[k])), len(lines))
    return "".join(lines[i:j]).rstrip() + "\n\n"


const = (SRC / "mol/data/constants.py").read_text()
c = HEAD + "from typing import Final, Set\n\n"
c += block(const, r"^PRO_STD_RESIDUES = ", r"^STD_RESIDUES_WITH_GAP_ID_TO_NAME")
c += block(const, r"^MSA_PROTEIN_SEQ_TO_ID = ", r"^TEMPLATE_PROTEIN_SEQ_TO_ID")
c += block(const, r"^PROTEIN_CHAIN: ", r"^ELEMS = |^def ")
(OUT / "constants.py").write_text(c)

common = (SRC / "mol/data/tools/common.py").read_text()
utils = (SRC / "mol/data/msa/msa_utils.py").read_text()
body = utils[utils.index("logger = get_logger(__name__)"):]
body = body.replace("logger = get_logger(__name__)", "logger = logging.getLogger(__name__)", 1)
u = HEAD + ("import logging\nimport re\nimport string\nfrom collections.abc import Mapping, MutableMapping, Sequence\n"
            "from typing import Any, List, Optional, Set, Tuple\n\nimport numpy as np\n\n"
            "from .constants import (DNA_CHAIN, MSA_DNA_SEQ_TO_ID, MSA_PROTEIN_SEQ_TO_ID, MSA_RNA_SEQ_TO_ID,\n"
            "                        PROTEIN_CHAIN, RNA_CHAIN, STANDARD_POLYMER_CHAIN_TYPES, STD_RESIDUES_WITH_GAP)\n\n\n")
u += block(common, r"^def parse_fasta", r"^def |^class ") + "\n" + body
(OUT / "msa_utils.py").write_text(u)

feat = (SRC / "mol/data/msa/msa_featurizer.py").read_text()
f = HEAD + ("import dataclasses\nfrom typing import Any, Dict, Mapping, Optional\n\nimport numpy as np\n\n"
            "from .constants import PROTEIN_CHAIN, RNA_CHAIN, STANDARD_POLYMER_CHAIN_TYPES, STD_RESIDUES_WITH_GAP\n"
            "from .msa_utils import MSA_GAP_IDX, MSAPairingEngine, NUM_SEQ_NUM_RES_MSA_FEATURES, RawMsa, map_to_standard\n\n\n")
f += block(feat, r"^class FeatureAssemblyLine", r"^class MSAFeaturizer")
f += block(feat, r"^@dataclasses.dataclass", r"^class InferenceMSAFeaturizer")
(OUT / "msa_assembly.py").write_text(f)
(OUT / "__init__.py").write_text(HEAD + '"""Mol-1 / Protenix numpy MSA assembly line, verbatim."""\n')
print("vendored", sha, "->", OUT)
