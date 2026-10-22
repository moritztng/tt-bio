"""Export the AE decoder's vocabulary tables from upstream (pinned sha) to tt_bio/rfd4/ae_tables.json."""
import json
import sys
from pathlib import Path

from atomworks.ml.conditions import C_SEQ
from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
from rfproteina.partial_autoencoder.decoder import get_allowed_residue_type_mask, get_atom_mask

out = Path(sys.argv[1])
d = {
    "upstream_sha": "2db07bb444d91695f8c10e6acb7648e9297f9221",
    "c_seq_key": C_SEQ.full_name,
    "mask_token": UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"],
    "n_tokens": UNIFIED_ATOM37_ENCODING.n_tokens,
    "allowed": get_allowed_residue_type_mask().int().tolist(),
    "atom_mask": get_atom_mask().int().tolist(),
}
out.write_text(json.dumps(d, separators=(",", ":")) + "\n")
print(d["c_seq_key"], d["mask_token"], d["n_tokens"], len(d["allowed"]), len(d["atom_mask"]))
