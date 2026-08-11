"""The capacity leg's binding case (9j4c_abag, 1095 tokens, protenix-v2, 50 samples), one arm.

_pair_transpose holds one extra pair tensor while its ROW_MAJOR round trip is in flight. At 1095
tokens that is 1095^2 x 256 x 2 = 0.5717 GiB, against a committed peak of 8.72 GiB and a 10.5 GiB
budget -- so the arithmetic says it fits with 1.21 GiB to spare. This measures it instead.
"""
import json, os, sys
from pathlib import Path
REPO = Path("/home/ttuser/.coworker/wt/pairformer-resident-chunking")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import release_gate as rg

leg = rg.CAPACITY_LEGS[0]
print("leg:", leg, flush=True)
row = rg.run_capacity(keep=False, leg=leg)
row["pair_transpose_rm"] = os.environ.get("TT_BIO_PAIR_TRANSPOSE_RM", "<default 1>")
row["l1_headroom"] = os.environ.get("TT_BIO_TRANSPOSE_L1_HEADROOM", "<default 2.5>")
out = sys.argv[1]
Path(out).write_text(json.dumps(row, indent=1))
print(json.dumps(row, indent=1), flush=True)
