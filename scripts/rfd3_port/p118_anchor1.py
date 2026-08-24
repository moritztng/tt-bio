"""p118 — score one arm against Anchor 1: release_gate.py's RFD3 leg, driven directly.

Why not `release_gate.py --model rfd3`. Its preflight refuses this box's interpreter, because
tt-bio declares transformers>=5.5.0 / huggingface_hub>=1.5.0 and qb2's venvs carry 4.57.6 / 0.36.2
(`release-gate-interpreter-must-match-pinned-runtime`). That guard is a whole-release guard and its
stated reason is that those versions change results. It cannot change THIS leg's result: `grep -rn
"transformers|huggingface_hub" tt_bio/rfd3/ tt_bio/rfd3_bias.py tt_bio/_vendor/rf3/` returns
nothing, so RFD3's design path never imports either. So this driver imports release_gate and calls
`run_rfd3` unchanged -- same fold command, same scoring, same committed thresholds, no new bar --
and skips only the interpreter preflight. Both arms run through this identical driver.

Usage: p118_anchor1.py <out.json>     (set RFD3_BLOCK_SPARSE=1 for the on arm)
"""
import importlib.util
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(sys.argv[1])

spec = importlib.util.spec_from_file_location("rg", REPO / "scripts" / "release_gate.py")
rg = importlib.util.module_from_spec(spec)
sys.modules["rg"] = rg
spec.loader.exec_module(rg)

# Same P300 mesh-graph setup main() does, so the fold subprocesses inherit it.
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor  # noqa: E402
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

from tt_bio.rfd3 import block_sparse as BS  # noqa: E402

row = rg.run_rfd3(keep=False)
result = {
    "arm": "on" if BS.enabled() else "off",
    "env_RFD3_BLOCK_SPARSE": os.environ.get("RFD3_BLOCK_SPARSE"),
    "bs_config": {"q_block": BS.config()[0], "buckets": list(BS.config()[1])},
    "row": row,
    "thresholds": {
        "RFD3_MIN_CLEAN_RATE": rg.RFD3_MIN_CLEAN_RATE,
        "RFD3_MIN_INBAND": rg.RFD3_MIN_INBAND,
        "RFD3_MAX_BREAKS": rg.RFD3_MAX_BREAKS,
        "RFD3_MAX_CLASHES": rg.RFD3_MAX_CLASHES,
        "RFD3_MIN_DISTINCT_AA": rg.RFD3_MIN_DISTINCT_AA,
        "RFD3_MAX_UNK": rg.RFD3_MAX_UNK,
        "num_designs": rg.RFD3_NUM_DESIGNS,
        "timesteps": rg.RFD3_TIMESTEPS,
        "seed": rg.RFD3_SEED,
        "det_timesteps": rg.RFD3_DET_TIMESTEPS,
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, default=str) + "\n")
print(json.dumps(result, indent=2, default=str), flush=True)
print("[p118] gate=%s arm=%s" % (row.get("gate"), result["arm"]), flush=True)
