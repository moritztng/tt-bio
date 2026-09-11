"""Price the atom-axis bucket from the measured per-call byte row. CPU only.

Inputs, both measured elsewhere and not re-derived here:
  * ``atom_layer_bytes_p300c.json``, the one row of ``perf/bioir_roofline/
    fold_bytes_512_p300c_qb2.json`` (origin/wk/bioir-roofline) this bet is priced off:
    517.06 MB of DRAM and 1.7698 ms median per call, 48 calls in an 8-step capture fold
    (=> 6/step => 1200 at the production 200 steps). qb2 card 1, p300c, ttnn 0.68.0.
  * ``atom_census.json``: 4116 real heavy atoms in cdk2x2_512, featuriser emits 4128.
  * the atom layer's own weights, 328 704 params = 0.657 MB bf16, counted from boltz2_conf.ckpt.

The model is bytes(NW) = weights + NW * per_window, fitted on the one measured point. It is a
model, not a measurement: what a card has to check is whether the per-call TIME follows it.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = json.load(open(HERE / "atom_layer_bytes_p300c.json"))
CENSUS = {r["fixture"]: r for r in json.load(open(HERE / "atom_census.json"))}

ATOM_WINDOW, CORES = 32, 110
FOLD_S, DEVICE_S = 23.504, 23.12       # published cell, 2026-08-26
INSTRUMENTED_S = SRC["fold_s_instrumented"]          # 27.776 s, the same fold with per-call syncs
WEIGHTS_MB = 328_704 * 2 / 1e6          # one atom DiffusionTransformerLayer, bf16

atom = SRC["atom_layer_row"]
CALLS = atom["calls_in_capture_fold"] // SRC["capture_steps"] * SRC["prod_steps"]
NW_NOW = 224
per_call_mb = atom["dram_total"] / 1e6
per_window_mb = (per_call_mb - WEIGHTS_MB) / NW_NOW
phase_s_instr = CALLS * atom["median_ms"] / 1e3
# The per-call median carries a device sync on both sides; the whole instrumented fold runs
# 18.2 % long because of them. Deflating the phase by the same factor is the conservative read.
deflate = FOLD_S / INSTRUMENTED_S
phase_s_lo, phase_s_hi = phase_s_instr * deflate, phase_s_instr


def rounds(nw):
    """Tile-broadcast rounds for a [1, NW, 32, 128] tile tensor over an 11x10 grid."""
    tiles = nw * (ATOM_WINDOW // 32) * (128 // 32)
    return -(-tiles // CORES)


def arm(name, n_padded):
    nw = n_padded // ATOM_WINDOW
    call_mb = WEIGHTS_MB + nw * per_window_mb
    byte_ratio = call_mb / per_call_mb
    round_ratio = rounds(nw) / rounds(NW_NOW)
    saved = []
    for ratio in (byte_ratio, round_ratio):
        saved += [phase_s_lo * (1 - ratio), phase_s_hi * (1 - ratio)]
    return {
        "arm": name, "n_padded": n_padded, "windows": nw,
        "per_call_MB": round(call_mb, 2),
        "phase_TB": round(CALLS * call_mb / 1e6, 4),
        "phase_TB_deleted": round(CALLS * (per_call_mb - call_mb) / 1e6, 4),
        "byte_ratio": round(byte_ratio, 4),
        "rounds": rounds(nw), "round_ratio": round(round_ratio, 4),
        "fold_s_saved_lo": round(min(saved), 3), "fold_s_saved_hi": round(max(saved), 3),
        "speedup_lo": round(FOLD_S / (FOLD_S - min(saved)), 4),
        "speedup_hi": round(FOLD_S / (FOLD_S - max(saved)), 4),
    }


c512 = CENSUS["cdk2x2_512"]
out = {
    "basis": {
        "fixture": "cdk2x2_512", "n_atom_real": c512["n_atom_real"],
        "n_atom_featurised": c512["n_atom_featurised"],
        "n_padded_today": c512["n_atom_device"], "windows_today": NW_NOW,
        "calls_per_fold": CALLS, "per_call_MB": round(per_call_mb, 2),
        "weights_MB": round(WEIGHTS_MB, 3),
        "weights_share_of_call": round(WEIGHTS_MB / per_call_mb, 6),
        "per_window_MB": round(per_window_mb, 4),
        "phase_TB": round(CALLS * per_call_mb / 1e6, 4),
        "phase_s_instrumented": round(phase_s_hi, 3),
        "phase_s_deflated": round(phase_s_lo, 3),
        "achieved_GB_s": round(per_call_mb / atom["median_ms"], 1),
        "rounds_today": rounds(NW_NOW),
    },
    "arms": [
        arm("ATOM_BUCKET=448 (14*32, the shape set is unchanged)", 4480),
        arm("tightest legal multiple of 32", c512["n_atom_featurised"]),
        arm("ATOM_BUCKET=1024, must also be a multiple of 32", 5120),
    ],
}
print(json.dumps(out, indent=2))
(HERE / "predict.json").write_text(json.dumps(out, indent=2) + "\n")
