#!/usr/bin/env python3
"""Stamp every artifact of this row that cites the bundle with which reference it was taken
against, now that `of3t-reference` has withdrawn one and published another.

`of3t-orchestrator`'s compose fails if two live artifacts cite different digests for the same
thing, and warns if one artifact silently answers a question another has since re-answered.
Both are avoided the same way: an artifact states the digest it was measured against AND its
standing, so a historical number stays readable as historical rather than reading as current.
Rewriting the digest inside an old artifact would be the other option and it is the wrong one --
the number really was taken against the withdrawn tape, and saying otherwise loses the only
fact that explains its size.

Idempotent: re-running overwrites the stamp rather than appending a second one.
"""
import json, sys
from pathlib import Path

OUT = Path("perf/of3t_gradients")
WITHDRAWN = "1a6af8bba8c5fa33dd5366f15e87fffc22595a351c3c77b476a56fab8b4b4b74"
REPUBLISHED = "89457d8977327699c84fc90741a013bc369f835dea6008676492c786fb87f113"

#: file -> (standing, what supersedes it and why)
STAMPS = {
    "reach_by_norm.json": (
        "CURRENT",
        f"RE-RUN against the republished r = 0 tape grads_f64_r0.pt ({REPUBLISHED}), not "
        f"carried over from the withdrawn train-mode one ({WITHDRAWN}). An earlier stamp said "
        "the figure could not move because grad_presence_r0.json is byte-identical to "
        "grad_presence_recycles0.json (sha256 "
        "3034929504c51c815ab5cfc6d6972e6db9cfdba423ad00370f2f11c845a70f1f on both). That is "
        "true of the reached SET, which stays at 3545 of 4147, and false of every norm share, "
        "which is computed from the gradient's VALUES: reach 98.01 % -> 98.6949 %, K22 tracer "
        "6.54 % -> 4.0545 %, pairformer_stack 5.27 % -> 3.1560 %, instrument A block 0 "
        "0.20 % -> 0.0861 %. Recomputed global norm 3.707776369277738 against the manifest's "
        "3.707776369277739."),
    "instrument_a_bundle_block0.json": (
        "HISTORICAL",
        "Taken against the withdrawn train-mode tape. Its reference side is one dropout draw "
        "(D18), whose own draw-to-draw floor is median 0.550 -- an order of magnitude above the "
        "5.0e-02 bar. Superseded for the bundle comparison by replay_vs_r0.json; the r = 0 "
        "block measurements in instrument_a_bundle_block*_r0*.json are unaffected because their "
        "reference is upstream's own float64 PairFormerBlock on the captured r = 0 boundary, "
        "not the published tape."),
    "full_model_of3_full_mat64.json": (
        "SUPERSEDED",
        "presence.ours_carried = 3497 here predates the confidence-head call. "
        "full_model_of3_full_mat64_conf.json and reach_by_norm.json carry the current answer, "
        "3545 of 4147 holding 98.01 % of the squared gradient norm. The 48-tensor gap is exactly "
        "the aux_heads.distogram call and nothing else changed."),
    "full_model_of3_full_mat64_conf.json": (
        "CURRENT",
        "3545 of 4147 tensors carried, 98.01 % of the squared gradient norm. Presence is "
        "unchanged under the republished r = 0 reference: grad_presence_r0.json is byte-"
        "identical to grad_presence_recycles0.json."),
    "capture_trunk_boundary_nodropout.json": (
        "HISTORICAL",
        "Its capture_vs_bundle block compares this r = 0 CPU replay against the WITHDRAWN "
        f"train-mode tape ({WITHDRAWN}) and therefore reads worst 2.256, median 1.096. "
        "replay_vs_r0.json re-runs the identical comparison against the republished r = 0 tape "
        f"({REPUBLISHED}) and reads median 6.2244e-02 on the same 171 tensors."),
}


def main():
    changed = []
    for name, (standing, why) in STAMPS.items():
        p = OUT / name
        if not p.is_file():
            print(f"absent, skipped: {name}")
            continue
        d = json.loads(p.read_text())
        d["reference_standing"] = {
            "standing": standing,
            "reference_withdrawn_sha256": WITHDRAWN,
            "reference_republished_sha256": REPUBLISHED,
            "note": why,
            "stamped_by": "perf/of3t_gradients/restate_references.py",
        }
        if standing != "CURRENT":
            d["superseded_by"] = why
        p.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
        changed.append(name)
        print(f"{standing:11s} {name}")
    print(f"{len(changed)} artifacts stamped")


if __name__ == "__main__":
    sys.exit(main())
