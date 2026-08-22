"""Fold the per-cell reports from a rented box into perf/openbind/gpu_reference.json.

    python perf/openbind/make_reference.py --results perf/openbind/results \
        --out perf/openbind/gpu_reference.json

The port's perf pass reads that JSON as its target. `tt_target_device_s = 4 * h200_device_s` per
rung at matched sample count: the org bar restated. Within 4x of one H200 per chip at matched
batch is the same bar as one 32-chip Galaxy beating one 8-GPU DGX H200.

Device time is the target, not the wall clock. On the RF3 campaign two independently rented H200s
agreed on device time within 1% at >=512 aa but differed up to 2.3x on host featurisation, which
was 39-64% of the wall clock. The port inherits the same host pipeline on its own CPU, so host
time is common cost to both arms, not a TT deficit. `tt_target_fold_s` is carried for
completeness and is only meaningful if both arms' host prep ran on comparable CPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

TARGET_MULTIPLE = 4.0


def cell_row(rep: dict) -> dict:
    spec = rep.get("spec_content") or {}
    shapes = rep.get("shapes") or []
    warm = shapes[1:] if len(shapes) > 1 else shapes
    dev = rep.get("device_s") or {}
    n_lig = max([s.get("n_is_ligand", 0) for s in warm] or [0])
    n_tok = 0
    for s in warm:
        tm = s.get("token_mask") or s.get("residue_index") or []
        if tm:
            n_tok = max(n_tok, tm[-1])
    n_atom = 0
    for s in warm:
        rm = s.get("ref_mask") or []
        if rm:
            n_atom = max(n_atom, rm[-1])
    counts = rep.get("counts") or {}
    cueq = {k: v for k, v in counts.items() if k.startswith("cueq:") or k.startswith("of3:")}
    row = {
        "cell": rep.get("label"),
        "spec": pathlib.Path(rep.get("spec", "")).name,
        "n_residues": spec.get("n_residues"),
        "ligand_ccd": spec.get("ligand_ccd"),
        "ligand_heavy_atoms_formula": spec.get("ligand_heavy_atoms_formula"),
        "n_ligand_tokens_measured": n_lig or None,
        "n_tokens_measured": n_tok or None,
        "n_atoms_measured": n_atom or None,
        "diffusion_samples": rep.get("samples"),
        "seed": rep.get("seed"),
        "h200_device_s": dev.get("median_s"),
        "h200_device_spread_pct": (round(100.0 * (dev["max_s"] - dev["min_s"]) / dev["median_s"], 2)
                                   if dev.get("median_s") else None),
        "h200_device_all_s": dev.get("all_s"),
        "h200_cold_device_s": rep.get("cold_device_s"),
        "h200_host_s": rep.get("host_s"),
        "h200_wall_total_s": rep.get("wall_s"),
        "trunk_s": (rep.get("warm_trunk") or {}).get("median_s"),
        "rollout_s": (rep.get("warm_rollout") or {}).get("median_s"),
        "confidence_heads_s": (rep.get("warm_confidence_heads") or {}).get("median_s"),
        "confidence_scores_s": (rep.get("warm_confidence_scores") or {}).get("median_s"),
        "peak_vram_reserved_GiB": (round(rep["peak_vram_reserved_B"] / 2**30, 3)
                                   if rep.get("peak_vram_reserved_B") else None),
        "peak_vram_alloc_GiB": (round(rep["peak_vram_alloc_B"] / 2**30, 3)
                                if rep.get("peak_vram_alloc_B") else None),
        "power": rep.get("power"),
        "kernel_counts": cueq,
        "torch_sdpa_calls": counts.get("torch_sdpa"),
        "diffusion_module_calls": counts.get("calls:diffusion_module"),
        "gpu_exclusive": rep.get("gpu_exclusive"),
        "ok": rep.get("ok"),
        "why": rep.get("why"),
    }
    if row["h200_device_s"] is not None:
        row["tt_target_device_s"] = round(TARGET_MULTIPLE * row["h200_device_s"], 3)
    if row["h200_device_s"] is not None and row["h200_host_s"] is not None:
        row["tt_target_fold_s"] = round(
            TARGET_MULTIPLE * row["h200_device_s"] + row["h200_host_s"], 3)
    return row


def _by_cell(arm: dict | None) -> dict:
    return {r["cell"]: r for r in (arm or {}).get("cells", [])}


def _ratio(a, b):
    return round(a / b, 4) if (a and b) else None


def delta_table(ob: dict | None, p2: dict | None) -> dict:
    """What upgrading tt-bio from preview2 to OB0 costs, in GPU seconds, on one card in one
    session. Both arms folded the same committed inputs back to back, so the card, the driver
    and the host are held fixed and only the weights + code version move."""
    a, b = _by_cell(ob), _by_cell(p2)
    rows = []
    for cell in sorted(set(a) & set(b)):
        x, y = a[cell], b[cell]
        rows.append({
            "cell": cell,
            "n_residues": x.get("n_residues"),
            "diffusion_samples": x.get("diffusion_samples"),
            "ob_device_s": x.get("h200_device_s"),
            "p2_device_s": y.get("h200_device_s"),
            "ob_speedup_x": _ratio(y.get("h200_device_s"), x.get("h200_device_s")),
            "ob_trunk_s": x.get("trunk_s"), "p2_trunk_s": y.get("trunk_s"),
            "trunk_speedup_x": _ratio(y.get("trunk_s"), x.get("trunk_s")),
            "ob_rollout_s": x.get("rollout_s"), "p2_rollout_s": y.get("rollout_s"),
            "rollout_speedup_x": _ratio(y.get("rollout_s"), x.get("rollout_s")),
            "ob_cueq_triangle_attention": (x.get("kernel_counts") or {}).get(
                "cueq:triangle_attention"),
            "p2_cueq_triangle_attention": (y.get("kernel_counts") or {}).get(
                "cueq:triangle_attention"),
        })
    return {"note": ("Same box, same session, same card. `*_speedup_x` > 1 means OB0 is faster "
                     "than preview2. Counts are per cell, i.e. over all 4 folds."),
            "rows": rows}


def cross_box(primary: dict | None, confirm: dict | None) -> dict:
    """Does the published device number reproduce on a second, independently rented H200?

    `gpu-reference-device-vs-host-split` was found exactly this way on the RF3 campaign: a
    single rental would have shipped the wrong bar. Here the answer is size-dependent, which is
    itself the finding -- see the state file.
    """
    a, b = _by_cell(primary), _by_cell(confirm)
    rows = []
    for cell in sorted(set(a) & set(b)):
        x, y = a[cell], b[cell]
        def d(k):
            p, q = x.get(k), y.get(k)
            return round(100.0 * (q - p) / p, 2) if (p and q) else None
        rows.append({"cell": cell, "n_residues": x.get("n_residues"),
                     "primary_device_s": x.get("h200_device_s"),
                     "confirm_device_s": y.get("h200_device_s"),
                     "device_delta_pct": d("h200_device_s"),
                     "trunk_delta_pct": d("trunk_s"),
                     "rollout_delta_pct": d("rollout_s"),
                     "host_delta_pct": d("h200_host_s")})
    return {"note": ("Percent change from the primary box to the confirmation box, same stack, "
                     "same committed inputs, different machine_id. The trunk reproduces; the "
                     "200-step diffusion rollout does not below 1024 aa, because it is "
                     "launch-bound and therefore carries the landlord's CPU into what the "
                     "harness calls device time."),
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir with <arm>/<cell>.json reports")
    ap.add_argument("--inputs", default=None, help="dir whose files get sha256'd into the record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", default="", help="free-text note about the rented box")
    args = ap.parse_args()

    res = pathlib.Path(args.results)
    arms: dict[str, dict] = {}
    env_by_arm: dict[str, dict] = {}
    for arm_dir in sorted(p for p in res.iterdir() if p.is_dir()):
        rows = []
        for j in sorted(arm_dir.glob("*.json")):
            rep = json.loads(j.read_text())
            rows.append(cell_row(rep))
            env_by_arm.setdefault(arm_dir.name, rep.get("env") or {})
            env_by_arm[arm_dir.name].setdefault("_ckpt", rep.get("ckpt"))
        arms[arm_dir.name] = {"cells": rows}

    inputs_dir = pathlib.Path(args.inputs) if args.inputs \
        else pathlib.Path(__file__).parent / "inputs"
    sha = {}
    if inputs_dir.exists():
        for p in sorted(inputs_dir.iterdir()):
            if p.is_file() and p.name != "SHA256SUMS":
                sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()

    doc = {
        "what": ("H200 GPU reference for OpenBind-0 (OB0), the denominator the tt-bio OB0 port is "
                 "scored against. Device time is the bar; host time is common cost."),
        "target_rule": ("tt_target_device_s = %g * h200_device_s at matched diffusion_samples. "
                        "Within 4x of one H200 per chip at matched batch is the same bar as one "
                        "32-chip Galaxy beating one 8-GPU DGX H200." % TARGET_MULTIPLE),
        "arms": {
            "ob": ("OpenBind-0: openfold3 0.5.0 + of3-ob-2025-06-30-174k.pt. The upstream "
                   "default checkpoint since v0.5.0."),
            "p2": ("OpenFold3 preview2: openfold3 0.4.5 + of3-p2-155k.pt, what tt-bio pins "
                   "today. Upstream made the two weight sets mutually incompatible, so this "
                   "arm differs from `ob` in BOTH weights and code version. That is not a "
                   "choice the harness made and it cannot be separated."),
            "confirm-ob": ("The same OB0 arm on a SECOND, independently rented H200 (different "
                           "machine_id, different landlord, different driver). It exists to "
                           "answer whether the published device number reproduces at all. It "
                           "does above 1024 aa and does not below -- see cross_box."),
        },
        "protocol": {
            "folds_per_cell": "1 cold (discarded) + 3 warm, one process, checkpoint loaded once",
            "reported": "median of the 3 warm folds",
            "device_boundary": ("OpenFold3AllAtom.predict_step, cuda-synchronised on both sides. "
                                "Featurisation runs in the dataloader and writing in the "
                                "callback, so both land in host time."),
            "msa": "none, single sequence. MSA depth is a separate axis.",
            "templates": "none",
            "recycles": 3,
            "diffusion_steps": 200,
            "seed": 42,
            "tf32": ("on, the shipped default of `run_openfold predict` (--use_tf32 default "
                     "True). Running it any other way would not be the fast path a researcher "
                     "gets."),
            "exclusivity": ("every cell records nvidia-smi compute apps before and after plus "
                            "sampled power draw, and fails if the card was ever shared"),
        },
        "box": args.box,
        "env_by_arm": env_by_arm,
        "input_sha256": sha,
        "results": arms,
    }
    doc["caveats"] = [
        ("Only the 1024 aa rung reproduces across landlords: device time moved +1.2% there but "
         "+10.7 to +17.0% at 128-512 aa between two independently rented H200s. The trunk "
         "reproduces to 0.3%; the 200-step diffusion rollout does not, because it is launch-bound "
         "and carries the host CPU into the device measurement. Treat 128-512 aa as soft."),
        ("The fused cuEquivariance triangle-attention kernel falls back to torch at 128 tokens "
         "and only there: cueq:triangle_attention._triangle_attention_torch reads 264/1966 on OB0 "
         "and 1050/2752 on preview2 at the 128 aa rung, 0 at every larger cell, on both boxes. "
         "The fallback is silent (_warn_triangle_attention_fallback is 0)."),
        ("The H200 is never saturated: 22% of its 700 W cap at 128 aa rising to only 74% at "
         "1024 aa. The 4x bar is generous everywhere on this workload."),
        ("OB0's entire speedup over preview2 is the diffusion rollout, and it comes from v0.5.0 "
         "dropping the attention-pair-bias kernel route that upstream marks '# TODO: Add back "
         "triton and cueq APB kernel'. Re-measure when that lands; the trunk is unchanged."),
        ("The OpenBind blog's 'chemical steering during diffusion sampling' is not in v0.5.0's "
         "code. Grepping the tree for steer/guidance/restraint finds one unrelated test "
         "docstring. Treat it as unreleased."),
    ]
    doc["delta_ob_vs_p2"] = delta_table(arms.get("ob"), arms.get("p2"))
    doc["cross_box"] = cross_box(arms.get("ob"), arms.get("confirm-ob"))

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(doc, indent=2) + "\n")

    for arm, d in arms.items():
        print("== %s ==" % arm)
        print("%-20s %5s %4s %10s %12s %10s %8s %s"
              % ("cell", "aa", "smp", "device_s", "tt_target_s", "host_s", "VRAM", "ok"))
        for r in d["cells"]:
            print("%-20s %5s %4s %10s %12s %10s %8s %s"
                  % (r["cell"], r["n_residues"], r["diffusion_samples"], r["h200_device_s"],
                     r.get("tt_target_device_s"), r["h200_host_s"],
                     r["peak_vram_reserved_GiB"], r["ok"]))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
