#!/usr/bin/env python3
"""What does a precision arm's gradient read against BUNDLE-MIN-043's float64 gradient?

One instrument, run over every parameter of the model rather than over a section, because the
question this row asks is about the training recipe and not about a port. A23 governs the
reporting: the headline for any set is the mass-weighted `rel_l2` over the concatenated set,
the median over tensors goes beside it and never instead of it, and every set statistic carries
the share of the reference's squared gradient norm that its set holds.

Per tensor it emits `rel_l2`, the norm ratio `r` and the cosine, because `rel` alone bounds `r`
to [1-rel, 1+rel] and cannot separate a magnitude error from a direction error. The whole
per-tensor array goes to a sidecar: a file that keeps only the extremes cannot be re-scored
under a denominator discovered later, and this campaign has changed denominator twice.
"""
import argparse
import json
import math
import re
from pathlib import Path

import torch

# A14: a reference norm below this cannot carry a relative error. Such tensors are reported as
# unmeasurable with their absolute values, never excluded silently -- and A23 is the reason they
# are not excluded from the mass-weighted statistic either: they are WEIGHTED, and their weight
# is what makes them harmless.
REF_NORM_FLOOR = 1e-12
MASS_BAR = 2.0e-02        # the campaign's mass-weighted bar
PER_TENSOR_BAR = 5.0e-02  # PROTOCOL 3d's per-tensor bar

# The leaf of3t-orchestrator localises the device arm's failure to: 24 tensors, 25.5795 % of the
# model, mass-weighted 10.6980 at norm ratio 7.8658 on device
# (perf/of3t_orchestrator/DISTANCE_TO_GO_BY_MASS.json, measured_and_outside_bar.items[0]).
ONE_LEAF = re.compile(
    r"^diffusion_module\.diffusion_transformer\.blocks\.\d+\."
    r"attention_pair_bias\.layer_norm_a\.layer_norm_s\.weight$")

# What the device arm reads on the individual tensors this row is asked to name, so upstream's
# own fp32 lands beside it rather than in a separate document. Provenance is PER ENTRY.
DEVICE_READS = {
    "diffusion_module.diffusion_transformer.blocks.8.attention_pair_bias"
    ".layer_norm_a.layer_norm_s.weight": {
        "device_rel_l2": 18.504,
        "source": "workstreams/of3t-refprec.txt, quoting of3t-orchestrator D62/D63: "
                  "'the tensor our device reads at rel 18.504'",
    },
}

ATTENTION_SIDE = re.compile(
    r"^diffusion_module\.(diffusion_transformer\.blocks\.\d+"
    r"|atom_attn_(enc|dec)\.atom_transformer\.blocks\.\d+)\.attention_pair_bias\.")


def per_tensor(ref, arm):
    """One row per parameter of the reference, with the arm's reading beside it."""
    rows = []
    for name, b in ref.items():
        row = {"param": name}
        b64 = None if b is None else b.to(torch.float64).reshape(-1)
        row["ref_present"] = b is not None
        row["arm_present"] = name in arm and arm[name] is not None
        rb = 0.0 if b64 is None else float(torch.linalg.vector_norm(b64))
        row["ref_norm"] = rb
        row["ref_sq"] = rb * rb
        if not row["arm_present"] or b64 is None:
            # The arm supplying no gradient where the reference has one is not "unmeasurable":
            # the arm's value there IS zero, and scoring it as a missing row would silently
            # remove its mass from the numerator while leaving it in the denominator. Score it
            # exactly the way the zero model is scored, and still name it.
            if b64 is not None and rb >= REF_NORM_FLOOR:
                row.update(arm_norm=0.0, diff_norm=rb, dot=0.0, rel_l2=1.0, r=0.0, cos=0.0,
                           unmeasurable="absent from the arm, scored as zero")
            else:
                row.update(arm_norm=None, diff_norm=None, dot=0.0, rel_l2=None, r=None,
                           cos=None, unmeasurable="absent on one side")
            rows.append(row)
            continue
        a64 = arm[name].to(torch.float64).reshape(-1)
        ra = float(torch.linalg.vector_norm(a64))
        d = float(torch.linalg.vector_norm(a64 - b64))
        row.update(arm_norm=ra, diff_norm=d, dot=float(torch.dot(a64, b64)))
        if rb < REF_NORM_FLOOR:
            row.update(rel_l2=None, r=None, cos=None,
                       unmeasurable=f"ref_norm {rb:.3e} < {REF_NORM_FLOOR:.0e} (A14)")
        else:
            row["rel_l2"] = d / rb
            row["r"] = ra / rb
            row["cos"] = (row["dot"] / (ra * rb)) if ra > 0 else 0.0
        rows.append(row)
    for name in arm:
        if name not in ref:
            rows.append({"param": name, "ref_present": False, "arm_present": True,
                         "ref_norm": 0.0, "ref_sq": 0.0, "rel_l2": None, "r": None, "cos": None,
                         "unmeasurable": "absent from the reference"})
    return rows


def stat(rows, total_sq, name):
    """A23: mass-weighted headline, median over tensors beside it, mass share always."""
    sq = sum(r["ref_sq"] for r in rows)
    d2 = sum((r["diff_norm"] or 0.0) ** 2 for r in rows)
    a2 = sum((r.get("arm_norm") or 0.0) ** 2 for r in rows)
    dot = sum(r.get("dot", 0.0) or 0.0 for r in rows)
    rels = sorted(r["rel_l2"] for r in rows if r["rel_l2"] is not None)
    med = (rels[len(rels) // 2] if len(rels) % 2 else
           0.5 * (rels[len(rels) // 2 - 1] + rels[len(rels) // 2])) if rels else None
    return {
        "set": name,
        "n": len(rows),
        "pct_of_model_mass": 100.0 * sq / total_sq if total_sq else 0.0,
        "mass_weighted_rel_l2": math.sqrt(d2 / sq) if sq else None,
        "mass_weighted_norm_ratio": math.sqrt(a2 / sq) if sq else None,
        "mass_weighted_cos": (dot / math.sqrt(a2 * sq)) if a2 > 0 and sq > 0 else None,
        "inside_mass_bar": (math.sqrt(d2 / sq) <= MASS_BAR) if sq else None,
        "median_rel_l2_over_tensors": med,
        "n_rel_measurable": len(rels),
        "n_over_per_tensor_bar": sum(1 for x in rels if x > PER_TENSOR_BAR),
        "pct_of_model_inside_per_tensor_bar": 100.0 * sum(
            r["ref_sq"] for r in rows
            if r["rel_l2"] is not None and r["rel_l2"] <= PER_TENSOR_BAR) / total_sq
        if total_sq else 0.0,
        "worst_rel_l2": max(rels) if rels else None,
        "worst_tensor": max((r for r in rows if r["rel_l2"] is not None),
                            key=lambda r: r["rel_l2"])["param"] if rels else None,
        "heaviest_tensor": max(rows, key=lambda r: r["ref_sq"])["param"] if rows else None,
    }


def section_of(name, sections):
    best = None
    for key in sections:
        if name == key or name.startswith(key + "."):
            if best is None or len(key) > len(best):
                best = key
    return best or name.split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--arm", required=True, action="append",
                    help="label=path/to/grads.pt, repeatable. The reference is loaded once.")
    ap.add_argument("--manifest", action="append", default=[],
                    help="label=path/to/manifest.json, for the loss-reproduction control")
    ap.add_argument("--sections", type=Path, required=True)
    ap.add_argument("--ref-loss", type=float, default=1.267624369070698)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sidecar-dir", required=True, type=Path)
    args = ap.parse_args()

    sections = list(json.loads(args.sections.read_text())["sections_pct_of_model"])
    print(f"loading reference {args.ref}", flush=True)
    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    total_sq = sum(float(torch.linalg.vector_norm(v.to(torch.float64))) ** 2
                   for v in ref.values() if v is not None)
    print(f"reference: {len(ref)} tensors, squared norm {total_sq!r}", flush=True)
    published = 10.279642678524985  # DISTANCE_TO_GO_BY_MASS.json, "what"
    assert abs(total_sq - published) < 1e-12, (
        f"denominator {total_sq!r} is not the campaign's published {published!r}; every share "
        "in this file would be in a different denominator than every share it is compared to")

    args.sidecar_dir.mkdir(parents=True, exist_ok=True)
    manifests = dict(m.split("=", 1) for m in args.manifest)
    arms = {}

    # A16, measured and not asserted: the arm replaced by zeros. Run through the same code path.
    todo = [("ZERO_MODEL_BASELINE", None)] + [tuple(a.split("=", 1)) for a in args.arm]

    for label, path in todo:
        if path is None:
            arm = {k: (torch.zeros_like(v) if v is not None else None) for k, v in ref.items()}
        else:
            print(f"loading arm {label} from {path}", flush=True)
            arm = torch.load(path, map_location="cpu", weights_only=False)
        rows = per_tensor(ref, arm)
        del arm
        by_name = {r["param"]: r for r in rows}
        for r in rows:
            r["section"] = section_of(r["param"], sections)
            r["pct_of_model_mass"] = 100.0 * r["ref_sq"] / total_sq if total_sq else 0.0

        arm_rows = [r for r in rows if r["param"].startswith("diffusion_module.")
                    and not r["param"].startswith("diffusion_module.diffusion_conditioning.")]
        att = [r for r in arm_rows if ATTENTION_SIDE.match(r["param"])]
        rest = [r for r in arm_rows if not ATTENTION_SIDE.match(r["param"])]

        sets = [stat(rows, total_sq, "MODEL (all tensors)")]
        sets.append(stat(arm_rows, total_sq,
                         "diffusion device-arm scope (diffusion_module minus "
                         "diffusion_conditioning), FULL denominator per A20"))
        sets.append(stat(att, total_sq, "attention side of the diffusion arm"))
        sets.append(stat(rest, total_sq, "rest of the diffusion arm"))
        leaf = [r for r in rows if ONE_LEAF.match(r["param"])]
        sets.append(stat(leaf, total_sq,
                         "the one leaf (blocks.N.attention_pair_bias.layer_norm_a"
                         ".layer_norm_s.weight), device reads 10.6980"))
        sets.append(stat([r for r in arm_rows if not ONE_LEAF.match(r["param"])], total_sq,
                         "diffusion device-arm scope minus the one leaf, "
                         "device reads 0.2929"))
        for sec in sections:
            sets.append(stat([r for r in rows if r["section"] == sec], total_sq,
                             f"section {sec}"))

        heavy = sorted(rows, key=lambda r: -r["ref_sq"])[:12]
        ordered = sorted((r for r in rows if r["rel_l2"] is not None),
                         key=lambda r: -r["ref_sq"])
        rep = {
            "label": label,
            "reference": {"file": args.ref.name, "n_tensors": len(ref),
                          "model_squared_gradient_norm": total_sq},
            "bars": {"mass_weighted": MASS_BAR, "per_tensor": PER_TENSOR_BAR},
            "headline": None,
            "sets": sets,
            "heaviest_tensors_named_individually": [
                {"param": r["param"], "pct_of_model_mass": r["pct_of_model_mass"],
                 "rel_l2": r["rel_l2"], "r": r["r"], "cos": r["cos"],
                 "ref_norm": r["ref_norm"], "arm_norm": r.get("arm_norm")}
                for r in heavy],
            "mass_reached_by_the_top_8_tensors_pct": sum(
                r["pct_of_model_mass"] for r in ordered[:8]),
            "named_tensors_beside_the_device_reading": [
                {"param": name,
                 "pct_of_model_mass": by_name[name]["pct_of_model_mass"],
                 "this_arm_rel_l2": by_name[name]["rel_l2"],
                 "this_arm_r": by_name[name]["r"],
                 "this_arm_cos": by_name[name]["cos"],
                 **d}
                for name, d in DEVICE_READS.items() if name in by_name],
            "unmeasurable": [{"param": r["param"], "why": r["unmeasurable"],
                              "ref_norm": r["ref_norm"], "arm_norm": r.get("arm_norm"),
                              "pct_of_model_mass": r["pct_of_model_mass"]}
                             for r in rows if r.get("unmeasurable")],
        }
        m = sets[0]

        def fmt(x, spec=".6e"):
            return "n/a" if x is None else format(x, spec)

        rep["headline"] = (
            f"mass-weighted rel_l2 {fmt(m['mass_weighted_rel_l2'])} over {m['n']} tensors "
            f"holding {fmt(m['pct_of_model_mass'], '.4f')} % of the model's squared gradient "
            f"norm, against the {MASS_BAR:.1e} bar "
            f"({'INSIDE' if m['inside_mass_bar'] else 'OUTSIDE'}); "
            f"median over tensors {fmt(m['median_rel_l2_over_tensors'])} beside it, "
            f"norm ratio {fmt(m['mass_weighted_norm_ratio'], '.6f')}, cos "
            f"{fmt(m['mass_weighted_cos'], '.6f')}")
        if label in manifests:
            mf = json.loads(Path(manifests[label]).read_text())
            loss = mf["loss"]
            rep["loss_reproduction"] = {
                "reference_loss": args.ref_loss,
                "arm_loss": loss,
                "abs_diff": abs(loss - args.ref_loss),
                "rel_diff": abs(loss - args.ref_loss) / abs(args.ref_loss),
                "bit_identical_on_rng_replay_within_the_arm":
                    mf.get("loss_bit_identical_on_replay"),
                "cast_policy": mf.get("cast_policy"),
                "dtype": mf.get("dtype"),
                "gradient_global_norm": mf["gradient"]["global_norm"],
                "n_with_gradient": mf["gradient"]["n_with_gradient"],
                "draws_replayed": (mf.get("replayed_draws") or {}).get("file"),
                "draws_sha256": (mf.get("replayed_draws") or {}).get("sha256"),
                "n_draw_mismatch": (mf.get("replayed_draws") or {}).get("n_mismatch"),
            }
        arms[label] = rep
        side = args.sidecar_dir / f"per_tensor_{label}.json"
        side.write_text(json.dumps(
            [{k: r[k] for k in ("param", "section", "pct_of_model_mass", "ref_norm",
                                "arm_norm", "diff_norm", "rel_l2", "r", "cos")}
             for r in sorted(rows, key=lambda r: -r["ref_sq"])], indent=0) + "\n")
        rep["per_tensor_sidecar"] = side.name
        print(label, "->", rep["headline"], flush=True)
        del rows, by_name

    args.out.write_text(json.dumps({"arms": arms}, indent=1, sort_keys=False) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
