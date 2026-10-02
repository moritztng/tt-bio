#!/usr/bin/env python3
"""Refold ONE fixed binder sequence against a chosen target and read the screen gate's own metric.

This isolates the miscalibration half of `bcw-accept`'s screen-gate finding from the gradient half.
`bcw-accept` measured 9 of 11 trajectories at ~800 tokens dying at the screen gate against 1 of 10
at ~352. The gate is one threshold: `min_plddt_screen = 0.60` on `plddt_metric` bound to the
designed binder chain of the complex prediction (`bindcraft/filters.py:826-830`). If the SAME
binder sequence reads materially lower against the full ectodomain than against the domain III
crop, then part of the deficit is the constant being wrong for the axis rather than the design
being worse -- and that is a product question, not a bug in our gradient.

`bindcraft score` cannot answer this: it scores a structure you already have, and here the point is
to refold a known sequence against a DIFFERENT target than the one it was designed against.

WHICH SEQUENCE, and this is the trap the row walked into first. The hash `d648869e8c237d46` carries
two different 150 aa sequences: the trajectory sequence (pre-MPNN, `!_Trajectories.csv` pLDDT 0.73)
and the accepted MPNN redesign `seq0` (`!_Ranked.csv` pLDDT 0.77). The screen gate fires at the
screen DESIGN stage, i.e. on the trajectory sequence. Refolding `seq0` would compare a post-MPNN
sequence against a pre-MPNN threshold and the resulting "miscalibration" would be an artifact of
the sequence swap. Pass the trajectory sequence.

RUN THE CONTROL FIRST. `--target hEGFR_d3` on the trajectory sequence should land near its recorded
0.73; that is what shows the harness is reading the pipeline's own quantity rather than a lookalike
of it. Only then is `--target hEGFR` (the full ectodomain) interpretable. A single 800-token number
with no reproduced 352 control attributes nothing.

Everything here is reused rather than reimplemented: the settings path is `rung.py`'s
(`read_settings` + `parse_setting_overrides` + `cleaned_campaign_settings` over `targets.json`),
the states come from `initialize_design_trajectory`, the sequence is injected with the shipped
`update_shared_sequences`, the predictor is `bindcraft2.design_model_class()` built as
`campaign.run_campaign` builds it, and the number is `filters.plddt_metric` itself.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "perf" / "bgx_size")):
    if p not in sys.path:
        sys.path.insert(0, p)

TARGETS = json.loads((ROOT / "perf" / "bgx_size" / "targets.json").read_text())


def resolve(path: str) -> str:
    """`rung.py`'s, so a relative `targets.json` path means the same thing in both."""
    return path if os.path.isabs(path) else str(ROOT / path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    help="a name in perf/bgx_size/targets.json, or 'custom' with --target-path")
    ap.add_argument("--target-path", default="", help="overrides targets.json, for the d3 crop")
    ap.add_argument("--chains", default="")
    ap.add_argument("--hotspots", default="")
    ap.add_argument("--binder-seq", required=True,
                    help="the amino-acid sequence, or @<path> to read one line from a file")
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--seed", type=int, default=100, help="campaign_seed, as the arms ran it")
    ap.add_argument("--memory", default="auto")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params",
                    help="AF2 weights dir, the --params campaign.sh passes rung.py")
    ap.add_argument("--recycles", type=int, default=-1,
                    help="-1 keeps the shipped design_recycles, which is what the gate saw")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--dry-run", action="store_true",
                    help="build the settings, the states and the fixed binder, print the residue "
                         "accounting, and stop before the predictor. Needs jax but no chip, so "
                         "every host-side API mismatch surfaces without spending a card lease")
    a = ap.parse_args()

    seq = a.binder_seq
    if seq.startswith("@"):
        # a FASTA or a bare sequence file; '>' headers are dropped rather than read as the sequence
        lines = [l.strip() for l in pathlib.Path(seq[1:]).read_text().splitlines()]
        seq = "".join(l for l in lines if l and not l.startswith(">"))
    seq = seq.strip().upper()
    bad = sorted(set(seq) - set("ARNDCQEGHILKMFPSTWYV"))
    if bad or not seq:
        raise SystemExit(f"--binder-seq gave {len(seq)} residues"
                         + (f" with non-standard amino acid(s) {bad}" if bad else ""))

    spec = dict(TARGETS.get(a.target, {}))
    if a.target_path:
        spec["path"] = a.target_path
    if a.chains:
        spec["chains"] = a.chains
    if a.hotspots:
        spec["hotspots"] = a.hotspots
    for need in ("path", "chains", "hotspots"):
        if not spec.get(need):
            raise SystemExit(f"--target {a.target!r} has no {need}; pass --{need.replace('_','-')}")

    if not a.dry_run:
        os.environ.setdefault("TT_VISIBLE_DEVICES", str(a.card))
        from tt_bio.main import ensure_p300_mesh_descriptor
        ensure_p300_mesh_descriptor()
        from tt_bio import bindcraft2

    from bindcraft.settings import (build_design_settings, parse_setting_overrides, read_settings,
                                    select_design_and_validation_models, resolve_cyclic_offset_mode)
    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.protein import BINDER_ALONE, Protein
    from bindcraft.prediction import update_shared_sequences
    from bindcraft.protein_preparation import design_residue_count, initialize_design_trajectory
    from bindcraft.design_workers import campaign_subbatch_size
    from bindcraft.af2 import (campaign_length_bucket, pad_design_chains,
                               padded_prediction_length)
    from bindcraft.campaign import MONOMER_POOL, MULTIMER_POOL
    from bindcraft.trajectory import pooled_stage_predictions
    from bindcraft import filters as F
    import jax

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # The settings a user would write, exactly as rung.py composes them: the shipped campaign with
    # its target swapped, inline rather than written into the BindCraft 2 install.
    request = {"modality": "binder", "campaign_name": f"refold_{a.target}_{len(seq)}",
               "number_of_final_designs": 1,
               "targets": [{"name": a.target, "target_path": resolve(spec["path"]),
                            "chains": spec["chains"], "hotspots": spec["hotspots"]}]}
    req = out / f"refold_request_{a.target}.json"
    req.write_text(json.dumps(request, indent=1))
    overrides = [f"campaign_seed={a.seed}", "max_trajectories=1", f"project_folder={out}",
                 f"binder_lengths=[{len(seq)}]", "compile_next_length=0"]
    if a.recycles >= 0:
        overrides.append(f"design_recycles={a.recycles}")
    settings = cleaned_campaign_settings(read_settings(req, parse_setting_overrides(overrides)))
    design_settings = build_design_settings(settings)

    # States with a randomly initialised binder of the right length, then the shipped sequence
    # transfer to pin it to the one we are grading. One-hot via from_fasta, so the alphabet and
    # the residue flags are the package's own.
    key = jax.random.PRNGKey(settings.get("campaign_seed") or 0)
    protein_states, multi_chain_binders, _losses = initialize_design_trajectory(design_settings, key)
    binder_chain = design_settings.designed_binder_chain
    fasta = out / f"refold_binder_{len(seq)}.fasta"
    fasta.write_text(f">{binder_chain}\n{seq}\n")
    fixed = Protein.from_fasta(str(fasta), chain_letter=binder_chain)
    if len(fixed) != len(seq):
        raise SystemExit(f"binder built {len(fixed)} residues from a {len(seq)} aa sequence")
    protein_states = update_shared_sequences(protein_states, {binder_chain: fixed})

    selected = select_design_and_validation_models(settings, MULTIMER_POOL, MONOMER_POOL)
    bucket = campaign_length_bucket(settings)
    subbatch = campaign_subbatch_size(settings, design_residue_count(settings))

    chain_residues = {state: {c: len(p) for c, p in cx.items()}
                      for state, cx in protein_states.items()}
    tokens = max(sum(v.values()) for v in chain_residues.values())
    # ONE campaign carries TWO token axes and conflating them misattributes the whole row.
    # `predict` (this harness, and the screen gate) pads chains only when target_pad_length is
    # set, which is 0 for a single target, so arm B's complex is 614 + 150 = 764 -> 768.
    # `sequence_gradients` (the gradient rounds, and so the VJP this row grades) calls
    # pad_design_chains unconditionally, padding the binder 150 -> 160, so the same campaign's
    # gradient runs at 614 + 160 = 774 -> 800. Arm A is 352 on both paths.
    grad_chains = pad_design_chains(protein_states, bucket, 0)
    grad_sum = max(sum(len(p) for p in cx.values()) for cx in grad_chains.values())
    accounting = {"target": a.target, "binder_aa": len(seq), "binder_chain": binder_chain,
                  "chains": chain_residues, "tokens_unpadded": tokens,
                  "predict_axis": padded_prediction_length(tokens, bucket),
                  "gradient_axis": padded_prediction_length(grad_sum, bucket),
                  "subbatch_size": subbatch,
                  "design_models": list(selected.design_models),
                  "design_recycles": settings["design_recycles"],
                  "min_plddt_screen": settings.get("min_plddt_screen")}
    print(json.dumps(accounting), flush=True)
    if a.dry_run:
        print("dry run: states and settings built, predictor not reached", flush=True)
        return

    cls = bindcraft2.design_model_class()
    model = cls(presets=selected.design_models, data_dir=a.params,
                max_cache_size=16, num_recycle=settings["design_recycles"],
                models=selected.design_models,
                cyclic_offset_mode=resolve_cyclic_offset_mode(settings),
                subbatch_size=subbatch,
                attention_backend=settings.get("attention_backend", "auto"),
                use_cueq=bool(settings.get("use_cueq", False)), length_bucket_size=bucket,
                multi_chain_binders=multi_chain_binders, target_pad_length=0)

    with bindcraft2.fast_round() if a.memory == "fast" else _nullctx():
        t0 = time.time()
        predictions = pooled_stage_predictions(
            model, protein_states, int(settings.get("multitarget_filter_models", 1) or 1))
        secs = time.time() - t0

    # The gate's own metric, bound the way design_stage_filters binds it.
    state = "complex" if "complex" in predictions else next(
        s for s in predictions if s != BINDER_ALONE)
    row = {"target": a.target, "binder_aa": len(seq), "tokens_unpadded": tokens,
           "prediction_state": state, "secs": round(secs, 2),
           "binder_plddt": F.plddt_metric(protein_states, predictions,
                                          prediction_state=state, chain=binder_chain),
           "target_plddt": F.plddt_metric(protein_states, predictions,
                                          prediction_state=state, chain="target"),
           "complex_plddt": F.plddt_metric(protein_states, predictions, prediction_state=state),
           "i_pTM": F.iptm_metric(protein_states, predictions, prediction_state=state),
           "pTM": F.ptm_metric(protein_states, predictions, prediction_state=state),
           "min_plddt_screen": settings.get("min_plddt_screen"),
           "sequence": seq}
    row["passes_screen_gate"] = (row["binder_plddt"] is not None
                                 and row["binder_plddt"] >= float(row["min_plddt_screen"]))
    print(json.dumps(row), flush=True)
    name = f"refold_{a.target}{'_' + a.tag if a.tag else ''}.json"
    (out / name).write_text(json.dumps(row, indent=1))
    print(f"wrote {out / name}", flush=True)


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


if __name__ == "__main__":
    main()
