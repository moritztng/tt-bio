#!/usr/bin/env python3
"""A real BindCraft 2 gradient trajectory on the card, for one researcher input.

Not a perf harness. It answers one question per input: does the campaign this file and these
settings describe actually run on a Blackhole chip, and are the hotspots it steers by the
residues the caller named. `--rounds` is the whole screen stage and every other stage is zero,
so a case is a few real gradient rounds rather than a campaign; `trajectory_only` keeps MPNN and
the validation ensemble out, which is where the time would otherwise go.

    round.py --case gap --rounds 2 --card 3

Every case writes one JSON line to out/rounds.jsonl: the tokens it ran at, the rounds it
completed, the hotspots on the target as prepared, the loss at the first and last round, and the
card's AICLK sampled every second DURING the run, because a fold time without its clock is not a
measurement on this chip.
"""
import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))
sys.path.insert(0, str(ROOT / "perf" / "bcx_predictor"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IN = HERE / "inputs"
BC2 = pathlib.Path("/home/ttuser/bcx_e2e/bc2")
AF2 = "/home/ttuser/bcx_e2e/af2_params"


def target(path, **kw):
    return [{"name": "T", "target_path": str(IN / path), **kw}]


#: The inputs worth a card. Every one of them is a file or a setting a researcher brings, and
#: each is in the matrix `probe.py` resolves on the host; this is the leg that proves the
#: campaign runs on it.
CASES = {
    "base": {"target": "hPDL1", "binder_lengths": [60]},
    "gap": {"targets": target("gap.pdb", hotspots="54,56,115"), "binder_lengths": [60]},
    "gap-range": {"targets": target("gap.pdb", hotspots="54-70"), "binder_lengths": [60]},
    "ligands": {"targets": target("ligands.pdb", hotspots="54,56,66,115"),
                "binder_lengths": [60]},
    "mse": {"targets": target("mse.pdb", hotspots="54,56,66,115"), "binder_lengths": [60]},
    "altloc": {"targets": target("altloc.pdb", hotspots="54,56,66,115"), "binder_lengths": [60]},
    "mmcif": {"targets": target("hPDL1.cif", hotspots="54,56,66,115"), "binder_lengths": [60]},
    "twochain": {"targets": target("twochain.pdb", chains="A,B", hotspots="A67,A96,B125"),
                 "binder_lengths": [60]},
    "onechain": {"targets": target("twochain.pdb", chains="A", hotspots="67,96"),
                 "binder_lengths": [60]},
    "peptide-short": {"target": "hPDL1", "modality": "peptide", "binder_lengths": [12]},
    "binder-long": {"target": "hPDL1", "binder_lengths": [180]},
    "length-range": {"target": "hPDL1", "binder_lengths": [60, 61]},
    "no-hotspots": {"targets": target("hPDL1.cif"), "binder_lengths": [60]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--card", default=os.environ.get("TT_VISIBLE_DEVICES", "3"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--real", action="store_true",
                    help="a whole campaign on this input rather than a few gradient rounds: "
                         "every design stage, ProteinMPNN, the validation ensemble and the "
                         "acceptance filters, on the shipped five-checkpoint multimer pool and "
                         "the shipped trajectory default. This is the leg that says the designs "
                         "are sane, not just that the input runs")
    ap.add_argument("--max-trajectories", dest="max_trajectories", type=int, default=1)
    ap.add_argument("--designs", type=int, default=1)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    project = out / f"run_{args.case}"
    if project.exists():
        import shutil
        shutil.rmtree(project)
    project.mkdir(parents=True)

    from tt_bio import bindcraft2
    bindcraft2.pin_card(args.card)

    import meter as M
    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.settings import build_design_settings, read_settings
    from bindcraft.protein import ResidueFlags, has_residue_flag
    from bindcraft.protein_preparation import prepare_targets
    import numpy as np

    request = {"campaign_name": f"bgx-{args.case}", "project_folder": str(project),
               "campaign_seed": args.seed, "max_trajectories": args.max_trajectories,
               "number_of_final_designs": args.designs, "compile_next_length": 0}
    if not args.real:
        # A few real gradient rounds: the whole screen stage and nothing after it, on one
        # monomer trunk, the same reduction `perf/bcx_round/run_round.py` makes. Five multimer
        # checkpoints and a validation ensemble would measure the pool, not the input.
        request.update({"trajectory_only": True, "validation_model": "monomer",
                        "design_models": ["model_1_ptm"], "validation_models": ["model_2_ptm"],
                        "screen_steps": args.rounds, "refine_steps": 0, "anneal_steps": 0,
                        "harden_steps": 0, "mutate_steps": 0})
    request.update(CASES[args.case])
    if "modality" not in request:
        request["modality"] = "binder"
    settings = cleaned_campaign_settings(read_settings(str(BC2 / "examples/pdl1.json"), request))

    import bindcraft.campaign as campaign
    if not args.real:
        campaign.MULTIMER_POOL = ("model_1_ptm", "model_2_ptm")

    prepared = prepare_targets(build_design_settings(settings))
    record = {"case": args.case, "rounds_asked": args.rounds, "card": args.card,
              "targets": {name: {"residues": len(p),
                                 "hotspots": [int(n) for n in np.asarray(p.residue_index)[
                                     np.asarray(has_residue_flag(p.flags, ResidueFlags.HOTSPOT))]]}
                          for name, p in prepared.items()},
              "binder_lengths": list(CASES[args.case].get("binder_lengths", ()))}

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()
    t0 = time.time()
    mpnn = str(BC2 / "bindcraft/weights/proteinmpnn/weights_neutral")
    try:
        with bindcraft2.campaign_predictor(card=args.card):
            # The real leg leaves `trajectories_per_card` off, which is the shipped default's own
            # resolution; the round leg pins 1, because two trajectories would interleave the
            # thing being checked.
            per_card = {} if args.real else {"trajectories_per_card": 1}
            trajectories = bindcraft2.run_campaign(
                settings, str(project), **per_card, af2_weights=AF2,
                **({"mpnn_weights": mpnn} if args.real else {}))
        record["outcome"], record["trajectories"] = "RAN", trajectories
    except Exception as error:
        record["outcome"] = f"{type(error).__name__}"
        record["error"] = str(error)[:600]
    t1 = time.time()
    M.CLOCK.stop()
    record["wall_s"] = round(t1 - t0, 1)
    record["clock"] = M.CLOCK.window(t0, t1)
    csv = project / "trajectories.csv"
    record["trajectories_csv"] = csv.read_text().strip().splitlines()[-2:] if csv.is_file() else []
    accepted = project / "accepted.csv"
    record["accepted"] = (len(accepted.read_text().strip().splitlines()) - 1
                          if accepted.is_file() else 0)
    record["real"] = bool(args.real)
    with open(out / "rounds.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return 0 if record["outcome"] == "RAN" else 1


if __name__ == "__main__":
    sys.exit(main())
