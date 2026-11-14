#!/usr/bin/env python3
"""Issue #21 on a card: one real campaign per arm, and the accepted count it produces.

The CPU arms (`refold21.py`) proved the mechanism and returned the reporter's three metrics, but
they stand the card's extra-MSA stack in with an identity map, so they cannot say how much of the
gap is the wrong WEIGHTS and how much the wrong STACK, and they cannot say whether a design is
ACCEPTED. That is this harness, and it needs a chip.

Three arms, one per process because each takes its `tt_bio` from a different tree:

  fixed    the worktree at the fix. `validation` stays at its `"jax"` default, which is the route
           the reporter ran and the only one an accepted count should be quoted from.
  prefix   `--ttbio /home/ttuser/bci_prefix` (f25be1624), the commit they hit. Same settings, same
           seed. This is the reporter's 0-of-10 reproduced on a real card rather than on CPU.
  control  the pre-fix tree with `--no-extra-msa`. The splice is installed and the Evoformer is
           still spliced, but no extra-MSA stack is swapped in, so a recovery here says the cause
           is the extra-MSA swap specifically and not the pool the weights come from.

Every arm writes `summary.json` beside the campaign: the per-candidate metrics straight out of
BindCraft 2's own refolded table, the accepted count out of its ranked table, and the card's AICLK
sampled every second DURING the run. A campaign that dies still writes what it had.
"""
import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

METRICS = ("Target_pLDDT", "pLDDT", "pTM", "i_pTM", "Interface_Residues", "Binder_RMSD")


def rows(path):
    try:
        with open(path) as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def candidate_table(project):
    """The reporter's own table: one row per MPNN candidate the validation ensemble scored."""
    from bindcraft.campaign_output import REFOLD_STAGE, accepted_table, stage_table

    out = {"candidates": [], "accepted": len(rows(accepted_table(project)))}
    for row in rows(stage_table(project, REFOLD_STAGE)):
        kept = {"design": row.get("design", "")}
        for name in METRICS:
            value = row.get(name)
            if value not in (None, ""):
                try:
                    kept[name] = float(value)
                except ValueError:
                    kept[name] = value
        out["candidates"].append(kept)
    for name in METRICS:
        seen = [c[name] for c in out["candidates"] if isinstance(c.get(name), float)]
        if seen:
            out[f"{name}_min"], out[f"{name}_max"] = min(seen), max(seen)
            out[f"{name}_mean"] = round(sum(seen) / len(seen), 6)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="a label for the summary: fixed, prefix, control")
    ap.add_argument("--ttbio", required=True,
                    help="the tree tt_bio comes from, so before and after are two commits")
    ap.add_argument("--out", required=True, help="project folder, one per arm")
    ap.add_argument("--settings", default=None, help="defaults to bc2's own examples/pdl1.json")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--mpnn", default=None)
    ap.add_argument("--binder", type=int, default=90, help="the reporter pinned theirs to 90")
    ap.add_argument("--designs", type=int, default=10, help="candidates to validate, theirs was 10")
    ap.add_argument("--max-trajectories", dest="max_trajectories", type=int, default=1)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--resident", type=int, default=1, help="the reporter ran resident=1")
    ap.add_argument("--validation", default="jax", choices=("jax", "device"))
    ap.add_argument("--extra-msa", dest="extra_msa", action="store_true", default=True)
    ap.add_argument("--no-extra-msa", dest="extra_msa", action="store_false",
                    help="the control that separates the wrong stack from the wrong weights")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V")
    args = ap.parse_args()

    sys.path.insert(0, args.ttbio)
    sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))
    sys.path.insert(0, str(ROOT / "perf" / "bcx_predictor"))

    # p300: the mesh descriptor is set by tt-bio's CLI entry points, and anything reaching the
    # device path without it gets a TT_FATAL on a single chip (`tt_bio/tenstorrent.py`).
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()

    import bc2_state as B
    import meter as M
    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.settings import parse_setting_overrides, read_settings
    from tt_bio import bindcraft2

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    settings_file = args.settings or os.path.join(B.BC2, "examples", "pdl1.json")
    overrides = [f"campaign_seed={args.seed}",
                 f"max_trajectories={args.max_trajectories}",
                 f"number_of_final_designs={args.designs}",
                 f"binder_lengths=[{args.binder}]",
                 f"project_folder={project}",
                 "compile_next_length=0"] + args.sets
    settings = cleaned_campaign_settings(read_settings(settings_file,
                                                       parse_setting_overrides(overrides)))

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()
    started = time.time()
    stamp = {"arm": args.arm, "ttbio": args.ttbio,
             "ttbio_commit": subprocess.run(["git", "-C", args.ttbio, "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip(),
             "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "pci": M.CLOCK.pci, "aiclk_sysfs": M.CLOCK.path,
             "validation": args.validation, "extra_msa": args.extra_msa,
             "resident": args.resident, "binder": args.binder, "designs": args.designs,
             "max_trajectories": args.max_trajectories, "seed": args.seed,
             "settings_file": settings_file, "project": project,
             "started_utc": time.strftime("%FT%TZ", time.gmtime())}
    print(json.dumps(stamp, indent=1), flush=True)

    def close(error=None):
        """Written before the predictor's teardown: `close_device` can abort the process and
        take the record with it (issue #20)."""
        M.CLOCK.stop()
        clocks = sorted(c for _, c, _ in M.CLOCK.samples)
        stamp.update({"error": error, "wall_seconds": round(time.time() - started, 1),
                      "trajectories_returned": trajectories,
                      "aiclk_n": len(clocks),
                      "aiclk_min": clocks[0] if clocks else None,
                      "aiclk_med": clocks[len(clocks) // 2] if clocks else None,
                      "aiclk_max": clocks[-1] if clocks else None,
                      "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
        try:
            stamp.update(candidate_table(project))
        except Exception as exc:                       # a table that is not there says so, it
            stamp["table_error"] = repr(exc)           # does not erase the clock and the stamp
        pathlib.Path(project, "summary.json").write_text(json.dumps(stamp, indent=1))
        print(json.dumps(stamp, indent=1), flush=True)

    trajectories = None
    mpnn = args.mpnn or os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn",
                                     "weights_neutral")
    with bindcraft2.campaign_predictor(trunk="device", validation=args.validation,
                                       checkpoints=args.params, resident=args.resident,
                                       extra_msa=args.extra_msa, template=True,
                                       exact=False) as build:
        stamp["fast"], stamp["memory"] = build.fast, build.memory.used
        try:
            trajectories = bindcraft2.run_campaign(settings, project, trajectories_per_card=1,
                                                   af2_weights=args.params, mpnn_weights=mpnn)
        except BaseException as exc:
            close(repr(exc))
            raise
        close()


if __name__ == "__main__":
    main()
