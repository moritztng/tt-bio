#!/usr/bin/env python3
"""Issue #21 on a card, with the real stacks and without the stochastic design loop.

Two card campaigns (rounds 1 and 2, `state/bci-validation.md`) spent 3.5 h of chip time and
measured nothing about #21: every trajectory was terminated by BindCraft 2's own stage gates, so
no candidate ever reached validation and `accepted: 0` was the design loop talking, not the fix.
The reporter's complaint is about the VALIDATION ensemble, so this harness builds exactly the two
models a campaign builds and skips the dice in between:

    1. the design model, `MULTIMER_POOL`, built first, so `campaign_predictor` routes it to the
       card. One fold selects a checkpoint into the trunk pool and builds the device stacks --
       that is the state a campaign is in when validation runs, and it is the whole precondition
       for the bug.
    2. the validation model, `MONOMER_POOL`, built second, which `campaign_predictor` hands the
       host-JAX control factory. This is the fold the reporter's ten numbers come from.

Then it scores candidate sequences through that validation model and asks BindCraft 2's own
filters whether each one is ACCEPTED, which is the question #21 ends on.

The complex is 4ZQK: chain A is human PD-L1 (115 aa) and chain B is PD-1 (118 aa), a real binder
of that exact target, so a correct validation fold has something worth accepting rather than the
hash-derived sequence the CPU arms used. Candidates are that binder and point mutants of it, which
is what lets the run see the reporter's sharpest signal: a `Target_pLDDT` identical across
different sequences.

Arms, one process each because each takes its `tt_bio` from a different tree:

    fixed    wk/bci-integration, the fix in.
    prefix   the same tree with the fix reverted. The reporter's commit, everything else equal.
    control  the pre-fix tree with `--no-extra-msa`: the splice is installed and the Evoformer is
             still spliced, but nothing is swapped into the extra-MSA stack. A recovery here says
             the cause is that swap and not the pool the weights come from.
"""
import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

METRICS = ("Target_pLDDT", "pLDDT", "pTM", "i_pTM", "Interface_Residues", "Binder_RMSD")


def chain_sequence(pdb, chain):
    """One-letter sequence of a chain, read from SEQRES so it does not depend on the model."""
    three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
             "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
             "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    out = []
    for line in open(pdb):
        if line.startswith("SEQRES") and line[11] == chain:
            out += [three.get(code, "G") for code in line[19:].split()]
    return "".join(out)


def mutants(sequence, count):
    """`count` candidate sequences: the native binder, then deterministic point mutants of it.

    Different sequences are the reporter's sharpest signal. Their ten MPNN candidates differ from
    each other and still scored an identical `Target_pLDDT`, because the damage is in the TARGET's
    pair track. A mutant is enough to show that; what matters is that the sequences differ.
    """
    out = [sequence]
    swap = {"A": "V", "V": "A", "L": "I", "I": "L", "S": "T", "T": "S", "D": "E", "E": "D",
            "K": "R", "R": "K", "N": "Q", "Q": "N", "F": "Y", "Y": "F", "G": "A", "P": "A",
            "M": "L", "W": "F", "H": "N", "C": "S"}
    for n in range(1, count):
        chars = list(sequence)
        for i in range(n * 7, len(chars), 17):
            chars[i] = swap.get(chars[i], "A")
        out.append("".join(chars))
    return out


def states_for(pdb, target_chain, binder_sequence):
    import jax.numpy as jnp
    from bindcraft.protein import Protein, ResidueFlags

    chains = Protein.from_structure(pdb, chains=target_chain)
    target = chains[target_chain] if target_chain in chains else next(iter(chains.values()))
    binder = Protein.from_fasta(">A\n" + binder_sequence)
    binder = binder.replace(flags=jnp.full((len(binder),), int(ResidueFlags.DESIGN),
                                           dtype=jnp.uint8))
    return {"complex": {"binder": binder, "target": target}, "binder_alone": {"binder": binder}}


def metrics_for(states, predictions):
    from bindcraft.filters import REGISTERED_FILTER_METRICS as REG

    out = {}
    for name in METRICS:
        function = REG.get(name)
        if function is None:
            continue
        try:
            value = function(states, predictions)
        except Exception as error:
            out[name] = f"ERR:{type(error).__name__}"
            continue
        out[name] = None if value is None else float(value)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ttbio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdb", default="/home/ttuser/bci_struct/4zqk.pdb")
    ap.add_argument("--target-chain", dest="target_chain", default="A")
    ap.add_argument("--binder-chain", dest="binder_chain", default="B")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--recycle", type=int, default=3)
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--resident", type=int, default=1)
    ap.add_argument("--extra-msa", dest="extra_msa", action="store_true", default=True)
    ap.add_argument("--no-extra-msa", dest="extra_msa", action="store_false")
    args = ap.parse_args()

    sys.path.insert(0, args.ttbio)
    sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()

    import meter as M
    from bindcraft import campaign
    from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL
    from bindcraft.filters import evaluate_design_filters
    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.settings import build_design_settings, read_settings
    from tt_bio import bindcraft2

    import subprocess
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings_file = args.settings or "/home/ttuser/bcx_e2e/bc2/examples/pdl1.json"
    settings = cleaned_campaign_settings(read_settings(settings_file))
    # The campaign own filter set, resolved the way run_campaign resolves it. build_filters takes
    # the filters BLOCK, not the settings: handed the whole settings dict it returns {}, and every
    # candidate would then be accepted by an empty filter set. Measured, not assumed.
    filters = build_design_settings(settings).filters

    native = chain_sequence(args.pdb, args.binder_chain)
    sequences = mutants(native, args.candidates)
    sequence_parameters = dict(softmax_weight=1.0, one_hot_weight=1.0, temperature=0.01,
                               logit_scale=2.0)

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()
    started = time.time()
    stamp = {
        "arm": args.arm, "ttbio": args.ttbio,
        "ttbio_commit": subprocess.run(["git", "-C", args.ttbio, "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "pci": M.CLOCK.pci, "aiclk_sysfs": M.CLOCK.path,
        "pdb": args.pdb, "target_chain": args.target_chain, "binder_chain": args.binder_chain,
        "target_length": None, "binder_length": len(native),
        "candidates_asked": len(sequences), "extra_msa": args.extra_msa,
        "resident": args.resident, "recycle": args.recycle, "bucket": args.bucket,
        "settings_file": settings_file, "filters": sorted(filters),
        "started_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    print(json.dumps(stamp, indent=1), flush=True)

    results = []

    def close(error=None):
        M.CLOCK.stop()
        clocks = sorted(c for _, c, _ in M.CLOCK.samples)
        stamp.update({
            "error": error, "wall_seconds": round(time.time() - started, 1),
            "candidates": results,
            "accepted": sum(1 for r in results if r.get("accepted") is True),
            "aiclk_n": len(clocks), "aiclk_min": clocks[0] if clocks else None,
            "aiclk_med": clocks[len(clocks) // 2] if clocks else None,
            "aiclk_max": clocks[-1] if clocks else None,
            "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
        })
        for name in METRICS:
            seen = [r["metrics"][name] for r in results
                    if isinstance(r.get("metrics", {}).get(name), float)]
            if seen:
                stamp[f"{name}_min"], stamp[f"{name}_max"] = min(seen), max(seen)
        (out / "summary.json").write_text(json.dumps(stamp, indent=1))
        print(json.dumps(stamp, indent=1), flush=True)

    try:
        with bindcraft2.campaign_predictor(trunk="device", validation="jax",
                                           checkpoints=args.params, resident=args.resident,
                                           extra_msa=args.extra_msa, template=True,
                                           exact=False) as build:
            stamp["fast"], stamp["memory_mode"] = build.fast, build.memory.used
            states = states_for(args.pdb, args.target_chain, sequences[0])
            stamp["target_length"] = len(states["complex"]["target"])

            # 1. The design model. First build, so it goes on card: it selects a checkpoint into
            #    the trunk pool and builds the device stacks. Without this fold the pool holds
            #    nothing and the bug has nothing to read.
            design = campaign.AlphaFoldDesignModel(
                presets=MULTIMER_POOL[:1], models=MULTIMER_POOL[:1], data_dir=args.params,
                max_cache_size=16, num_recycle=args.recycle, length_bucket_size=args.bucket)
            design_started = time.time()
            design_predictions = design.predict(states, **sequence_parameters)
            stamp["design"] = {"seconds": round(time.time() - design_started, 1),
                               "models": list(MULTIMER_POOL[:1]),
                               "metrics": metrics_for(states, design_predictions)}
            swap = build.extra_msa
            stamp["after_design"] = {
                "extra_msa_calls": dict(getattr(swap, "calls", {}) or {}),
                "extra_msa_swapped": len(getattr(swap, "swapped", []) or []),
                "trunk_selected": getattr(getattr(swap, "pool", None), "selected", None),
            }
            print(json.dumps(stamp["design"]) + " " + json.dumps(stamp["after_design"]),
                  flush=True)

            # 2. The validation model. Second build, so `campaign_predictor` hands it the host
            #    control factory: BindCraft 2's own trunk, which is what the user is told runs.
            validation = campaign.AlphaFoldDesignModel(
                presets=MONOMER_POOL, data_dir=args.params, max_cache_size=16,
                num_recycle=args.recycle, dropout=False, length_bucket_size=args.bucket)

            for index, sequence in enumerate(sequences):
                candidate_states = states_for(args.pdb, args.target_chain, sequence)
                before = len(getattr(swap, "swapped", []) or [])
                calls_before = dict(getattr(swap, "calls", {}) or {})
                at = time.time()
                predictions = validation.predict(candidate_states, **sequence_parameters)
                values = metrics_for(candidate_states, predictions)
                try:
                    passed, filter_values = evaluate_design_filters(filters, candidate_states,
                                                                    predictions)
                except Exception as error:
                    passed, filter_values = f"ERR:{type(error).__name__}: {error}", {}
                results.append({
                    "candidate": index, "sequence": sequence,
                    "mutations_from_native": sum(1 for a, b in zip(sequence, native) if a != b),
                    "metrics": values,
                    "accepted": passed is True,
                    "failed_filters": passed if isinstance(passed, list) else [],
                    "filter_values": {k: (float(v) if isinstance(v, (int, float)) else v)
                                      for k, v in dict(filter_values).items()},
                    "device_extra_msa_stacks_built_during_this_fold":
                        len(getattr(swap, "swapped", []) or []) - before,
                    "device_extra_msa_calls_during_this_fold":
                        {k: v - calls_before.get(k, 0)
                         for k, v in dict(getattr(swap, "calls", {}) or {}).items()},
                    "seconds": round(time.time() - at, 1),
                })
                print(json.dumps(results[-1]), flush=True)
            close()
    except BaseException as error:
        close(repr(error))
        raise


if __name__ == "__main__":
    main()
