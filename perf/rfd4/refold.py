"""Refold a directory of RFD4 designs on Tenstorrent and leave the cache evaluate reads.

    python3 perf/rfd4/refold.py --inputs runs/tt --out runs/tt_eval --family unconditional

Upstream's `OracleMetric.compute` folds a design only when its confidence JSON is missing from the
design's sample directory and otherwise scores what is there
(`rfproteina/metrics/oracles/oracle_metric.py:471-520`), so writing that pair is the whole seam
between a Tenstorrent refold and every oracle-bound metric column:

    <out>/evaluation_samples/<example_id>/<example_id>_<oracle_name>_confidences.json
    <out>/evaluation_samples/<example_id>/<example_id>_<oracle_name>_refolded.cif

`example_id` is computed by `tt_bio.rfd4.evaluate.example_id` from the same `--inputs` root the
evaluate run will use, so the two agree by construction rather than by convention. Run this, then

    tt-bio evaluate --model rfd4 <inputs> --metrics_tags <tags> --out_dir <out>

and the oracle columns, the refolding RMSDs and the pass criteria all come off these files.

The oracle is OpenFold3 through `tt_bio/rfd4/oracle_of3.py`, which is upstream's own second
oracle config (`configs/metrics/refolding_oracles/of3.yaml`, the same `AF3Oracle` class on
converted OpenFold3 weights) at upstream's protocol: 5 recycles, 200 rollout steps, 1 diffusion
sample. AF3 itself is not redistributable, so of3 is the oracle tt-bio can serve; every number it
produces is an OF3 number and says so.

One deliberate departure from that config: upstream folds the oracle single-sequence, which suits a
monomer design whose chain has no alignment to find. A binder design is a complex, and its target is
a natural protein. Folded single-sequence, OpenFold3 cannot fold the 500-residue target of this set
at all (plDDT 27-35 across all 128 designs), and an interface metric over a target that did not fold
is not a measurement: the in-silico success rate came back a clean 0 %. So the target chain folds
with its alignment and the designed chain folds single-sequence, which is what `--msa-dir` now means
and what `report_msa` prints every run. `--single-sequence` restores upstream's protocol exactly.

Resumable by design: a design whose confidences JSON exists is skipped, and the JSON is written
after the refold, so an interrupted run never leaves a half-written cache entry behind. The clock
is sampled during the run (`perf/clocksample.py`) because a fold time taken below ~1200 MHz on
Blackhole is an artifact, not a measurement.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1].parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from perf import clocksample  # noqa: E402

#: family -> (oracle config key, the oracle_name that keys its cache files). Upstream's
#: `configs/metrics/refolding_oracles/of3.yaml`; the key prefixes the columns, the name the files.
FAMILY_ORACLE = {
    "unconditional": ("oracle", "of3"),
    "enzyme": ("oracle", "of3"),
    "motif_scaffold": ("oracle", "of3"),
    "small_molecule": ("oracle", "of3"),
    "partial_diffusion": ("oracle", "of3"),
    "binder": ("oracle_ppi", "of3_ppi"),
    "antibody_cdr": ("oracle_ppi", "of3_ppi"),
    "nanobody_cdr": ("oracle_ppi", "of3_ppi"),
    "nucleic_acid": ("oracle_dna", "of3_dna"),
}


def designs(inputs: pathlib.Path) -> list[pathlib.Path]:
    """The structures evaluate will grade, in its own order."""
    from tt_bio.rfd4.evaluate import structure_files

    return [p for p in structure_files(inputs) if "_input" not in p.name and "_refolded" not in p.name]


def chain_of_interest(structure_text: str, requested: str | None) -> str:
    """The chain the `*_coi` reward keys are about.

    A monomer has exactly one chain and needs no argument; anything with more than one does, and
    guessing it would quietly grade the wrong chain.
    """
    from tt_bio.rfd4.oracle_of3 import structure_chains

    chains = structure_chains(structure_text)
    if requested is not None:
        if requested not in chains:
            raise SystemExit(f"--chain {requested!r} is not in the design (chains {chains})")
        return requested
    if len(chains) != 1:
        raise SystemExit(f"design has chains {chains}; name the designed one with --chain")
    return chains[0]


def report_msa(structure_text: str, msa_dir: str, single_sequence: bool) -> dict[str, int]:
    """Print the alignment depth each chain will fold with, and refuse a silent no-MSA run.

    Returns {chain: rows}, 0 meaning the chain folds single-sequence. A designed binder has no
    alignment and should have none; a natural target without one is the failure this guard exists
    for, and it is invisible in the exit code.
    """
    from tt_bio.cache import seq_hash
    from tt_bio.rfd4.oracle_of3 import chain_sequence, structure_chains

    depth = {}
    for chain in structure_chains(structure_text):
        a3m = pathlib.Path(msa_dir).expanduser() / f"{seq_hash(chain_sequence(structure_text, chain))}.a3m"
        cached = not single_sequence and a3m.is_file()
        depth[chain] = sum(1 for ln in a3m.read_text().splitlines() if ln.startswith(">")) if cached else 0
    mode = "single sequence (upstream of3 protocol)" if single_sequence else f"alignments from {msa_dir}"
    print("oracle MSA: " + mode + "; " +
          ", ".join(f"chain {c} {n} rows" if n else f"chain {c} single-sequence"
                    for c, n in depth.items()), flush=True)
    if not single_sequence and not any(depth.values()):
        raise SystemExit(
            f"no chain has a cached alignment under {msa_dir}, so every chain would fold "
            "single-sequence while the run claims otherwise. Stage the target's a3m as "
            "<sha256(sequence)[:16]>.a3m there, or pass --single-sequence to mean it.")
    return depth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="the design root the evaluate run will take")
    ap.add_argument("--out", required=True, help="the evaluate out_dir; the cache lands under it")
    ap.add_argument("--family", required=True, choices=sorted(FAMILY_ORACLE),
                    help="which oracle key the family's columns come from")
    ap.add_argument("--chain", default=None, help="chain of interest (required for a complex)")
    ap.add_argument("--limit", type=int, default=None, help="refold at most this many designs")
    ap.add_argument("--shards", type=int, default=1,
                    help="split the work this many ways so several cards can run the same root")
    ap.add_argument("--shard", type=int, default=0, help="which stride of --shards this run takes")
    ap.add_argument("--deadline", type=int, default=None,
                    help="unix epoch: start no refold that could run past it; exit 3 with work left")
    ap.add_argument("--fold-s", type=int, default=600,
                    help="worst-case seconds for one refold, the margin --deadline keeps")
    ap.add_argument("--recycles", type=int, default=5)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--msa-dir", default="/tmp/rfd4_refold_msa")
    ap.add_argument("--single-sequence", action="store_true",
                    help="upstream's of3 protocol: fold every chain without an alignment, even one "
                         "cached under --msa-dir. On a natural target this is what makes the oracle "
                         "unable to fold it at all; see the module docstring.")
    ap.add_argument("--templates", default="/tmp/rfd4_refold_templates")
    ap.add_argument("--record", default=None, help="jsonl, one line per refold")
    args = ap.parse_args(argv)

    from tt_bio.rfd4.evaluate import example_id

    inputs = pathlib.Path(args.inputs).resolve()
    samples = pathlib.Path(args.out).resolve() / "evaluation_samples"
    key, oracle_name = FAMILY_ORACLE[args.family]

    todo = []
    for path in designs(inputs):
        eid = example_id(path, inputs)
        out_dir = samples / eid
        if (out_dir / f"{eid}_{oracle_name}_confidences.json").exists():
            continue
        todo.append((path, eid, out_dir))
    # The skip above already makes a run idempotent; the stride is what keeps two cards working the
    # same root from both starting at design 1.
    if args.shards > 1:
        todo = todo[args.shard :: args.shards]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} design(s) to refold into {samples}", flush=True)
    if not todo:
        return 0
    # Before the 2.3 GB oracle load, not after the first design: at a window edge the load alone can
    # outlast the booking.
    if args.deadline and time.time() + args.fold_s > args.deadline:
        print(f"deadline: stopping with {len(todo)} design(s) left", flush=True)
        return 3
    # Before the oracle loads 2.3 GB of weights, not after: an unwritable record path cost a
    # full model load and left nothing behind.
    if args.record:
        pathlib.Path(args.record).parent.mkdir(parents=True, exist_ok=True)

    # Which chain the oracle will actually align, printed before the weights load and checked while
    # it is still free to stop. A cached alignment is read only when single_sequence is off
    # (`tt_bio/worker.py::_build_chain_specs`), so --msa-dir on its own is not enough, and a refold
    # that silently falls back to one row still exits 0: on this set every such design came back at
    # plDDT 27-35 and an in-silico success rate read a clean 0 %.
    report_msa(todo[0][0].read_text(), args.msa_dir, args.single_sequence)

    # Imported here, not at the top of main: it pulls ttnn in, and the paths above (nothing to do, or a
    # window too short to start) must stay runnable off a card.
    from tt_bio.rfd4.oracle import OracleRequest
    from tt_bio.rfd4.oracle_of3 import OpenFold3Oracle

    oracle = OpenFold3Oracle(msa_dir=args.msa_dir, template_structures=args.templates,
                             name=key, oracle_name=oracle_name, recycles=args.recycles,
                             sampling_steps=args.sampling_steps,
                             single_sequence=args.single_sequence)
    oracle.setup()

    record = open(args.record, "a") if args.record else None
    failures = 0
    with clocksample.during(period=2.0) as clk:
        for i, (path, eid, out_dir) in enumerate(todo, 1):
            if args.deadline and time.time() + args.fold_s > args.deadline:
                print(f"deadline: stopping with {len(todo) - i + 1} design(s) left", flush=True)
                if record:
                    record.close()
                return 3
            text = path.read_text()
            job = OracleRequest(structure=text, example_id=eid,
                                chain_of_interest=chain_of_interest(text, args.chain),
                                out_dir=out_dir)
            t0 = time.time()
            try:
                metrics = oracle.score_one(job)
                err = None
            except Exception as exc:  # one bad design must not end the run
                metrics, err = {}, f"{type(exc).__name__}: {exc}"
                failures += 1
            dt = time.time() - t0
            print(f"[{i}/{len(todo)}] {eid} {dt:.1f}s "
                  f"{'FAILED ' + err if err else 'plddt=%.1f' % metrics.get('overall_plddt', float('nan'))}",
                  flush=True)
            if record:
                record.write(json.dumps({"example_id": eid, "design": str(path), "s": round(dt, 2),
                                         "oracle": oracle_name, "error": err,
                                         "overall_plddt": metrics.get("overall_plddt")}) + "\n")
                record.flush()
    summary = clk.summary().get(0)
    if record:
        record.close()
    print(f"aiclk during the run: {summary}", flush=True)
    if summary and summary.get("median", 0) < 1200:
        print("CLOCK SUSPECT: the timings above are artifacts, the refolds are not", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
