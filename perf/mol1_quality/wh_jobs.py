"""Write one SPD pool job per paired quality case for Mol-1 on a lent Wormhole Galaxy chip.

    python3 wh_jobs.py ROW_DIR TREE CACHE BACKLOG_DIR [PRIO] [SEEDS] [OUT]

SEEDS (default all five, e.g. "42 66") lets seeds 42+66 go at a higher priority than 101/2024/8888 (D-orch-36);
the job file name carries the first seed when SEEDS is given, so both halves of a case can sit in the queue.
OUT (default tt-wh) is the results dir under ROW_DIR; a regrade on another tree folds into its own and names its jobs after it.

ROW_DIR holds inputs/ (quality_inputs.py tree, paired cases), kalign/bin (templates need kalign) and run_cell.sh (the SPD per-chip harness: chip flock,
lease, AICLK sampled during the run). A job runs its case's five seeds in order with run_tt.sh's recipe flags and
writes the upstream layout under ROW_DIR/OUT; a seed with rc=0 is skipped, so a recalled job can simply be requeued.
"""
import json, sys
from pathlib import Path

SEEDS = "42 66 101 2024 8888"
FLAGS = {"use_msa_false": "--single_sequence", "use_template_true": "--use_template",
         "use_rna_msa_true": "--use_rna_msa"}
BODY = """# mol1-quality WH: {name}, seeds {seeds}, upstream defaults (20 cycles, 200 steps, 5 samples)
# mem: 60
# threads: 4
# cap: 14400
R={row}
export PATH=$R/kalign/bin:$PATH TREE={tree} TT_BIO_CACHE={cache} BOLTZ_CACHE={cache} SHARE=4
rc=0
for s in {seeds}; do
  d=$R/{out}/.runs/{name}/seed_$s; grep -qs rc=0 $d/rc && continue; mkdir -p $d
  bash $R/run_cell.sh $CHIP $d 5400 -- bash -c "cd $R/inputs && python -c 'from tt_bio.main import cli; cli()' predict {file} --model mol1 --seed $s --out_dir $R/{out} {flags}"
  r=$?; echo rc=$r > $d/rc; [ $r = 0 ] || rc=$r
done
exit $rc
"""


def main():
    row, tree, cache, backlog = map(Path, sys.argv[1:5]); prio = sys.argv[5] if len(sys.argv) > 5 else "4"
    seeds = sys.argv[6] if len(sys.argv) > 6 else SEEDS; out = sys.argv[7] if len(sys.argv) > 7 else "tt-wh"; tag = ("" if seeds == SEEDS else f"-s{seeds.split()[0]}") + ("" if out == "tt-wh" else f"-{out}")
    idx = json.loads((row / "inputs" / "INDEX.json").read_text())
    head = set(idx.get("headline_only", ())); rec = idx.get("recipes", {})
    backlog.mkdir(parents=True, exist_ok=True); n = 0
    for cid, f in sorted(idx["cases"].items()):
        if cid in head:
            continue
        name = f[:-len(".json")]
        (backlog / f"{prio}-mol1-quality-{name}{tag}.sh").write_text(BODY.format(
            name=name, seeds=seeds, row=row, out=out, tree=tree, cache=cache, file=f,
            flags=FLAGS.get(rec.get(cid, "default"), "")))
        n += 1
    print(f"{n} jobs -> {backlog}")


if __name__ == "__main__":
    main()
