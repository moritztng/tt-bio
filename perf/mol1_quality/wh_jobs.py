"""Write one SPD pool job per paired quality case for Mol-1 on a lent Wormhole Galaxy chip.

    python3 wh_jobs.py ROW_DIR TREE CACHE BACKLOG_DIR [PRIO]

ROW_DIR holds inputs/ (quality_inputs.py tree, paired cases), kalign/bin (templates need kalign) and run_cell.sh (the SPD per-chip harness: chip flock,
lease, AICLK sampled during the run). A job runs its case's five seeds in order with run_tt.sh's recipe flags and
writes the upstream layout under ROW_DIR/tt-wh; a seed with rc=0 is skipped, so a recalled job can simply be requeued.
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
  d=$R/tt-wh/.runs/{name}/seed_$s; grep -qs rc=0 $d/rc && continue; mkdir -p $d
  bash $R/run_cell.sh $CHIP $d 5400 -- bash -c "cd $R/inputs && python -c 'from tt_bio.main import cli; cli()' predict {file} --model mol1 --seed $s --out_dir $R/tt-wh {flags}"
  r=$?; echo rc=$r > $d/rc; [ $r = 0 ] || rc=$r
done
exit $rc
"""


def main():
    row, tree, cache, backlog = map(Path, sys.argv[1:5]); prio = sys.argv[5] if len(sys.argv) > 5 else "4"
    idx = json.loads((row / "inputs" / "INDEX.json").read_text())
    head = set(idx.get("headline_only", ())); rec = idx.get("recipes", {})
    backlog.mkdir(parents=True, exist_ok=True); n = 0
    for cid, f in sorted(idx["cases"].items()):
        if cid in head:
            continue
        name = f[:-len(".json")]
        (backlog / f"{prio}-mol1-quality-{name}.sh").write_text(BODY.format(
            name=name, seeds=SEEDS, row=row, tree=tree, cache=cache, file=f,
            flags=FLAGS.get(rec.get(cid, "default"), "")))
        n += 1
    print(f"{n} jobs -> {backlog}")


if __name__ == "__main__":
    main()
