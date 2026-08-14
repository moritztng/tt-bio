"""BoltzGen seconds per design on one Blackhole p150a, pinned fixture bg_R3, shipped defaults.

Runs the shipped CLI (`tt-bio design --model boltzgen --steps design`) with nothing detuned:
sampling_steps stays at the 500 the design config ships, the protocol is the default
protein-anything, the production diffusion batch is 1, and both design checkpoints are used
the way the CLI uses them. `--diffusion_trace` and `--fast` ship off and are not passed.

The `--debug --log` stream prints `batch k/N` when design k completes, so the wall between
consecutive stamps is one whole design: trunk, all 500 denoising steps, post-processing and
writing. That matches the GPU arm's END-TO-END figure (H200 12.6471 s/design, B200 16.0916),
not its predict_step-only figure, and the state doc says so where the number is quoted.

Design 1 is dropped: it carries kernel compile for every new shape. The design that pays the
checkpoint switch is dropped too, detected from the "Switched checkpoint." line rather than
by picking the slowest, and its cost is reported. Six designs therefore leave four warm, the
same n the GPU arm reported.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:perf-page-design-models \
    PYTHONPATH=$(pwd) ~/.coworker/scripts/benchlock.sh perf-page-design-models -- \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/dspage/bg_page.py
"""
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

import gemmi

FIXTURE = "perf/dsfix/fixtures/bg_R3.yaml"
OUT = pathlib.Path("perf/dspage/results/bg_page.jsonl")
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
N_DESIGNS = 6
EXP_STEPS = 500                  # the shipped design.yaml default, asserted not passed
EXP_STAMPS = EXP_STEPS + 1       # the progress line prints k=0 as well as 1..500
EXP_CHAINS = {100, 414}          # 100 designed binder + 414 target residues
BATCH = re.compile(r"batch (\d+)/(\d+)")
DIFF = re.compile(r"diff (\d+)/(\d+)")
HOST, CARD, TTNN = "qb2", 0, "0.68.0"


def run(out_dir):
    cmd = [PY, "-u", "-m", "tt_bio.main", "design", FIXTURE,
           "--model", "boltzgen", "--steps", "design",
           "--num_designs", str(N_DESIGNS), "--out_dir", out_dir, "--debug", "--log"]
    env = dict(os.environ, TT_VISIBLE_DEVICES="0",
               TT_BIO_LEASE_HOLDER="worker:perf-page-design-models",
               PYTHONPATH=os.getcwd(), PYTHONUNBUFFERED="1")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    stamps, steps, diff_t, switched, cur, tail = {}, {}, {}, set(), 0, []
    for line in p.stdout:
        tail.append(line)
        if len(tail) > 80:
            tail.pop(0)
        mb = BATCH.search(line)
        if mb:
            cur = int(mb.group(1))
            stamps[cur] = time.time()
            continue
        md = DIFF.search(line)
        if md:
            steps.setdefault(cur + 1, []).append(int(md.group(2)))
            diff_t.setdefault(cur + 1, []).append(time.time())
        elif "Switched checkpoint" in line:
            switched.add(cur + 1)
    p.wait()
    return stamps, steps, diff_t, switched, time.time() - t0, p.returncode, "".join(tail)


def validate(out_dir):
    cifs = sorted(pathlib.Path(out_dir, "intermediate_designs").glob("*.cif"))
    bad, atoms = [], []
    if len(cifs) != N_DESIGNS:
        bad.append("%d designs written, expected %d" % (len(cifs), N_DESIGNS))
    for c in cifs:
        st = gemmi.read_structure(str(c))
        st.setup_entities()
        sizes = {len(ch) for ch in st[0]}
        na = sum(1 for ch in st[0] for r in ch for _ in r)
        nf = sum(1 for ch in st[0] for r in ch for a in r
                 if not all(abs(v) < 1e6 and v == v for v in (a.pos.x, a.pos.y, a.pos.z)))
        atoms.append(na)
        if sizes != EXP_CHAINS:
            bad.append("%s: chains %s != %s" % (c.name, sorted(sizes), sorted(EXP_CHAINS)))
        if not na:
            bad.append("%s: 0 atoms" % c.name)
        if nf:
            bad.append("%s: %d non-finite coords" % (c.name, nf))
    return (not bad), bad, atoms


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_text().strip():
        print("[bg] already measured, %s" % OUT, flush=True)
        return
    out_dir = "/tmp/bg_page"
    os.system("rm -rf %s" % out_dir)
    stamps, steps, diff_t, switched, wall, rc, tail = run(out_dir)
    if rc != 0 or len(stamps) < N_DESIGNS + 1:
        print("[bg] FAILED rc=%d, %d stamps\n%s" % (rc, len(stamps), tail[-3000:]), flush=True)
        sys.exit(1)

    # design k's wall is stamp k minus stamp k-1; stamp 0 is printed before design 1 starts.
    per = {k: stamps[k] - stamps[k - 1] for k in range(1, N_DESIGNS + 1)}
    dropped = {1: "cold, carries kernel compile"}
    for k in sorted(switched):
        if k in per:
            dropped[k] = "pays the checkpoint switch"
    warm = [per[k] for k in sorted(per) if k not in dropped]
    med = statistics.median(warm)
    bad_steps = {k: (sorted(set(v)), len(v)) for k, v in steps.items()
                 if sorted(set(v)) != [EXP_STEPS] or len(v) != EXP_STAMPS}
    # The denoising loop on its own, first to last step stamp, so the trunk's share of a
    # design is visible rather than inferred from another host's ladder.
    diff_span = {k: v[-1] - v[0] for k, v in diff_t.items() if len(v) > 1}
    warm_diff = [diff_span[k] for k in sorted(diff_span) if k not in dropped]
    ok, bad, atoms = validate(out_dir)
    rec = {
        "fixture": FIXTURE, "rung": "R3", "num_designs": N_DESIGNS,
        "sampling_steps_resolved": sorted({s for v in steps.values() for s in v}),
        "steps_seen_per_design": {str(k): len(v) for k, v in sorted(steps.items())},
        "steps_ok": not bad_steps, "steps_fail": bad_steps,
        "diffusion_batch": 1, "checkpoints": "diverse + adherence, CLI default",
        "per_design_s": {str(k): round(v, 3) for k, v in sorted(per.items())},
        "dropped": {str(k): v for k, v in sorted(dropped.items())},
        "n_warm": len(warm), "s_per_design": round(med, 3),
        "s_per_design_min": round(min(warm), 3), "s_per_design_max": round(max(warm), 3),
        "spread_pct": round(100 * (max(warm) - min(warm)) / med, 2),
        "designs_per_hour": round(3600 / med, 1),
        "diffusion_s_median": round(statistics.median(warm_diff), 3),
        "ms_per_step": round(1000 * statistics.median(warm_diff) / EXP_STEPS, 3),
        "trunk_and_post_s": round(med - statistics.median(warm_diff), 3),
        "proc_wall_s": round(wall, 1), "atoms": atoms,
        "output_ok": ok, "output_fail": bad,
        "host": HOST, "card": CARD, "ttnn": TTNN,
    }
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("[bg] %.3f s/design (n=%d warm, spread %.2f%%), steps_ok=%s output_ok=%s %s"
          % (rec["s_per_design"], len(warm), rec["spread_pct"], rec["steps_ok"], ok, bad),
          flush=True)


if __name__ == "__main__":
    main()
