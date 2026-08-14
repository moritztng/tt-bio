#!/usr/bin/env python3
"""Build a production-scale particle set by replicating RELION's own it012 particle table.

The crossover question this task exists to answer is a function of ONE variable: how many
particles the E-step loop runs over. Every other property of the job -- box size, sampling
order, current image size, the reference maps, the CTF model, the two half-sets -- has to be
held fixed, or the scaling fit measures a different job at every point and cannot be read.

Replicating `run_it012_data.star`'s 4,452 rows k times does exactly that. `_rlnRandomSubset`
travels with the row, so the two half-sets stay balanced and no particle crosses between them;
`_rlnImageName` is untouched, so every replica reads the same stack RELION already validated;
every per-particle prior (angles, offsets, norm, CTF) is the value RELION's own iteration 12
converged on, so iteration 13 starts from the same state at every scale.

What this deliberately does NOT give is a scientifically new answer: k copies of a particle carry
one particle's information, so the resolution a replicated run reports is not a resolution. This
set is an instrument for the wall-clock scaling law, and the doc says so where it is used.

Writes `Prod/run_it012_data_xK.star` and a matching `Prod/opt_xK.star` whose only difference from
RELION's own optimiser is the line naming the particle table.
"""
import os
import sys

JOB = "Refine3D/job019"
SRC_DATA = f"{JOB}/run_it012_data.star"
SRC_OPT = f"{JOB}/run_it012_optimiser.star"
OUT = "Prod"


def split_star(path):
    """Return (header lines up to and including the particle loop, particle rows)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    # the particle table is the last `loop_` block; its rows are everything after the last
    # `_rln<label> #n` line that follows `data_particles`.
    i_particles = max(i for i, l in enumerate(lines) if l.strip() == "data_particles")
    i_last_label = max(
        i for i, l in enumerate(lines) if i > i_particles and l.strip().startswith("_rln")
    )
    head = lines[: i_last_label + 1]
    rows = [l for l in lines[i_last_label + 1:] if l.strip()]
    return head, rows


def main():
    factors = [int(a) for a in sys.argv[1:]] or [1, 3, 8, 25]
    head, rows = split_star(SRC_DATA)
    print(f"source rows: {len(rows)}")
    os.makedirs(OUT, exist_ok=True)
    with open(SRC_OPT) as fh:
        opt = fh.read().splitlines()

    for k in factors:
        data_rel = f"{OUT}/run_it012_data_x{k}.star"
        with open(data_rel, "w") as fh:
            fh.write("\n".join(head) + "\n")
            for _ in range(k):
                fh.write("\n".join(rows) + "\n")
            fh.write(" \n")
        opt_rel = f"{OUT}/opt_x{k}.star"
        n_sub = 0
        with open(opt_rel, "w") as fh:
            for line in opt:
                if line.startswith("_rlnExperimentalDataStarFile"):
                    fh.write(f"_rlnExperimentalDataStarFile{' ' * 26}{data_rel}\n")
                    n_sub += 1
                else:
                    fh.write(line + "\n")
        assert n_sub == 1, f"expected exactly one data-star line, substituted {n_sub}"
        print(f"x{k}: {len(rows) * k} particles -> {data_rel}, {opt_rel}")


if __name__ == "__main__":
    main()
